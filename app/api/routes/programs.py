from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db, require_roles
from app.models.program import Program
from app.models.program_structure import ProgramTerm
from app.models.user import User, UserRole
from app.schemas.program import ProgramCreate, ProgramOut, ProgramUpdate
from app.services.notifications import notify_admin_update
from app.services.reevaluation import record_curriculum_change

router = APIRouter()


@router.get("/", response_model=list[ProgramOut])
def list_programs(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> list[ProgramOut]:
    return list(db.execute(select(Program).order_by(Program.code.asc())).scalars())


@router.post("/", response_model=ProgramOut, status_code=status.HTTP_201_CREATED)
def create_program(
    payload: ProgramCreate,
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> ProgramOut:
    existing = db.execute(select(Program).where(Program.code == payload.code)).scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Program code already exists")
    program = Program(**payload.model_dump())
    db.add(program)
    notify_admin_update(
        db,
        title="Program Created",
        message=f"{current_user.name} created program {payload.code} ({payload.name}).",
        actor_user_id=current_user.id,
    )
    db.commit()
    db.refresh(program)
    return program


@router.put("/{program_id}", response_model=ProgramOut)
def update_program(
    program_id: str,
    payload: ProgramUpdate,
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> ProgramOut:
    program = db.get(Program, program_id)
    if program is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Program not found")

    data = payload.model_dump(exclude_unset=True)
    if "code" in data:
        existing = db.execute(select(Program).where(Program.code == data["code"], Program.id != program_id)).scalar_one_or_none()
        if existing:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Program code already exists")

    for key, value in data.items():
        setattr(program, key, value)
    if data:
        term_numbers = list(
            db.execute(
                select(ProgramTerm.term_number).where(ProgramTerm.program_id == program_id)
            ).scalars()
        )
        if not term_numbers:
            term_numbers = [None]
        for term_number in term_numbers:
            record_curriculum_change(
                db,
                program_id=program_id,
                term_number=term_number,
                change_type="program_updated",
                entity_type="program",
                entity_id=program_id,
                description=f"Program metadata updated for {program.code}",
                details={"changed_fields": sorted(data.keys())},
                triggered_by=current_user,
            )
        notify_admin_update(
            db,
            title="Program Updated",
            message=f"{current_user.name} updated program {program.code}.",
            actor_user_id=current_user.id,
        )
    db.commit()
    db.refresh(program)
    return program


@router.delete("/{program_id}")
def delete_program(
    program_id: str,
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> dict:
    program = db.get(Program, program_id)
    if program is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Program not found")
    term_numbers = list(
        db.execute(
            select(ProgramTerm.term_number).where(ProgramTerm.program_id == program_id)
        ).scalars()
    )
    if not term_numbers:
        term_numbers = [None]
    for term_number in term_numbers:
        record_curriculum_change(
            db,
            program_id=program_id,
            term_number=term_number,
            change_type="program_deleted",
            entity_type="program",
            entity_id=program_id,
            description=f"Program {program.code} deleted",
            details={},
            triggered_by=current_user,
        )
    notify_admin_update(
        db,
        title="Program Deleted",
        message=f"{current_user.name} deleted program {program.code}.",
        actor_user_id=current_user.id,
    )
    db.delete(program)
    db.commit()
    return {"success": True}
