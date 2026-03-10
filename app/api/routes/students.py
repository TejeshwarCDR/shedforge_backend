from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_roles
from app.core.security import get_password_hash
from app.models.program import Program
from app.models.user import User, UserRole
from app.schemas.user import StudentCreate, StudentListOut, StudentUpdate

router = APIRouter()


@router.get("/students", response_model=list[StudentListOut])
def list_students(
    program_id: str | None = Query(default=None),
    semester_number: int | None = Query(default=None, ge=1, le=20),
    section_name: str | None = Query(default=None),
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> list[StudentListOut]:
    statement = select(User).where(User.role == UserRole.student)
    if program_id:
        statement = statement.where(User.program_id == program_id)
    if semester_number is not None:
        statement = statement.where(User.semester_number == semester_number)
    if section_name:
        statement = statement.where(User.section_name == section_name.strip())
    students = db.execute(
        statement.order_by(User.semester_number.asc(), User.section_name.asc(), User.name.asc())
    ).scalars().all()
    return list(students)


def _resolve_program_id(db: Session, requested_program_id: str | None) -> str:
    if requested_program_id:
        program = db.get(Program, requested_program_id)
        if program is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Program not found")
        return program.id

    default_program = db.execute(select(Program).order_by(Program.created_at.asc())).scalars().first()
    if default_program is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No program available. Create a program before managing students.",
        )
    return default_program.id


@router.post("/students", response_model=StudentListOut, status_code=status.HTTP_201_CREATED)
def create_student(
    payload: StudentCreate,
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> StudentListOut:
    normalized_email = payload.email.strip().lower()
    existing = db.execute(select(User).where(User.email == normalized_email)).scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Student email already exists")

    student = User(
        name=payload.name,
        email=normalized_email,
        hashed_password=get_password_hash(payload.password),
        role=UserRole.student,
        program_id=_resolve_program_id(db, payload.program_id),
        department=payload.department,
        section_name=payload.section_name,
        semester_number=payload.semester_number,
        batch_year=payload.batch_year,
        roll_number=payload.roll_number,
        is_active=payload.is_active,
    )
    db.add(student)
    db.commit()
    db.refresh(student)
    return student


@router.put("/students/{student_id}", response_model=StudentListOut)
def update_student(
    student_id: str,
    payload: StudentUpdate,
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> StudentListOut:
    student = db.get(User, student_id)
    if student is None or student.role != UserRole.student:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found")

    data = payload.model_dump(exclude_unset=True)
    if "email" in data:
        normalized_email = data["email"].strip().lower()
        existing = db.execute(select(User).where(User.email == normalized_email, User.id != student_id)).scalar_one_or_none()
        if existing:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Student email already exists")
        data["email"] = normalized_email
    if "program_id" in data:
        data["program_id"] = _resolve_program_id(db, data["program_id"])
    if "password" in data:
        student.hashed_password = get_password_hash(data.pop("password"))

    for key, value in data.items():
        setattr(student, key, value)
    db.commit()
    db.refresh(student)
    return student


@router.delete("/students/{student_id}")
def delete_student(
    student_id: str,
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> dict[str, bool]:
    student = db.get(User, student_id)
    if student is None or student.role != UserRole.student:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found")
    db.delete(student)
    db.commit()
    return {"success": True}
