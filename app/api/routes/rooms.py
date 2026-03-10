from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db, require_roles
from app.models.program import Program
from app.models.room import Room
from app.models.user import User, UserRole
from app.schemas.room import RoomCreate, RoomOut, RoomUpdate
from app.services.notifications import notify_admin_update

router = APIRouter()


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
            detail="No program available. Create a program before managing rooms.",
        )
    return default_program.id


@router.get("/", response_model=list[RoomOut])
def list_rooms(
    program_id: str | None = Query(default=None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[RoomOut]:
    statement = select(Room)
    if program_id:
        statement = statement.where(Room.program_id == program_id)
    elif current_user.role in {UserRole.faculty, UserRole.student} and current_user.program_id:
        statement = statement.where(Room.program_id == current_user.program_id)
    return list(db.execute(statement).scalars())


@router.post("/", response_model=RoomOut, status_code=status.HTTP_201_CREATED)
def create_room(
    payload: RoomCreate,
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> RoomOut:
    payload_data = payload.model_dump()
    resolved_program_id = _resolve_program_id(db, payload_data.get("program_id"))
    payload_data["program_id"] = resolved_program_id
    existing = db.execute(
        select(Room).where(Room.program_id == resolved_program_id, Room.name == payload_data["name"])
    ).scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Room name already exists in this program")
    room = Room(**payload_data)
    db.add(room)
    notify_admin_update(
        db,
        title="Room Added",
        message=f"{current_user.name} added room {payload.name} ({payload.type.value}).",
        actor_user_id=current_user.id,
    )
    db.commit()
    db.refresh(room)
    return room


@router.put("/{room_id}", response_model=RoomOut)
def update_room(
    room_id: str,
    payload: RoomUpdate,
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> RoomOut:
    room = db.get(Room, room_id)
    if room is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Room not found")

    data = payload.model_dump(exclude_unset=True)
    if "program_id" in data:
        data["program_id"] = _resolve_program_id(db, data["program_id"])
    if "name" in data or "program_id" in data:
        candidate_name = data.get("name", room.name)
        candidate_program_id = data.get("program_id", room.program_id)
        existing = db.execute(
            select(Room).where(
                Room.program_id == candidate_program_id,
                Room.name == candidate_name,
                Room.id != room_id,
            )
        ).scalar_one_or_none()
        if existing:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Room name already exists in this program")

    for key, value in data.items():
        setattr(room, key, value)
    if data:
        notify_admin_update(
            db,
            title="Room Updated",
            message=f"{current_user.name} updated room {room.name}.",
            actor_user_id=current_user.id,
        )
    db.commit()
    db.refresh(room)
    return room


@router.delete("/{room_id}")
def delete_room(
    room_id: str,
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> dict:
    room = db.get(Room, room_id)
    if room is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Room not found")
    notify_admin_update(
        db,
        title="Room Deleted",
        message=f"{current_user.name} deleted room {room.name}.",
        actor_user_id=current_user.id,
    )
    db.delete(room)
    db.commit()
    return {"success": True}
