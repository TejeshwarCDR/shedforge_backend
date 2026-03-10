import uuid
from datetime import datetime
from enum import Enum

from sqlalchemy import Boolean, DateTime, Enum as SAEnum, Integer, JSON, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.db.base import Base


class RoomType(str, Enum):
    lecture = "lecture"
    lab = "lab"
    seminar = "seminar"


class Room(Base):
    __tablename__ = "rooms"
    __table_args__ = (
        UniqueConstraint("program_id", "name", name="uq_rooms_program_name"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    program_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(100), index=True, nullable=False)
    building: Mapped[str] = mapped_column(String(200), nullable=False)
    capacity: Mapped[int] = mapped_column(Integer, nullable=False, default=30)
    type: Mapped[RoomType] = mapped_column(SAEnum(RoomType, name="room_type"), nullable=False)
    has_lab_equipment: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    has_projector: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    availability_windows: Mapped[list[dict]] = mapped_column(JSON, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), onupdate=func.now())
