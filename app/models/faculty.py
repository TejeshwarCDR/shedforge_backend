import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.db.base import Base


class Faculty(Base):
    __tablename__ = "faculty"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    program_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    designation: Mapped[str] = mapped_column(String(200), nullable=False, default="Faculty")
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    department: Mapped[str] = mapped_column(String(200), nullable=False)
    workload_hours: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_hours: Mapped[int] = mapped_column(Integer, nullable=False, default=20)
    availability: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    availability_windows: Mapped[list[dict]] = mapped_column(JSON, nullable=False, default=list)
    avoid_back_to_back: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    preferred_min_break_minutes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    preference_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    preferred_subject_codes: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    semester_preferences: Mapped[dict[str, list[str]]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), onupdate=func.now())
