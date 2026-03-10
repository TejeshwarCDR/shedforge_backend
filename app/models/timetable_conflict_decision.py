import uuid
from datetime import datetime
from enum import Enum

from sqlalchemy import Boolean, DateTime, Enum as SAEnum, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.db.base import Base


class ConflictDecision(str, Enum):
    yes = "yes"
    no = "no"


class TimetableConflictDecision(Base):
    __tablename__ = "timetable_conflict_decisions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    conflict_id: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    decision: Mapped[ConflictDecision] = mapped_column(
        SAEnum(ConflictDecision, name="conflict_decision"),
        nullable=False,
    )
    resolved: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    conflict_snapshot: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    decided_by_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    decided_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), onupdate=func.now())
