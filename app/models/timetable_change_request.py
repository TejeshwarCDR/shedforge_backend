import uuid
from datetime import datetime
from enum import Enum

from sqlalchemy import JSON, DateTime, Enum as SAEnum, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.db.base import Base


class TimetableChangeRequestStatus(str, Enum):
    pending = "pending"
    approved = "approved"
    rejected = "rejected"
    applied = "applied"


class TimetableChangeRequest(Base):
    __tablename__ = "timetable_change_requests"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    program_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    term_number: Mapped[int | None] = mapped_column(nullable=True, index=True)
    slot_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)

    requested_by_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    requested_by_role: Mapped[str] = mapped_column(String(20), nullable=False)
    approver_user_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    approver_role: Mapped[str | None] = mapped_column(String(20), nullable=True)

    status: Mapped[TimetableChangeRequestStatus] = mapped_column(
        SAEnum(TimetableChangeRequestStatus, name="timetable_change_request_status"),
        nullable=False,
        default=TimetableChangeRequestStatus.pending,
    )
    proposal: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    request_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    decision_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    resolution_note: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), onupdate=func.now())
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    applied_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
