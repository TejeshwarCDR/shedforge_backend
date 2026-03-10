import uuid
from datetime import datetime

from sqlalchemy import DateTime, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.db.base import Base


class LeaveSubstituteAssignment(Base):
    __tablename__ = "leave_substitute_assignments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    leave_request_id: Mapped[str] = mapped_column(String(36), nullable=False, unique=True, index=True)
    substitute_faculty_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    assigned_by_id: Mapped[str] = mapped_column(String(36), nullable=False)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), onupdate=func.now())
