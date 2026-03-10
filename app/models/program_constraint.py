import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Integer, JSON, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.db.base import Base


class ProgramConstraint(Base):
    __tablename__ = "program_constraints"
    __table_args__ = (
        UniqueConstraint("program_id", name="uq_program_constraints_program"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    program_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    daily_time_slots: Mapped[list[dict]] = mapped_column(JSON, nullable=False, default=list)
    faculty_min_hours_per_week: Mapped[int] = mapped_column(Integer, nullable=False, default=14)
    faculty_max_hours_per_week: Mapped[int] = mapped_column(Integer, nullable=False, default=20)
    temporal_window_semesters: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    auto_assign_research_slots: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    enforce_student_credit_load: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    enforce_ltp_split: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    enforce_lab_contiguous_blocks: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
