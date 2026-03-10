import uuid
from datetime import datetime

from sqlalchemy import DateTime, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.db.base import Base


class SemesterConstraint(Base):
    __tablename__ = "semester_constraints"
    __table_args__ = (
        UniqueConstraint("term_number", name="uq_semester_constraints_term"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    term_number: Mapped[int] = mapped_column(Integer, nullable=False)
    earliest_start_time: Mapped[str] = mapped_column(String(5), nullable=False)
    latest_end_time: Mapped[str] = mapped_column(String(5), nullable=False)
    max_hours_per_day: Mapped[int] = mapped_column(Integer, nullable=False, default=6)
    max_hours_per_week: Mapped[int] = mapped_column(Integer, nullable=False, default=30)
    min_break_minutes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_consecutive_hours: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
