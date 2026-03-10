from datetime import datetime

from sqlalchemy import DateTime, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.db.base import Base


class InstitutionSettings(Base):
    __tablename__ = "institution_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    working_hours: Mapped[list[dict]] = mapped_column(JSON, nullable=False)
    period_minutes: Mapped[int] = mapped_column(Integer, nullable=False, default=50)
    lab_contiguous_slots: Mapped[int] = mapped_column(Integer, nullable=False, default=2)
    break_windows: Mapped[list[dict]] = mapped_column(JSON, nullable=False, default=list)
    academic_year: Mapped[str] = mapped_column(String(20), nullable=False, default="2026-2027")
    semester_cycle: Mapped[str] = mapped_column(String(10), nullable=False, default="odd")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
