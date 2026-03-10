import uuid
from datetime import datetime
from enum import Enum

from sqlalchemy import Boolean, DateTime, Enum as SAEnum, Float, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.db.base import Base


class CourseType(str, Enum):
    theory = "theory"
    lab = "lab"
    elective = "elective"


class Course(Base):
    __tablename__ = "courses"
    __table_args__ = (
        UniqueConstraint("program_id", "code", name="uq_courses_program_code"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    program_id: Mapped[str] = mapped_column(String(36), index=True, nullable=False)
    code: Mapped[str] = mapped_column(String(50), index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    type: Mapped[CourseType] = mapped_column(SAEnum(CourseType, name="course_type"), nullable=False)
    credits: Mapped[float] = mapped_column(Float, nullable=False, default=3.0)
    duration_hours: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    sections: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    hours_per_week: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    semester_number: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    batch_year: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    theory_hours: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    lab_hours: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tutorial_hours: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    batch_segregation: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    practical_contiguous_slots: Mapped[int] = mapped_column(Integer, nullable=False, default=2)
    assign_faculty: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    assign_classroom: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    default_room_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    elective_category: Mapped[str | None] = mapped_column(String(120), nullable=True)
    faculty_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), onupdate=func.now())
