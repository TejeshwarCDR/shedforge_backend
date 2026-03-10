import uuid
from datetime import datetime
from enum import Enum

from sqlalchemy import JSON, Boolean, DateTime, Enum as SAEnum, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.db.base import Base


class ProgramTerm(Base):
    __tablename__ = "program_terms"
    __table_args__ = (
        UniqueConstraint("program_id", "term_number", name="uq_program_terms_program_term"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    program_id: Mapped[str] = mapped_column(String(36), nullable=False)
    term_number: Mapped[int] = mapped_column(Integer, nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    credits_required: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ProgramSection(Base):
    __tablename__ = "program_sections"
    __table_args__ = (
        UniqueConstraint("program_id", "term_number", "name", name="uq_program_sections_program_term_name"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    program_id: Mapped[str] = mapped_column(String(36), nullable=False)
    term_number: Mapped[int] = mapped_column(Integer, nullable=False)
    name: Mapped[str] = mapped_column(String(50), nullable=False)
    capacity: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ProgramCourse(Base):
    __tablename__ = "program_courses"
    __table_args__ = (
        UniqueConstraint("program_id", "term_number", "course_id", name="uq_program_courses_program_term_course"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    program_id: Mapped[str] = mapped_column(String(36), nullable=False)
    term_number: Mapped[int] = mapped_column(Integer, nullable=False)
    course_id: Mapped[str] = mapped_column(String(36), nullable=False)
    is_required: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    lab_batch_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    allow_parallel_batches: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    prerequisite_course_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ElectiveConflictPolicy(str, Enum):
    no_overlap = "no_overlap"


class ProgramElectiveGroup(Base):
    __tablename__ = "program_elective_groups"
    __table_args__ = (
        UniqueConstraint("program_id", "term_number", "name", name="uq_program_elective_groups_program_term_name"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    program_id: Mapped[str] = mapped_column(String(36), nullable=False)
    term_number: Mapped[int] = mapped_column(Integer, nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    conflict_policy: Mapped[ElectiveConflictPolicy] = mapped_column(
        SAEnum(ElectiveConflictPolicy, name="elective_conflict_policy"),
        nullable=False,
        default=ElectiveConflictPolicy.no_overlap,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ProgramElectiveGroupMember(Base):
    __tablename__ = "program_elective_group_members"
    __table_args__ = (
        UniqueConstraint("group_id", "program_course_id", name="uq_program_elective_group_members_group_course"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    group_id: Mapped[str] = mapped_column(String(36), nullable=False)
    program_course_id: Mapped[str] = mapped_column(String(36), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ProgramSharedLectureGroup(Base):
    __tablename__ = "program_shared_lecture_groups"
    __table_args__ = (
        UniqueConstraint(
            "program_id",
            "term_number",
            "name",
            name="uq_program_shared_lecture_groups_program_term_name",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    program_id: Mapped[str] = mapped_column(String(36), nullable=False)
    term_number: Mapped[int] = mapped_column(Integer, nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    course_id: Mapped[str] = mapped_column(String(36), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ProgramSharedLectureGroupMember(Base):
    __tablename__ = "program_shared_lecture_group_members"
    __table_args__ = (
        UniqueConstraint(
            "group_id",
            "section_name",
            name="uq_program_shared_lecture_group_members_group_section",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    group_id: Mapped[str] = mapped_column(String(36), nullable=False)
    section_name: Mapped[str] = mapped_column(String(50), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
