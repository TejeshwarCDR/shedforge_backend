from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum

from sqlalchemy import Boolean, DateTime, Enum as SAEnum, Float, Integer, JSON, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.db.base import Base


class TimetableGenerationSettings(Base):
    __tablename__ = "timetable_generation_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    solver_strategy: Mapped[str] = mapped_column(String(40), nullable=False, default="auto")
    population_size: Mapped[int] = mapped_column(Integer, nullable=False, default=120)
    generations: Mapped[int] = mapped_column(Integer, nullable=False, default=300)
    mutation_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.12)
    crossover_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.8)
    elite_count: Mapped[int] = mapped_column(Integer, nullable=False, default=8)
    tournament_size: Mapped[int] = mapped_column(Integer, nullable=False, default=4)
    stagnation_limit: Mapped[int] = mapped_column(Integer, nullable=False, default=60)
    annealing_iterations: Mapped[int] = mapped_column(Integer, nullable=False, default=900)
    annealing_initial_temperature: Mapped[float] = mapped_column(Float, nullable=False, default=6.0)
    annealing_cooling_rate: Mapped[float] = mapped_column(Float, nullable=False, default=0.995)
    random_seed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    objective_weights: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class TimetableSlotLock(Base):
    __tablename__ = "timetable_slot_locks"
    __table_args__ = (
        UniqueConstraint(
            "program_id",
            "term_number",
            "day",
            "start_time",
            "section_name",
            "course_id",
            "batch",
            name="uq_timetable_slot_locks_identity",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    program_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    term_number: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    day: Mapped[str] = mapped_column(String(20), nullable=False)
    start_time: Mapped[str] = mapped_column(String(5), nullable=False)
    end_time: Mapped[str] = mapped_column(String(5), nullable=False)
    section_name: Mapped[str] = mapped_column(String(50), nullable=False)
    course_id: Mapped[str] = mapped_column(String(36), nullable=False)
    batch: Mapped[str | None] = mapped_column(String(50), nullable=True)
    room_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    faculty_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    notes: Mapped[str | None] = mapped_column(String(500), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_by_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ReevaluationStatus(str, Enum):
    pending = "pending"
    resolved = "resolved"
    dismissed = "dismissed"


class TimetableReevaluationEvent(Base):
    __tablename__ = "timetable_reevaluation_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    program_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    term_number: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    change_type: Mapped[str] = mapped_column(String(80), nullable=False)
    entity_type: Mapped[str] = mapped_column(String(80), nullable=False)
    entity_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    description: Mapped[str] = mapped_column(String(500), nullable=False)
    details: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    status: Mapped[ReevaluationStatus] = mapped_column(
        SAEnum(ReevaluationStatus, name="reevaluation_status"),
        nullable=False,
        default=ReevaluationStatus.pending,
        index=True,
    )
    triggered_by_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    triggered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    resolved_by_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    resolution_note: Mapped[str | None] = mapped_column(String(500), nullable=True)
