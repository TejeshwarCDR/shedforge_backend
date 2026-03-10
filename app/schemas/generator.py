from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from app.models.timetable_generation import ReevaluationStatus
from app.schemas.settings import DAY_VALUES, TIME_PATTERN, parse_time_to_minutes
from app.schemas.timetable import OfficialTimetablePayload


class ObjectiveWeights(BaseModel):
    room_conflict: int = Field(default=2000, ge=1, le=5000)
    faculty_conflict: int = Field(default=400, ge=1, le=5000)
    section_conflict: int = Field(default=1000, ge=1, le=5000)
    room_capacity: int = Field(default=200, ge=1, le=5000)
    room_type: int = Field(default=150, ge=1, le=5000)
    faculty_availability: int = Field(default=180, ge=1, le=5000)
    locked_slot: int = Field(default=1000, ge=1, le=10000)
    semester_limit: int = Field(default=200, ge=1, le=5000)
    workload_overflow: int = Field(default=1500, ge=1, le=5000)
    workload_underflow: int = Field(default=40, ge=0, le=5000)
    spread_balance: int = Field(default=20, ge=0, le=1000)
    faculty_subject_preference: int = Field(default=30, ge=0, le=2000)


GenerationSolverStrategy = Literal["auto", "hybrid", "simulated_annealing", "genetic", "fast"]


class GenerationSettingsBase(BaseModel):
    solver_strategy: GenerationSolverStrategy = "fast"
    population_size: int = Field(default=120, ge=20, le=2000)
    generations: int = Field(default=300, ge=10, le=5000)
    mutation_rate: float = Field(default=0.12, ge=0.001, le=1.0)
    crossover_rate: float = Field(default=0.8, ge=0.001, le=1.0)
    elite_count: int = Field(default=8, ge=1, le=100)
    tournament_size: int = Field(default=4, ge=2, le=50)
    stagnation_limit: int = Field(default=60, ge=5, le=1000)
    annealing_iterations: int = Field(default=900, ge=100, le=20_000)
    annealing_initial_temperature: float = Field(default=6.0, ge=0.1, le=1000.0)
    annealing_cooling_rate: float = Field(default=0.995, ge=0.80, le=0.99999)
    random_seed: int | None = Field(default=None, ge=0, le=2_000_000_000)
    objective_weights: ObjectiveWeights = Field(default_factory=ObjectiveWeights)

    @model_validator(mode="after")
    def validate_relationships(self) -> "GenerationSettingsBase":
        if self.elite_count >= self.population_size:
            raise ValueError("elite_count must be less than population_size")
        if self.tournament_size > self.population_size:
            raise ValueError("tournament_size cannot exceed population_size")
        return self


class GenerationSettingsUpdate(GenerationSettingsBase):
    pass


class GenerationSettingsOut(GenerationSettingsBase):
    id: int


class SlotLockBase(BaseModel):
    program_id: str = Field(min_length=1, max_length=36)
    term_number: int = Field(ge=1, le=20)
    day: str
    start_time: str
    end_time: str
    section_name: str = Field(min_length=1, max_length=50)
    course_id: str = Field(min_length=1, max_length=36)
    batch: str | None = Field(default=None, min_length=1, max_length=50)
    room_id: str | None = Field(default=None, min_length=1, max_length=36)
    faculty_id: str | None = Field(default=None, min_length=1, max_length=36)
    notes: str | None = Field(default=None, max_length=500)
    is_active: bool = True

    @field_validator("day")
    @classmethod
    def validate_day(cls, value: str) -> str:
        day = value.strip()
        if day not in DAY_VALUES:
            raise ValueError("Invalid day value")
        return day

    @field_validator("start_time", "end_time")
    @classmethod
    def validate_time(cls, value: str) -> str:
        if not TIME_PATTERN.match(value):
            raise ValueError("Time must be in HH:MM 24-hour format")
        return value

    @model_validator(mode="after")
    def validate_time_order(self) -> "SlotLockBase":
        if parse_time_to_minutes(self.end_time) <= parse_time_to_minutes(self.start_time):
            raise ValueError("end_time must be after start_time")
        return self


class SlotLockCreate(SlotLockBase):
    pass


class SlotLockOut(SlotLockBase):
    id: str
    created_by_id: str | None = None

    model_config = {"from_attributes": True}


class GenerateTimetableRequest(BaseModel):
    program_id: str = Field(min_length=1, max_length=36)
    term_number: int = Field(ge=1, le=20)
    alternative_count: int = Field(default=3, ge=1, le=5)
    persist_official: bool = False
    settings_override: GenerationSettingsBase | None = None


class FacultyWorkloadBridgeSuggestion(BaseModel):
    term_number: int | None = Field(default=None, ge=1, le=20)
    course_id: str = Field(min_length=1, max_length=36)
    course_code: str = Field(min_length=1, max_length=50)
    course_name: str = Field(min_length=1, max_length=200)
    section_name: str = Field(min_length=1, max_length=50)
    batch: str | None = Field(default=None, min_length=1, max_length=50)
    weekly_hours: float = Field(ge=0.0, le=100.0)
    is_preferred_subject: bool = False
    feasible_without_conflict: bool = True


class FacultyWorkloadGapSuggestion(BaseModel):
    faculty_id: str = Field(min_length=1, max_length=36)
    faculty_name: str = Field(min_length=1, max_length=200)
    department: str | None = Field(default=None, max_length=200)
    target_hours: float = Field(ge=0.0, le=200.0)
    assigned_hours: float = Field(ge=0.0, le=200.0)
    preferred_assigned_hours: float = Field(ge=0.0, le=200.0)
    gap_hours: float = Field(ge=0.0, le=200.0)
    suggested_bridges: list[FacultyWorkloadBridgeSuggestion] = Field(default_factory=list)


class OccupancyMatrix(BaseModel):
    section_matrix: dict[str, dict[str, int]] = Field(default_factory=dict)
    faculty_matrix: dict[str, dict[str, int]] = Field(default_factory=dict)
    room_matrix: dict[str, dict[str, int]] = Field(default_factory=dict)
    faculty_labels: dict[str, str] = Field(default_factory=dict)
    room_labels: dict[str, str] = Field(default_factory=dict)


class GeneratedAlternative(BaseModel):
    rank: int
    fitness: float
    hard_conflicts: int
    soft_penalty: float
    payload: OfficialTimetablePayload
    workload_gap_suggestions: list[FacultyWorkloadGapSuggestion] = Field(default_factory=list)
    occupancy_matrix: OccupancyMatrix | None = None


class AutoResolvedConflictEntry(BaseModel):
    conflict_id: str = Field(min_length=1, max_length=255)
    conflict_type: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=500)
    resolution: str = Field(min_length=1, max_length=500)
    resolved: bool = True


class GenerateTimetableResponse(BaseModel):
    alternatives: list[GeneratedAlternative]
    settings_used: GenerationSettingsBase
    runtime_ms: int
    published_version_label: str | None = None
    publish_warning: str | None = None
    auto_saved_version_label: str | None = None
    auto_resolved_conflicts: list[AutoResolvedConflictEntry] = Field(default_factory=list)


class GenerateTimetableCycleRequest(BaseModel):
    program_id: str = Field(min_length=1, max_length=36)
    cycle: Literal["odd", "even", "all", "custom"] | None = None
    term_numbers: list[int] | None = None
    alternative_count: int = Field(default=3, ge=1, le=5)
    pareto_limit: int = Field(default=12, ge=1, le=30)
    persist_official: bool = False
    settings_override: GenerationSettingsBase | None = None

    @field_validator("term_numbers")
    @classmethod
    def normalize_term_numbers(cls, value: list[int] | None) -> list[int] | None:
        if value is None:
            return None
        unique: list[int] = []
        seen: set[int] = set()
        for item in value:
            if item < 1 or item > 20:
                raise ValueError("term_numbers must be between 1 and 20")
            if item in seen:
                continue
            seen.add(item)
            unique.append(item)
        return sorted(unique)

    @model_validator(mode="after")
    def validate_cycle_scope(self) -> "GenerateTimetableCycleRequest":
        if self.cycle == "custom":
            if not self.term_numbers:
                raise ValueError("term_numbers is required for custom cycle")
        elif self.term_numbers:
            raise ValueError("term_numbers can only be provided for custom cycle")
        return self


class GeneratedCycleTermResult(BaseModel):
    term_number: int
    generation: GenerateTimetableResponse
    published_version_label: str | None = None
    auto_saved_version_label: str | None = None


class GeneratedCycleSolutionTerm(BaseModel):
    term_number: int
    alternative_rank: int
    fitness: float
    hard_conflicts: int
    soft_penalty: float
    payload: OfficialTimetablePayload
    workload_gap_suggestions: list[FacultyWorkloadGapSuggestion] = Field(default_factory=list)
    occupancy_matrix: OccupancyMatrix | None = None


class GeneratedCycleSolution(BaseModel):
    rank: int
    resource_penalty: int
    faculty_preference_penalty: float
    workload_gap_penalty: float
    hard_conflicts: int
    soft_penalty: float
    runtime_ms: int
    terms: list[GeneratedCycleSolutionTerm]
    workload_gap_suggestions: list[FacultyWorkloadGapSuggestion] = Field(default_factory=list)


class GenerateTimetableCycleResponse(BaseModel):
    program_id: str
    cycle: Literal["odd", "even", "all", "custom"]
    term_numbers: list[int]
    results: list[GeneratedCycleTermResult]
    pareto_front: list[GeneratedCycleSolution] = Field(default_factory=list)
    selected_solution_rank: int | None = None


GenerationJobKind = Literal["single", "cycle"]
GenerationJobStatus = Literal["queued", "running", "succeeded", "failed"]


class GenerationJobAccepted(BaseModel):
    job_id: str
    kind: GenerationJobKind
    status: GenerationJobStatus
    created_at: datetime


class GenerationJobEvent(BaseModel):
    id: int = Field(ge=1)
    at: datetime
    stage: str = Field(min_length=1, max_length=100)
    level: Literal["info", "success", "warn", "error"] = "info"
    message: str = Field(min_length=1, max_length=500)
    progress_percent: float | None = Field(default=None, ge=0.0, le=100.0)
    metrics: dict[str, Any] = Field(default_factory=dict)
    latest_generation: GenerateTimetableResponse | None = None


class GenerationJobStatusOut(BaseModel):
    job_id: str
    kind: GenerationJobKind
    status: GenerationJobStatus
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    updated_at: datetime
    progress_percent: float | None = Field(default=None, ge=0.0, le=100.0)
    stage: str | None = None
    message: str | None = None
    events: list[GenerationJobEvent] = Field(default_factory=list)
    last_event_id: int = Field(default=0, ge=0)
    latest_generation: GenerateTimetableResponse | None = None
    result: GenerateTimetableResponse | GenerateTimetableCycleResponse | None = None
    error_message: str | None = Field(default=None, max_length=1000)
    next_poll_after_ms: int = Field(default=1500, ge=250, le=10_000)


class ReevaluationEventOut(BaseModel):
    id: str
    program_id: str
    term_number: int | None = None
    change_type: str
    entity_type: str
    entity_id: str | None = None
    description: str
    details: dict = Field(default_factory=dict)
    status: ReevaluationStatus
    triggered_by_id: str | None = None
    triggered_at: datetime
    resolved_by_id: str | None = None
    resolved_at: datetime | None = None
    resolution_note: str | None = None
    has_official_impact: bool = False

    model_config = {"from_attributes": True}


class ReevaluateTimetableRequest(BaseModel):
    program_id: str = Field(min_length=1, max_length=36)
    term_number: int = Field(ge=1, le=20)
    alternative_count: int = Field(default=3, ge=1, le=5)
    persist_official: bool = False
    mark_resolved: bool = True
    resolution_note: str | None = Field(default=None, max_length=500)
    settings_override: GenerationSettingsBase | None = None


class ReevaluateTimetableResponse(BaseModel):
    generation: GenerateTimetableResponse
    resolved_events: int
    pending_events: int
