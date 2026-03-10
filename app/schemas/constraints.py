from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from app.schemas.settings import TIME_PATTERN, parse_time_to_minutes


class SemesterConstraintBase(BaseModel):
    term_number: int = Field(ge=1, le=20)
    earliest_start_time: str = Field(min_length=4, max_length=5)
    latest_end_time: str = Field(min_length=4, max_length=5)
    max_hours_per_day: int = Field(ge=1, le=24)
    max_hours_per_week: int = Field(ge=1, le=200)
    min_break_minutes: int = Field(ge=0, le=180)
    max_consecutive_hours: int = Field(ge=1, le=12)

    @field_validator("earliest_start_time", "latest_end_time")
    @classmethod
    def validate_time_format(cls, value: str) -> str:
        if not TIME_PATTERN.match(value):
            raise ValueError("Time must be in HH:MM 24-hour format")
        return value

    @model_validator(mode="after")
    def validate_time_order(self) -> "SemesterConstraintBase":
        start = parse_time_to_minutes(self.earliest_start_time)
        end = parse_time_to_minutes(self.latest_end_time)
        if end <= start:
            raise ValueError("Latest end time must be after earliest start time")
        if self.max_consecutive_hours > self.max_hours_per_day:
            raise ValueError("Max consecutive hours cannot exceed max hours per day")
        return self


class SemesterConstraintUpsert(SemesterConstraintBase):
    pass


class SemesterConstraintOut(SemesterConstraintBase):
    id: str

    model_config = {"from_attributes": True}


ConstraintSlotTag = Literal["teaching", "block", "break", "lunch"]


class ProgramDailyTimeSlot(BaseModel):
    start_time: str = Field(min_length=4, max_length=5)
    end_time: str = Field(min_length=4, max_length=5)
    tag: ConstraintSlotTag = "teaching"
    label: str | None = Field(default=None, max_length=80)

    @field_validator("start_time", "end_time")
    @classmethod
    def validate_time_format(cls, value: str) -> str:
        if not TIME_PATTERN.match(value):
            raise ValueError("Time must be in HH:MM 24-hour format")
        return value

    @field_validator("label", mode="before")
    @classmethod
    def normalize_label(cls, value: str | None) -> str | None:
        if value is None:
            return None
        trimmed = str(value).strip()
        return trimmed or None

    @model_validator(mode="after")
    def validate_time_order(self) -> "ProgramDailyTimeSlot":
        start = parse_time_to_minutes(self.start_time)
        end = parse_time_to_minutes(self.end_time)
        if end <= start:
            raise ValueError("end_time must be after start_time")
        if self.tag != "teaching" and not (self.label and self.label.strip()):
            self.label = self.tag.title()
        return self


class ProgramConstraintBase(BaseModel):
    program_id: str = Field(min_length=1, max_length=36)
    daily_time_slots: list[ProgramDailyTimeSlot] = Field(default_factory=list, max_length=40)
    faculty_min_hours_per_week: int = Field(default=14, ge=0, le=80)
    faculty_max_hours_per_week: int = Field(default=20, ge=1, le=80)
    temporal_window_semesters: int = Field(default=3, ge=1, le=9)
    auto_assign_research_slots: bool = True
    enforce_student_credit_load: bool = True
    enforce_ltp_split: bool = True
    enforce_lab_contiguous_blocks: bool = True

    @model_validator(mode="after")
    def validate_program_constraints(self) -> "ProgramConstraintBase":
        if self.faculty_max_hours_per_week < self.faculty_min_hours_per_week:
            raise ValueError("faculty_max_hours_per_week must be >= faculty_min_hours_per_week")

        normalized_slots = [
            slot if isinstance(slot, ProgramDailyTimeSlot) else ProgramDailyTimeSlot.model_validate(slot)
            for slot in self.daily_time_slots
        ]
        sorted_slots = sorted(normalized_slots, key=lambda slot: parse_time_to_minutes(slot.start_time))
        for index, slot in enumerate(sorted_slots):
            if index == 0:
                continue
            prev = sorted_slots[index - 1]
            prev_end = parse_time_to_minutes(prev.end_time)
            current_start = parse_time_to_minutes(slot.start_time)
            if current_start < prev_end:
                raise ValueError("Daily time slots cannot overlap")
        if sorted_slots and not any(slot.tag == "teaching" for slot in sorted_slots):
            raise ValueError("At least one daily slot must have tag='teaching'")
        self.daily_time_slots = sorted_slots
        return self


class ProgramConstraintUpsert(ProgramConstraintBase):
    pass


class ProgramConstraintOut(ProgramConstraintBase):
    id: str
    updated_at: datetime | None = None

    model_config = {"from_attributes": True}


class ConstraintViolation(BaseModel):
    code: str = Field(min_length=1, max_length=64)
    severity: Literal["hard", "warn"] = "warn"
    message: str = Field(min_length=1, max_length=500)
    term_number: int | None = Field(default=None, ge=1, le=20)
    course_id: str | None = Field(default=None, min_length=1, max_length=36)
    faculty_id: str | None = Field(default=None, min_length=1, max_length=36)


class ProgramConstraintReport(BaseModel):
    program_id: str = Field(min_length=1, max_length=36)
    generated_at: datetime
    violation_count: int = Field(ge=0)
    violations: list[ConstraintViolation] = Field(default_factory=list)
