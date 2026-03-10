from __future__ import annotations

import re
from typing import Literal
from typing import ClassVar

from pydantic import BaseModel, EmailStr, Field, field_validator, model_validator

DAY_VALUES = {
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
}

TIME_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


def parse_time_to_minutes(value: str) -> int:
    if not TIME_PATTERN.match(value):
        raise ValueError("Time must be in HH:MM 24-hour format")
    hours, minutes = value.split(":")
    return int(hours) * 60 + int(minutes)


class WorkingHoursEntry(BaseModel):
    day: str
    start_time: str
    end_time: str
    enabled: bool = True

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
    def validate_time_order(self) -> "WorkingHoursEntry":
        if self.enabled:
            start = parse_time_to_minutes(self.start_time)
            end = parse_time_to_minutes(self.end_time)
            if end <= start:
                raise ValueError("End time must be after start time")
        return self


class WorkingHoursUpdate(BaseModel):
    hours: list[WorkingHoursEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_unique_days(self) -> "WorkingHoursUpdate":
        seen: set[str] = set()
        duplicates: set[str] = set()
        for entry in self.hours:
            if entry.day in seen:
                duplicates.add(entry.day)
            else:
                seen.add(entry.day)
        if duplicates:
            raise ValueError(f"Duplicate day entries: {', '.join(sorted(duplicates))}")
        return self


class BreakWindowEntry(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    start_time: str
    end_time: str

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("Break name cannot be empty")
        return trimmed

    @field_validator("start_time", "end_time")
    @classmethod
    def validate_time(cls, value: str) -> str:
        if not TIME_PATTERN.match(value):
            raise ValueError("Time must be in HH:MM 24-hour format")
        return value

    @model_validator(mode="after")
    def validate_time_order(self) -> "BreakWindowEntry":
        start = parse_time_to_minutes(self.start_time)
        end = parse_time_to_minutes(self.end_time)
        if end <= start:
            raise ValueError("Break end time must be after start time")
        return self


class SchedulePolicyUpdate(BaseModel):
    period_minutes: int = Field(ge=5, le=180)
    lab_contiguous_slots: int = Field(ge=1, le=8)
    breaks: list[BreakWindowEntry] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def validate_breaks(self) -> "SchedulePolicyUpdate":
        self.breaks = [
            BreakWindowEntry(
                name=item.name.strip(),
                start_time=item.start_time,
                end_time=item.end_time,
            )
            for item in self.breaks
        ]

        windows = sorted(
            ((parse_time_to_minutes(item.start_time), parse_time_to_minutes(item.end_time), item.name) for item in self.breaks),
            key=lambda item: item[0],
        )

        for index, (_, _, name) in enumerate(windows):
            for _, _, other_name in windows[index + 1 :]:
                if other_name == name:
                    raise ValueError(f"Duplicate break name: {name}")

        for index, (start, end, _) in enumerate(windows):
            if index == 0:
                continue
            prev_start, prev_end, _ = windows[index - 1]
            if start < prev_end and end > prev_start:
                raise ValueError("Break windows cannot overlap")

        return self


SemesterCycle = Literal["odd", "even"]


class AcademicCycleSettings(BaseModel):
    academic_year: str = Field(min_length=7, max_length=20)
    semester_cycle: SemesterCycle = "odd"

    @field_validator("academic_year")
    @classmethod
    def normalize_academic_year(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("academic_year cannot be empty")
        if not re.match(r"^\d{4}\s*-\s*\d{4}$", cleaned):
            raise ValueError("academic_year must follow YYYY-YYYY format")
        start_raw, end_raw = cleaned.replace(" ", "").split("-")
        start_year = int(start_raw)
        end_year = int(end_raw)
        if end_year != start_year + 1:
            raise ValueError("academic_year end year must be start year + 1")
        return f"{start_year}-{end_year}"


class SmtpConfigurationOut(BaseModel):
    configured: bool
    host: str | None = None
    port: int
    username_set: bool
    from_email: str | None = None
    from_name: str
    use_tls: bool
    use_ssl: bool
    backup_configured: bool
    backup_host: str | None = None
    backup_port: int
    notification_prefer_backup: bool
    retry_attempts: int
    retry_backoff_seconds: float
    rate_limit_cooldown_seconds: int
    timeout_seconds: int


class SmtpTestRequest(BaseModel):
    to_email: EmailStr | None = None


class SmtpTestResponse(BaseModel):
    success: bool
    message: str
    recipient: EmailStr


DEFAULT_WORKING_HOURS: list[WorkingHoursEntry] = [
    WorkingHoursEntry(day="Monday", start_time="08:50", end_time="16:35", enabled=True),
    WorkingHoursEntry(day="Tuesday", start_time="08:50", end_time="16:35", enabled=True),
    WorkingHoursEntry(day="Wednesday", start_time="08:50", end_time="16:35", enabled=True),
    WorkingHoursEntry(day="Thursday", start_time="08:50", end_time="16:35", enabled=True),
    WorkingHoursEntry(day="Friday", start_time="08:50", end_time="16:35", enabled=True),
    WorkingHoursEntry(day="Saturday", start_time="08:50", end_time="16:35", enabled=False),
    WorkingHoursEntry(day="Sunday", start_time="08:50", end_time="16:35", enabled=False),
]

DEFAULT_SCHEDULE_POLICY = SchedulePolicyUpdate(
    period_minutes=50,
    lab_contiguous_slots=2,
    breaks=[
        BreakWindowEntry(name="Short Break", start_time="10:30", end_time="10:45"),
        BreakWindowEntry(name="Lunch Break", start_time="13:15", end_time="14:05"),
    ],
)

DEFAULT_ACADEMIC_CYCLE = AcademicCycleSettings(
    academic_year="2026-2027",
    semester_cycle="odd",
)
