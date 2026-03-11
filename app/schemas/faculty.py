from pydantic import BaseModel, Field, field_validator, model_validator

from app.schemas.settings import DAY_VALUES, TIME_PATTERN, parse_time_to_minutes


class AvailabilityWindow(BaseModel):
    day: str
    start_time: str
    end_time: str

    @field_validator("day")
    @classmethod
    def validate_day(cls, value: str) -> str:
        day = value.strip()
        if day not in DAY_VALUES:
            raise ValueError("Invalid day value")
        return day

    @field_validator("start_time", "end_time")
    @classmethod
    def validate_time_format(cls, value: str) -> str:
        if not TIME_PATTERN.match(value):
            raise ValueError("Time must be in HH:MM 24-hour format")
        return value

    @model_validator(mode="after")
    def validate_order(self) -> "AvailabilityWindow":
        if parse_time_to_minutes(self.end_time) <= parse_time_to_minutes(self.start_time):
            raise ValueError("end_time must be after start_time")
        return self


class FacultyBase(BaseModel):
    program_id: str | None = Field(default=None, min_length=1, max_length=36)
    name: str = Field(min_length=1, max_length=200)
    designation: str = Field(default="Faculty", min_length=1, max_length=200)
    email: str
    department: str = Field(min_length=1, max_length=200)
    workload_hours: int = Field(ge=0, le=200)
    max_hours: int = Field(ge=1, le=200)
    availability: list[str] = Field(default_factory=list, max_length=14)
    availability_windows: list[AvailabilityWindow] = Field(default_factory=list, max_length=100)
    avoid_back_to_back: bool = False
    preferred_min_break_minutes: int = Field(default=0, ge=0, le=120)
    preference_notes: str | None = Field(default=None, max_length=2000)
    preferred_subject_codes: list[str] = Field(default_factory=list, max_length=100)
    semester_preferences: dict[str, list[str]] = Field(default_factory=dict)

    @field_validator("preferred_subject_codes")
    @classmethod
    def normalize_preferred_subject_codes(cls, value: list[str]) -> list[str]:
        seen: set[str] = set()
        normalized_codes: list[str] = []
        for item in value:
            code = item.strip().upper()
            if not code:
                continue
            if len(code) > 50:
                raise ValueError("Preferred subject code length cannot exceed 50 characters")
            if code in seen:
                continue
            seen.add(code)
            normalized_codes.append(code)
        return normalized_codes

    @field_validator("semester_preferences")
    @classmethod
    def normalize_semester_preferences(cls, value: dict[str, list[str]]) -> dict[str, list[str]]:
        normalized: dict[str, list[str]] = {}
        for key, codes in value.items():
            semester_key = str(key).strip()
            if not semester_key.isdigit():
                raise ValueError("semester_preferences keys must be numeric semester identifiers")
            semester_number = int(semester_key)
            if semester_number < 1 or semester_number > 20:
                raise ValueError("semester_preferences keys must be between 1 and 20")
            seen: set[str] = set()
            normalized_codes: list[str] = []
            for item in codes:
                code = item.strip().upper()
                if not code:
                    continue
                if len(code) > 50:
                    raise ValueError("Preferred subject code length cannot exceed 50 characters")
                if code in seen:
                    continue
                seen.add(code)
                normalized_codes.append(code)
            normalized[str(semester_number)] = normalized_codes
        return normalized


class FacultyCreate(FacultyBase):
    pass


class FacultyUpdate(BaseModel):
    program_id: str | None = Field(default=None, min_length=1, max_length=36)
    name: str | None = Field(default=None, min_length=1, max_length=200)
    designation: str | None = Field(default=None, min_length=1, max_length=200)
    email: str | None = None
    department: str | None = Field(default=None, min_length=1, max_length=200)
    workload_hours: int | None = Field(default=None, ge=0, le=200)
    max_hours: int | None = Field(default=None, ge=1, le=200)
    availability: list[str] | None = Field(default=None, max_length=14)
    availability_windows: list[AvailabilityWindow] | None = Field(default=None, max_length=100)
    avoid_back_to_back: bool | None = None
    preferred_min_break_minutes: int | None = Field(default=None, ge=0, le=120)
    preference_notes: str | None = Field(default=None, max_length=2000)
    preferred_subject_codes: list[str] | None = Field(default=None, max_length=100)
    semester_preferences: dict[str, list[str]] | None = None

    @field_validator("preferred_subject_codes")
    @classmethod
    def normalize_optional_preferred_subject_codes(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        seen: set[str] = set()
        normalized_codes: list[str] = []
        for item in value:
            code = item.strip().upper()
            if not code:
                continue
            if len(code) > 50:
                raise ValueError("Preferred subject code length cannot exceed 50 characters")
            if code in seen:
                continue
            seen.add(code)
            normalized_codes.append(code)
        return normalized_codes

    @field_validator("semester_preferences")
    @classmethod
    def normalize_optional_semester_preferences(
        cls,
        value: dict[str, list[str]] | None,
    ) -> dict[str, list[str]] | None:
        if value is None:
            return None
        normalized: dict[str, list[str]] = {}
        for key, codes in value.items():
            semester_key = str(key).strip()
            if not semester_key.isdigit():
                raise ValueError("semester_preferences keys must be numeric semester identifiers")
            semester_number = int(semester_key)
            if semester_number < 1 or semester_number > 20:
                raise ValueError("semester_preferences keys must be between 1 and 20")
            seen: set[str] = set()
            normalized_codes: list[str] = []
            for item in codes:
                code = item.strip().upper()
                if not code:
                    continue
                if len(code) > 50:
                    raise ValueError("Preferred subject code length cannot exceed 50 characters")
                if code in seen:
                    continue
                seen.add(code)
                normalized_codes.append(code)
            normalized[str(semester_number)] = normalized_codes
        return normalized


class FacultyOut(FacultyBase):
    id: str

    model_config = {"from_attributes": True}
