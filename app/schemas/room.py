from pydantic import BaseModel, Field, field_validator, model_validator

from app.models.room import RoomType
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


class RoomBase(BaseModel):
    program_id: str | None = Field(default=None, min_length=1, max_length=36)
    name: str = Field(min_length=1, max_length=100)
    building: str = Field(min_length=1, max_length=200)
    capacity: int = Field(ge=1, le=1000)
    type: RoomType
    has_lab_equipment: bool = False
    has_projector: bool = False
    availability_windows: list[AvailabilityWindow] = Field(default_factory=list, max_length=100)


class RoomCreate(RoomBase):
    pass


class RoomUpdate(BaseModel):
    program_id: str | None = Field(default=None, min_length=1, max_length=36)
    name: str | None = Field(default=None, min_length=1, max_length=100)
    building: str | None = Field(default=None, min_length=1, max_length=200)
    capacity: int | None = Field(default=None, ge=1, le=1000)
    type: RoomType | None = None
    has_lab_equipment: bool | None = None
    has_projector: bool | None = None
    availability_windows: list[AvailabilityWindow] | None = Field(default=None, max_length=100)


class RoomOut(RoomBase):
    id: str

    model_config = {"from_attributes": True}
