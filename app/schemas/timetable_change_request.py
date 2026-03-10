from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from app.models.timetable_change_request import TimetableChangeRequestStatus
from app.schemas.timetable import parse_time_to_minutes


class TimetableChangeRequestProposalIn(BaseModel):
    slot_id: str = Field(min_length=1, max_length=36, alias="slotId")
    day: str = Field(min_length=3, max_length=20)
    start_time: str = Field(alias="startTime")
    end_time: str = Field(alias="endTime")
    room_id: str | None = Field(default=None, alias="roomId", min_length=1, max_length=36)
    faculty_id: str | None = Field(default=None, alias="facultyId", min_length=1, max_length=36)
    assistant_faculty_ids: list[str] | None = Field(default=None, alias="assistantFacultyIds")
    section: str | None = Field(default=None, min_length=1, max_length=50)
    request_kind: Literal["slot_move", "resource_reassign", "extra_class"] = Field(
        default="slot_move",
        alias="requestKind",
    )
    note: str | None = Field(default=None, max_length=1000)

    model_config = {
        "populate_by_name": True,
        "from_attributes": True,
    }

    @field_validator("start_time", "end_time")
    @classmethod
    def validate_time_format(cls, value: str) -> str:
        parse_time_to_minutes(value)
        return value

    @model_validator(mode="after")
    def validate_time_range(self) -> "TimetableChangeRequestProposalIn":
        if parse_time_to_minutes(self.end_time) <= parse_time_to_minutes(self.start_time):
            raise ValueError("endTime must be after startTime")
        return self

    @field_validator("assistant_faculty_ids")
    @classmethod
    def normalize_assistant_faculty_ids(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        ordered: list[str] = []
        seen: set[str] = set()
        for item in value:
            faculty_id = str(item or "").strip()
            if not faculty_id or faculty_id in seen:
                continue
            seen.add(faculty_id)
            ordered.append(faculty_id)
        return ordered


class TimetableChangeRequestDecisionIn(BaseModel):
    decision: Literal["approve", "reject"]
    note: str | None = Field(default=None, max_length=1000)


class TimetableChangeRequestOut(BaseModel):
    id: str
    program_id: str | None = Field(default=None, alias="programId")
    term_number: int | None = Field(default=None, alias="termNumber")
    slot_id: str = Field(alias="slotId")

    requested_by_id: str = Field(alias="requestedById")
    requested_by_role: str = Field(alias="requestedByRole")
    requested_by_name: str | None = Field(default=None, alias="requestedByName")
    approver_user_id: str | None = Field(default=None, alias="approverUserId")
    approver_role: str | None = Field(default=None, alias="approverRole")
    approver_name: str | None = Field(default=None, alias="approverName")

    status: TimetableChangeRequestStatus
    proposal: dict
    request_note: str | None = Field(default=None, alias="requestNote")
    decision_note: str | None = Field(default=None, alias="decisionNote")
    resolution_note: str | None = Field(default=None, alias="resolutionNote")

    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime | None = Field(default=None, alias="updatedAt")
    decided_at: datetime | None = Field(default=None, alias="decidedAt")
    applied_at: datetime | None = Field(default=None, alias="appliedAt")

    model_config = {
        "populate_by_name": True,
        "from_attributes": True,
    }


class TimetableChangeRequestDecisionOut(BaseModel):
    request: TimetableChangeRequestOut
    message: str
