from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.models.leave_request import LeaveStatus, LeaveType
from app.models.leave_substitute_offer import LeaveSubstituteOfferStatus


class LeaveRequestCreate(BaseModel):
    leave_date: date
    leave_type: LeaveType
    reason: str = Field(min_length=3, max_length=1000)


class LeaveRequestStatusUpdate(BaseModel):
    status: LeaveStatus
    admin_comment: str | None = Field(default=None, max_length=1000)


class LeaveSubstituteAssignmentCreate(BaseModel):
    substitute_faculty_id: str = Field(min_length=1, max_length=36)
    notes: str | None = Field(default=None, max_length=1000)


class LeaveSwapOfferCreate(BaseModel):
    slot_id: str = Field(min_length=1, max_length=36)
    substitute_faculty_id: str = Field(min_length=1, max_length=36)


class LeaveSubstituteAssignmentOut(BaseModel):
    id: str
    leave_request_id: str
    substitute_faculty_id: str
    substitute_faculty_name: str | None = None
    substitute_faculty_email: str | None = None
    assigned_by_id: str
    notes: str | None = None
    created_at: datetime
    updated_at: datetime | None = None

    model_config = {"from_attributes": True}


class LeaveRequestOut(BaseModel):
    id: str
    user_id: str
    faculty_id: str | None = None
    leave_date: date
    leave_type: LeaveType
    reason: str
    status: LeaveStatus
    admin_comment: str | None = None
    reviewed_by_id: str | None = None
    reviewed_at: datetime | None = None
    substitute_assignment: LeaveSubstituteAssignmentOut | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class LeaveSubstituteOfferRespond(BaseModel):
    decision: Literal["accept", "reject"]
    response_note: str | None = Field(default=None, max_length=1000)


class LeaveSubstituteOfferOut(BaseModel):
    id: str
    leave_request_id: str
    slot_id: str
    substitute_faculty_id: str
    substitute_faculty_name: str | None = None
    substitute_faculty_email: str | None = None
    offered_by_id: str
    status: LeaveSubstituteOfferStatus
    expires_at: datetime | None = None
    responded_at: datetime | None = None
    response_note: str | None = None
    created_at: datetime
    updated_at: datetime | None = None
    leave_date: date | None = None
    absent_faculty_id: str | None = None
    absent_faculty_name: str | None = None
    day: str | None = None
    start_time: str | None = Field(default=None, alias="startTime")
    end_time: str | None = Field(default=None, alias="endTime")
    section: str | None = None
    batch: str | None = None
    course_code: str | None = None
    course_name: str | None = None
    room_name: str | None = None

    model_config = {
        "from_attributes": True,
        "populate_by_name": True,
    }
