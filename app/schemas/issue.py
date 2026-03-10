from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from app.models.timetable_issue import IssueCategory, IssueStatus
from app.models.user import UserRole


class IssueCreate(BaseModel):
    category: IssueCategory = IssueCategory.other
    affected_slot_id: str | None = Field(default=None, max_length=100)
    description: str = Field(min_length=5, max_length=5000)

    @field_validator("description")
    @classmethod
    def normalize_description(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Description cannot be empty")
        return cleaned


class IssueMessageCreate(BaseModel):
    message: str = Field(min_length=1, max_length=5000)

    @field_validator("message")
    @classmethod
    def normalize_message(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Message cannot be empty")
        return cleaned


class IssueUpdate(BaseModel):
    status: IssueStatus | None = None
    resolution_notes: str | None = Field(default=None, max_length=5000)
    assigned_to_id: str | None = Field(default=None, max_length=36)


class IssueMessageOut(BaseModel):
    id: str
    issue_id: str
    author_id: str
    author_role: UserRole
    message: str
    created_at: datetime

    model_config = {"from_attributes": True}


class IssueOut(BaseModel):
    id: str
    reporter_id: str
    reporter_name: str | None = None
    reporter_role: UserRole | None = None
    category: IssueCategory
    affected_slot_id: str | None
    description: str
    status: IssueStatus
    resolution_notes: str | None
    assigned_to_id: str | None
    created_at: datetime
    updated_at: datetime | None
    message_count: int = 0
    latest_message_preview: str | None = None

    model_config = {"from_attributes": True}


class IssueDetailOut(IssueOut):
    messages: list[IssueMessageOut] = Field(default_factory=list)
