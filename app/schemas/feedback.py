from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from app.models.feedback import FeedbackCategory, FeedbackPriority, FeedbackStatus
from app.models.user import UserRole


class FeedbackCreate(BaseModel):
    subject: str = Field(min_length=3, max_length=200)
    category: FeedbackCategory = FeedbackCategory.other
    priority: FeedbackPriority = FeedbackPriority.medium
    message: str = Field(min_length=5, max_length=5000)

    @field_validator("subject", "message")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Field cannot be empty")
        return cleaned


class FeedbackMessageCreate(BaseModel):
    message: str = Field(min_length=1, max_length=5000)

    @field_validator("message")
    @classmethod
    def normalize_message(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Message cannot be empty")
        return cleaned


class FeedbackUpdate(BaseModel):
    status: FeedbackStatus | None = None
    priority: FeedbackPriority | None = None
    assigned_admin_id: str | None = Field(default=None, max_length=36)


class FeedbackMessageOut(BaseModel):
    id: str
    feedback_id: str
    author_id: str
    author_role: UserRole
    message: str
    created_at: datetime

    model_config = {"from_attributes": True}


class FeedbackOut(BaseModel):
    id: str
    reporter_id: str
    reporter_name: str | None = None
    reporter_role: UserRole | None = None
    subject: str
    category: FeedbackCategory
    priority: FeedbackPriority
    status: FeedbackStatus
    assigned_admin_id: str | None
    resolved_at: datetime | None
    latest_message_at: datetime
    created_at: datetime
    updated_at: datetime | None
    message_count: int = 0
    latest_message_preview: str | None = None

    model_config = {"from_attributes": True}


class FeedbackDetailOut(FeedbackOut):
    messages: list[FeedbackMessageOut] = Field(default_factory=list)
