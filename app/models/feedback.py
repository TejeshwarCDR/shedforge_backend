import uuid
from datetime import datetime
from enum import Enum

from sqlalchemy import DateTime, Enum as SAEnum, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.db.base import Base


class FeedbackStatus(str, Enum):
    open = "open"
    under_review = "under_review"
    awaiting_user = "awaiting_user"
    resolved = "resolved"
    closed = "closed"


class FeedbackCategory(str, Enum):
    timetable = "timetable"
    technical = "technical"
    usability = "usability"
    account = "account"
    suggestion = "suggestion"
    grievance = "grievance"
    other = "other"


class FeedbackPriority(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    urgent = "urgent"


class FeedbackItem(Base):
    __tablename__ = "feedback_items"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    reporter_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    subject: Mapped[str] = mapped_column(String(200), nullable=False)
    category: Mapped[FeedbackCategory] = mapped_column(
        SAEnum(FeedbackCategory, name="feedback_category"),
        nullable=False,
        default=FeedbackCategory.other,
    )
    priority: Mapped[FeedbackPriority] = mapped_column(
        SAEnum(FeedbackPriority, name="feedback_priority"),
        nullable=False,
        default=FeedbackPriority.medium,
    )
    status: Mapped[FeedbackStatus] = mapped_column(
        SAEnum(FeedbackStatus, name="feedback_status"),
        nullable=False,
        default=FeedbackStatus.open,
    )
    assigned_admin_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    latest_message_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), onupdate=func.now())


class FeedbackMessage(Base):
    __tablename__ = "feedback_messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    feedback_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    author_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    author_role: Mapped[str] = mapped_column(String(20), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
