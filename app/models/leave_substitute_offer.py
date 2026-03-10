import uuid
from datetime import datetime
from enum import Enum

from sqlalchemy import DateTime, Enum as SAEnum, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.db.base import Base


class LeaveSubstituteOfferStatus(str, Enum):
    pending = "pending"
    accepted = "accepted"
    rejected = "rejected"
    expired = "expired"
    superseded = "superseded"
    cancelled = "cancelled"
    rescheduled = "rescheduled"


class LeaveSubstituteOffer(Base):
    __tablename__ = "leave_substitute_offers"
    __table_args__ = (
        UniqueConstraint(
            "leave_request_id",
            "slot_id",
            "substitute_faculty_id",
            name="uq_leave_substitute_offer_identity",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    leave_request_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    slot_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    substitute_faculty_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    offered_by_id: Mapped[str] = mapped_column(String(36), nullable=False)
    status: Mapped[LeaveSubstituteOfferStatus] = mapped_column(
        SAEnum(LeaveSubstituteOfferStatus, name="leave_substitute_offer_status"),
        nullable=False,
        default=LeaveSubstituteOfferStatus.pending,
        index=True,
    )
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    responded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    response_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), onupdate=func.now())
