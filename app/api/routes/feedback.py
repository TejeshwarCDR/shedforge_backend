from collections import defaultdict
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db, require_roles
from app.models.feedback import (
    FeedbackCategory,
    FeedbackItem,
    FeedbackMessage,
    FeedbackPriority,
    FeedbackStatus,
)
from app.models.notification import NotificationType
from app.models.user import User, UserRole
from app.schemas.feedback import (
    FeedbackCreate,
    FeedbackDetailOut,
    FeedbackMessageCreate,
    FeedbackMessageOut,
    FeedbackOut,
    FeedbackUpdate,
)
from app.services.audit import log_activity
from app.services.notifications import notify_roles, notify_users

router = APIRouter()


def _is_admin(user: User) -> bool:
    return user.role == UserRole.admin


def _message_to_out(message: FeedbackMessage) -> FeedbackMessageOut:
    return FeedbackMessageOut(
        id=message.id,
        feedback_id=message.feedback_id,
        author_id=message.author_id,
        author_role=UserRole(message.author_role),
        message=message.message,
        created_at=message.created_at,
    )


def _feedback_to_out(
    *,
    feedback: FeedbackItem,
    reporter: User | None,
    message_count: int,
    latest_message_preview: str | None,
) -> FeedbackOut:
    return FeedbackOut(
        id=feedback.id,
        reporter_id=feedback.reporter_id,
        reporter_name=reporter.name if reporter else None,
        reporter_role=reporter.role if reporter else None,
        subject=feedback.subject,
        category=feedback.category,
        priority=feedback.priority,
        status=feedback.status,
        assigned_admin_id=feedback.assigned_admin_id,
        resolved_at=feedback.resolved_at,
        latest_message_at=feedback.latest_message_at,
        created_at=feedback.created_at,
        updated_at=feedback.updated_at,
        message_count=message_count,
        latest_message_preview=latest_message_preview,
    )


def _feedback_access_check(feedback: FeedbackItem, current_user: User) -> None:
    if _is_admin(current_user):
        return
    if feedback.reporter_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")


def _feedback_message_stats(
    db: Session,
    feedback_ids: list[str],
) -> tuple[dict[str, int], dict[str, str]]:
    if not feedback_ids:
        return {}, {}

    rows = list(
        db.execute(
            select(FeedbackMessage)
            .where(FeedbackMessage.feedback_id.in_(feedback_ids))
            .order_by(FeedbackMessage.created_at.desc())
        ).scalars()
    )
    counts: dict[str, int] = defaultdict(int)
    previews: dict[str, str] = {}
    for row in rows:
        counts[row.feedback_id] += 1
        if row.feedback_id not in previews:
            previews[row.feedback_id] = row.message[:180]
    return counts, previews


@router.get("/feedback", response_model=list[FeedbackOut])
def list_feedback(
    status_filter: FeedbackStatus | None = Query(default=None, alias="status"),
    category: FeedbackCategory | None = Query(default=None),
    priority: FeedbackPriority | None = Query(default=None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[FeedbackOut]:
    query = select(FeedbackItem).order_by(FeedbackItem.latest_message_at.desc())
    if status_filter is not None:
        query = query.where(FeedbackItem.status == status_filter)
    if category is not None:
        query = query.where(FeedbackItem.category == category)
    if priority is not None:
        query = query.where(FeedbackItem.priority == priority)
    if not _is_admin(current_user):
        query = query.where(FeedbackItem.reporter_id == current_user.id)

    feedback_rows = list(db.execute(query).scalars())
    feedback_ids = [item.id for item in feedback_rows]
    message_counts, message_previews = _feedback_message_stats(db, feedback_ids)

    reporter_ids = {item.reporter_id for item in feedback_rows}
    reporters = (
        list(db.execute(select(User).where(User.id.in_(reporter_ids))).scalars())
        if reporter_ids
        else []
    )
    reporter_map = {item.id: item for item in reporters}

    return [
        _feedback_to_out(
            feedback=item,
            reporter=reporter_map.get(item.reporter_id),
            message_count=message_counts.get(item.id, 0),
            latest_message_preview=message_previews.get(item.id),
        )
        for item in feedback_rows
    ]


@router.post("/feedback", response_model=FeedbackOut, status_code=status.HTTP_201_CREATED)
def create_feedback(
    payload: FeedbackCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> FeedbackOut:
    if current_user.role == UserRole.student and not (current_user.section_name or "").strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Student section is required before submitting feedback.",
        )
    now = datetime.now(timezone.utc)
    feedback = FeedbackItem(
        reporter_id=current_user.id,
        subject=payload.subject,
        category=payload.category,
        priority=payload.priority,
        status=FeedbackStatus.open,
        latest_message_at=now,
    )
    db.add(feedback)
    db.flush()

    message = FeedbackMessage(
        feedback_id=feedback.id,
        author_id=current_user.id,
        author_role=current_user.role.value,
        message=payload.message,
    )
    db.add(message)

    notify_roles(
        db,
        roles=[UserRole.admin],
        title="New Feedback Submitted",
        message=f"{current_user.name} ({current_user.role.value}) submitted feedback: {payload.subject}",
        notification_type=NotificationType.feedback,
        exclude_user_id=current_user.id,
        deliver_email=True,
    )
    log_activity(
        db,
        user=current_user,
        action="feedback.create",
        entity_type="feedback",
        entity_id=feedback.id,
        details={"category": payload.category.value, "priority": payload.priority.value},
    )
    db.commit()
    db.refresh(feedback)

    return _feedback_to_out(
        feedback=feedback,
        reporter=current_user,
        message_count=1,
        latest_message_preview=payload.message[:180],
    )


@router.get("/feedback/{feedback_id}", response_model=FeedbackDetailOut)
def get_feedback(
    feedback_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> FeedbackDetailOut:
    feedback = db.get(FeedbackItem, feedback_id)
    if feedback is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Feedback not found")
    _feedback_access_check(feedback, current_user)

    reporter = db.get(User, feedback.reporter_id)
    messages = list(
        db.execute(
            select(FeedbackMessage)
            .where(FeedbackMessage.feedback_id == feedback_id)
            .order_by(FeedbackMessage.created_at.asc())
        ).scalars()
    )
    latest_preview = messages[-1].message[:180] if messages else None

    return FeedbackDetailOut(
        **_feedback_to_out(
            feedback=feedback,
            reporter=reporter,
            message_count=len(messages),
            latest_message_preview=latest_preview,
        ).model_dump(),
        messages=[_message_to_out(item) for item in messages],
    )


@router.post("/feedback/{feedback_id}/messages", response_model=FeedbackMessageOut, status_code=status.HTTP_201_CREATED)
def add_feedback_message(
    feedback_id: str,
    payload: FeedbackMessageCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> FeedbackMessageOut:
    feedback = db.get(FeedbackItem, feedback_id)
    if feedback is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Feedback not found")
    _feedback_access_check(feedback, current_user)

    now = datetime.now(timezone.utc)
    message = FeedbackMessage(
        feedback_id=feedback.id,
        author_id=current_user.id,
        author_role=current_user.role.value,
        message=payload.message,
    )
    db.add(message)
    feedback.latest_message_at = now

    if _is_admin(current_user):
        if feedback.status in {FeedbackStatus.open, FeedbackStatus.under_review}:
            feedback.status = FeedbackStatus.awaiting_user
        notify_users(
            db,
            user_ids=[feedback.reporter_id],
            title="Admin Responded to Your Feedback",
            message=f"Update on '{feedback.subject}': {payload.message[:180]}",
            notification_type=NotificationType.feedback,
            deliver_email=True,
        )
    else:
        if feedback.status in {FeedbackStatus.awaiting_user, FeedbackStatus.resolved, FeedbackStatus.closed}:
            feedback.status = FeedbackStatus.under_review
            feedback.resolved_at = None
        notify_roles(
            db,
            roles=[UserRole.admin],
            title="New Feedback Reply",
            message=f"{current_user.name} replied on feedback '{feedback.subject}'.",
            notification_type=NotificationType.feedback,
            deliver_email=True,
        )

    log_activity(
        db,
        user=current_user,
        action="feedback.message.create",
        entity_type="feedback",
        entity_id=feedback.id,
        details={"message_preview": payload.message[:120]},
    )
    db.commit()
    db.refresh(message)
    return _message_to_out(message)


@router.put("/feedback/{feedback_id}", response_model=FeedbackOut)
def update_feedback(
    feedback_id: str,
    payload: FeedbackUpdate,
    current_user: User = Depends(require_roles(UserRole.admin)),
    db: Session = Depends(get_db),
) -> FeedbackOut:
    feedback = db.get(FeedbackItem, feedback_id)
    if feedback is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Feedback not found")

    data = payload.model_dump(exclude_unset=True)
    if "assigned_admin_id" in data and data["assigned_admin_id"] is not None:
        assigned_admin = db.get(User, data["assigned_admin_id"])
        if assigned_admin is None or assigned_admin.role != UserRole.admin:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="assigned_admin_id must reference an admin user",
            )

    previous_status = feedback.status
    for key, value in data.items():
        setattr(feedback, key, value)

    if "status" in data:
        if feedback.status == FeedbackStatus.resolved:
            feedback.resolved_at = datetime.now(timezone.utc)
        elif feedback.status in {FeedbackStatus.open, FeedbackStatus.under_review, FeedbackStatus.awaiting_user}:
            feedback.resolved_at = None

    feedback.latest_message_at = datetime.now(timezone.utc)
    db.flush()

    if previous_status != feedback.status:
        notify_users(
            db,
            user_ids=[feedback.reporter_id],
            title="Feedback Status Updated",
            message=f"Your feedback '{feedback.subject}' is now '{feedback.status.value}'.",
            notification_type=NotificationType.feedback,
            deliver_email=True,
        )

    log_activity(
        db,
        user=current_user,
        action="feedback.update",
        entity_type="feedback",
        entity_id=feedback.id,
        details=data,
    )
    db.commit()
    db.refresh(feedback)

    reporter = db.get(User, feedback.reporter_id)
    count, preview = _feedback_message_stats(db, [feedback.id])
    return _feedback_to_out(
        feedback=feedback,
        reporter=reporter,
        message_count=count.get(feedback.id, 0),
        latest_message_preview=preview.get(feedback.id),
    )
