from __future__ import annotations

from datetime import datetime, timezone
import logging

from anyio import from_thread
from fastapi import BackgroundTasks
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.notification import Notification, NotificationType
from app.models.user import User, UserRole
from app.services.email import EmailDeliveryError, send_email
from app.services.notification_hub import notification_hub

logger = logging.getLogger(__name__)


def _safe_iso(value: datetime | None) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc).isoformat()
    return value.isoformat()


def notification_to_event_payload(notification: Notification, *, event: str = "notification.created") -> dict:
    return {
        "event": event,
        "notification": {
            "id": notification.id,
            "user_id": notification.user_id,
            "title": notification.title,
            "message": notification.message,
            "notification_type": notification.notification_type.value,
            "is_read": notification.is_read,
            "created_at": _safe_iso(notification.created_at),
        },
    }


def publish_realtime_notification(notification: Notification, *, event: str = "notification.created") -> None:
    payload = notification_to_event_payload(notification, event=event)
    try:
        from_thread.run(notification_hub.publish, notification.user_id, payload)
    except Exception:  # pragma: no cover - runtime environment dependent
        logger.debug("Unable to push realtime notification for user %s", notification.user_id, exc_info=True)


def _send_notification_email(recipient: User, *, title: str, message: str) -> bool:
    if not recipient.email:
        return True
    try:
        send_email(
            to_email=recipient.email,
            subject=f"ShedForge Notification: {title}",
            text_content=f"{title}\n\n{message}",
        )
        return True
    except EmailDeliveryError as exc:  # pragma: no cover - transport behavior
        logger.warning("Notification email delivery failed for %s", recipient.email, exc_info=True)
        # If the error is a rate limit, return False to signal "stop trying"
        if "rate-limited" in str(exc).lower():
            return False
        return True


def create_notification(
    db: Session,
    *,
    user_id: str,
    title: str,
    message: str,
    notification_type: NotificationType = NotificationType.system,
    recipient: User | None = None,
    deliver_email: bool = False,
    deliver_realtime: bool = True,
    background_tasks: BackgroundTasks | None = None,
) -> tuple[Notification, bool]:
    # Set application-side timestamp for deterministic ordering even on DBs with coarse NOW() precision.
    created_at = datetime.now(timezone.utc)
    record = Notification(
        user_id=user_id,
        title=title,
        message=message,
        notification_type=notification_type,
        created_at=created_at,
    )
    db.add(record)
    db.flush()

    email_success = True
    if deliver_email and recipient is not None:
        if background_tasks:
            background_tasks.add_task(_send_notification_email, recipient, title=title, message=message)
        else:
            email_success = _send_notification_email(recipient, title=title, message=message)
    if deliver_realtime:
        publish_realtime_notification(record, event="notification.created")
    return record, email_success


def notify_users(
    db: Session,
    *,
    user_ids: list[str] | set[str] | tuple[str, ...],
    title: str,
    message: str,
    notification_type: NotificationType = NotificationType.system,
    exclude_user_id: str | None = None,
    deliver_email: bool = False,
    background_tasks: BackgroundTasks | None = None,
) -> list[Notification]:
    requested_ids = [item for item in dict.fromkeys(user_ids) if item and item != exclude_user_id]
    if not requested_ids:
        return []

    recipients = list(
        db.execute(
            select(User).where(
                User.id.in_(requested_ids),
                User.is_active.is_(True),
            )
        ).scalars()
    )
    results: list[Notification] = []
    for recipient in recipients:
        record, _ = create_notification(
            db,
            user_id=recipient.id,
            title=title,
            message=message,
            notification_type=notification_type,
            recipient=recipient,
            deliver_email=deliver_email,
            background_tasks=background_tasks,
        )
        results.append(record)
    return results


def notify_roles(
    db: Session,
    *,
    roles: list[UserRole] | set[UserRole] | tuple[UserRole, ...],
    title: str,
    message: str,
    notification_type: NotificationType = NotificationType.system,
    exclude_user_id: str | None = None,
    deliver_email: bool = False,
    background_tasks: BackgroundTasks | None = None,
) -> list[Notification]:
    if not roles:
        return []
    recipients = list(
        db.execute(
            select(User).where(
                User.role.in_(list(roles)),
                User.is_active.is_(True),
            )
        ).scalars()
    )
    results: list[Notification] = []
    for recipient in recipients:
        if exclude_user_id and recipient.id == exclude_user_id:
            continue
        record, _ = create_notification(
            db,
            user_id=recipient.id,
            title=title,
            message=message,
            notification_type=notification_type,
            recipient=recipient,
            deliver_email=deliver_email,
            background_tasks=background_tasks,
        )
        results.append(record)
    return results


def notify_all_users(
    db: Session,
    *,
    title: str,
    message: str,
    notification_type: NotificationType = NotificationType.system,
    exclude_user_id: str | None = None,
    deliver_email: bool = False,
    background_tasks: BackgroundTasks | None = None,
) -> list[Notification]:
    recipients = list(db.execute(select(User).where(User.is_active.is_(True))).scalars())
    results: list[Notification] = []
    for recipient in recipients:
        if exclude_user_id and recipient.id == exclude_user_id:
            continue
        record, _ = create_notification(
            db,
            user_id=recipient.id,
            title=title,
            message=message,
            notification_type=notification_type,
            recipient=recipient,
            deliver_email=deliver_email,
            background_tasks=background_tasks,
        )
        results.append(record)
    return results


def notify_admin_update(
    db: Session,
    *,
    title: str,
    message: str,
    actor_user_id: str | None = None,
    include_students: bool = False,
    deliver_email: bool = False,
    background_tasks: BackgroundTasks | None = None,
) -> list[Notification]:
    roles: list[UserRole] = [UserRole.admin, UserRole.scheduler, UserRole.faculty]
    if include_students:
        roles.append(UserRole.student)
    return notify_roles(
        db,
        roles=roles,
        title=title,
        message=message,
        notification_type=NotificationType.system,
        exclude_user_id=actor_user_id,
        deliver_email=deliver_email,
        background_tasks=background_tasks,
    )
