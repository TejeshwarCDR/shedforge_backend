from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.notification import Notification, NotificationType
from app.models.user import User


def create_notification(
    db: Session,
    *,
    user_id: str,
    title: str,
    message: str,
    notification_type: NotificationType = NotificationType.system,
) -> Notification:
    record = Notification(
        user_id=user_id,
        title=title,
        message=message,
        notification_type=notification_type,
    )
    db.add(record)
    return record


def notify_all_users(
    db: Session,
    *,
    title: str,
    message: str,
    notification_type: NotificationType = NotificationType.system,
    exclude_user_id: str | None = None,
) -> None:
    users = db.execute(select(User.id)).scalars().all()
    for user_id in users:
        if exclude_user_id and user_id == exclude_user_id:
            continue
        create_notification(
            db,
            user_id=user_id,
            title=title,
            message=message,
            notification_type=notification_type,
        )
