from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from jose import JWTError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db
from app.core.security import decode_token
from app.models.notification import Notification, NotificationType
from app.models.user import User
from app.schemas.notification import NotificationOut
from app.services.audit import log_activity
from app.services.notification_hub import notification_hub
from app.services.notifications import publish_realtime_notification

router = APIRouter()


@router.get("/notifications", response_model=list[NotificationOut])
def list_notifications(
    notification_type: NotificationType | None = Query(default=None),
    is_read: bool | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[NotificationOut]:
    query = (
        select(Notification)
        .where(Notification.user_id == current_user.id)
        .order_by(Notification.created_at.desc())
    )
    if notification_type:
        query = query.where(Notification.notification_type == notification_type)
    if is_read is not None:
        query = query.where(Notification.is_read == is_read)
    query = query.offset(offset).limit(limit)
    return list(db.execute(query).scalars())


@router.post("/notifications/{notification_id}/read", response_model=NotificationOut)
def mark_notification_read(
    notification_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> NotificationOut:
    notification = db.get(Notification, notification_id)
    if notification is None or notification.user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Notification not found")
    notification.is_read = True
    log_activity(
        db,
        user=current_user,
        action="notification.read",
        entity_type="notification",
        entity_id=notification_id,
    )
    db.commit()
    db.refresh(notification)
    publish_realtime_notification(notification, event="notification.read")
    return notification


@router.post("/notifications/read-all")
def mark_all_notifications_read(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict[str, int]:
    notifications = list(
        db.execute(
            select(Notification).where(
                Notification.user_id == current_user.id,
                Notification.is_read.is_(False),
            )
        ).scalars()
    )
    for notification in notifications:
        notification.is_read = True
        publish_realtime_notification(notification, event="notification.read")

    if notifications:
        log_activity(
            db,
            user=current_user,
            action="notification.read_all",
            entity_type="notification",
            details={"count": len(notifications)},
        )
    db.commit()
    return {"updated": len(notifications)}


def _extract_ws_token(websocket: WebSocket) -> str | None:
    token = websocket.query_params.get("token")
    if token:
        return token

    auth_header = websocket.headers.get("authorization")
    if not auth_header:
        return None
    scheme, _, value = auth_header.partition(" ")
    if scheme.lower() != "bearer":
        return None
    return value.strip() or None


@router.websocket("/notifications/ws")
async def notifications_websocket(
    websocket: WebSocket,
    db: Session = Depends(get_db),
) -> None:
    token = _extract_ws_token(websocket)
    if not token:
        await websocket.close(code=1008)
        return

    try:
        payload = decode_token(token)
        user_id = payload.get("sub")
    except JWTError:
        await websocket.close(code=1008)
        return

    if not user_id:
        await websocket.close(code=1008)
        return

    user = db.get(User, user_id)

    if user is None or not user.is_active:
        await websocket.close(code=1008)
        return

    await notification_hub.connect(user.id, websocket)
    try:
        await websocket.send_json({"event": "connected", "user_id": user.id})
        while True:
            message = await websocket.receive_text()
            if message.strip().lower() == "ping":
                await websocket.send_json({"event": "pong"})
    except WebSocketDisconnect:
        pass
    finally:
        await notification_hub.disconnect(user.id, websocket)
