from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_db, require_roles
from app.models.activity_log import ActivityLog
from app.models.user import User, UserRole
from app.schemas.activity import ActivityLogOut

router = APIRouter()


@router.get("/activity/logs", response_model=list[ActivityLogOut])
def list_activity_logs(
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> list[ActivityLogOut]:
    query = select(ActivityLog).order_by(ActivityLog.created_at.desc()).limit(500)
    return list(db.execute(query).scalars())
