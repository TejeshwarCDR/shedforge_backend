from collections import defaultdict
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db, require_roles
from app.models.notification import NotificationType
from app.models.timetable_issue import IssueCategory, IssueMessage, IssueStatus, TimetableIssue
from app.models.user import User, UserRole
from app.schemas.issue import (
    IssueCreate,
    IssueDetailOut,
    IssueMessageCreate,
    IssueMessageOut,
    IssueOut,
    IssueUpdate,
)
from app.services.audit import log_activity
from app.services.notifications import notify_all_users, notify_roles, notify_users

router = APIRouter()


def _is_manager(user: User) -> bool:
    return user.role in {UserRole.admin, UserRole.scheduler}


def _issue_access_check(issue: TimetableIssue, current_user: User) -> None:
    if _is_manager(current_user):
        return
    if issue.reporter_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")


def _issue_message_to_out(message: IssueMessage) -> IssueMessageOut:
    return IssueMessageOut(
        id=message.id,
        issue_id=message.issue_id,
        author_id=message.author_id,
        author_role=UserRole(message.author_role),
        message=message.message,
        created_at=message.created_at,
    )


def _issue_message_stats(db: Session, issue_ids: list[str]) -> tuple[dict[str, int], dict[str, str]]:
    if not issue_ids:
        return {}, {}

    rows = list(
        db.execute(
            select(IssueMessage)
            .where(IssueMessage.issue_id.in_(issue_ids))
            .order_by(IssueMessage.created_at.desc())
        ).scalars()
    )
    counts: dict[str, int] = defaultdict(int)
    previews: dict[str, str] = {}
    for row in rows:
        counts[row.issue_id] += 1
        if row.issue_id not in previews:
            previews[row.issue_id] = row.message[:180]
    return counts, previews


def _issue_to_out(
    *,
    issue: TimetableIssue,
    reporter: User | None,
    message_count: int,
    latest_message_preview: str | None,
) -> IssueOut:
    return IssueOut(
        id=issue.id,
        reporter_id=issue.reporter_id,
        reporter_name=reporter.name if reporter else None,
        reporter_role=reporter.role if reporter else None,
        category=issue.category,
        affected_slot_id=issue.affected_slot_id,
        description=issue.description,
        status=issue.status,
        resolution_notes=issue.resolution_notes,
        assigned_to_id=issue.assigned_to_id,
        created_at=issue.created_at,
        updated_at=issue.updated_at,
        message_count=message_count,
        latest_message_preview=latest_message_preview,
    )


@router.get("/issues", response_model=list[IssueOut])
def list_issues(
    status_filter: IssueStatus | None = Query(default=None, alias="status"),
    category: IssueCategory | None = Query(default=None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[IssueOut]:
    query = select(TimetableIssue).order_by(TimetableIssue.created_at.desc())
    if status_filter is not None:
        query = query.where(TimetableIssue.status == status_filter)
    if category is not None:
        query = query.where(TimetableIssue.category == category)
    if not _is_manager(current_user):
        query = query.where(TimetableIssue.reporter_id == current_user.id)

    issue_rows = list(db.execute(query).scalars())
    issue_ids = [item.id for item in issue_rows]
    message_counts, message_previews = _issue_message_stats(db, issue_ids)

    reporter_ids = {item.reporter_id for item in issue_rows}
    reporters = (
        list(db.execute(select(User).where(User.id.in_(reporter_ids))).scalars())
        if reporter_ids
        else []
    )
    reporter_map = {item.id: item for item in reporters}

    return [
        _issue_to_out(
            issue=item,
            reporter=reporter_map.get(item.reporter_id),
            message_count=message_counts.get(item.id, 0),
            latest_message_preview=message_previews.get(item.id),
        )
        for item in issue_rows
    ]


@router.post("/issues", response_model=IssueOut, status_code=status.HTTP_201_CREATED)
def create_issue(
    payload: IssueCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> IssueOut:
    if current_user.role == UserRole.student and not (current_user.section_name or "").strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Student section is required before submitting timetable requests.",
        )

    now = datetime.now(timezone.utc)
    issue = TimetableIssue(
        reporter_id=current_user.id,
        category=payload.category,
        affected_slot_id=payload.affected_slot_id,
        description=payload.description,
        updated_at=now,
    )
    db.add(issue)
    db.flush()

    first_message = IssueMessage(
        issue_id=issue.id,
        author_id=current_user.id,
        author_role=current_user.role.value,
        message=payload.description,
    )
    db.add(first_message)

    notify_all_users(
        db,
        title="New Timetable Issue Reported",
        message=f"{current_user.name} reported a timetable issue ({payload.category.value}).",
        notification_type=NotificationType.issue,
        exclude_user_id=current_user.id,
    )
    log_activity(
        db,
        user=current_user,
        action="issue.create",
        entity_type="issue",
        entity_id=issue.id,
        details={"category": payload.category.value, "affected_slot_id": payload.affected_slot_id},
    )
    db.commit()
    db.refresh(issue)
    return _issue_to_out(
        issue=issue,
        reporter=current_user,
        message_count=1,
        latest_message_preview=payload.description[:180],
    )


@router.get("/issues/{issue_id}", response_model=IssueDetailOut)
def get_issue(
    issue_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> IssueDetailOut:
    issue = db.get(TimetableIssue, issue_id)
    if issue is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Issue not found")
    _issue_access_check(issue, current_user)

    reporter = db.get(User, issue.reporter_id)
    messages = list(
        db.execute(
            select(IssueMessage)
            .where(IssueMessage.issue_id == issue_id)
            .order_by(IssueMessage.created_at.asc())
        ).scalars()
    )
    latest_preview = messages[-1].message[:180] if messages else None
    return IssueDetailOut(
        **_issue_to_out(
            issue=issue,
            reporter=reporter,
            message_count=len(messages),
            latest_message_preview=latest_preview,
        ).model_dump(),
        messages=[_issue_message_to_out(item) for item in messages],
    )


@router.post("/issues/{issue_id}/messages", response_model=IssueMessageOut, status_code=status.HTTP_201_CREATED)
def add_issue_message(
    issue_id: str,
    payload: IssueMessageCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> IssueMessageOut:
    issue = db.get(TimetableIssue, issue_id)
    if issue is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Issue not found")
    _issue_access_check(issue, current_user)

    now = datetime.now(timezone.utc)
    message = IssueMessage(
        issue_id=issue.id,
        author_id=current_user.id,
        author_role=current_user.role.value,
        message=payload.message,
    )
    db.add(message)
    issue.updated_at = now

    if _is_manager(current_user):
        if issue.status == IssueStatus.open:
            issue.status = IssueStatus.in_progress
        notify_users(
            db,
            user_ids=[issue.reporter_id],
            title="Update on Your Timetable Issue",
            message=f"Admin responded on issue '{issue.id[:8]}...': {payload.message[:180]}",
            notification_type=NotificationType.issue,
            deliver_email=True,
        )
    else:
        if issue.status == IssueStatus.resolved:
            issue.status = IssueStatus.in_progress
        notify_roles(
            db,
            roles=[UserRole.admin, UserRole.scheduler],
            title="New Reply on Timetable Issue",
            message=f"{current_user.name} replied on issue '{issue.id[:8]}...'.",
            notification_type=NotificationType.issue,
            exclude_user_id=current_user.id,
            deliver_email=True,
        )

    log_activity(
        db,
        user=current_user,
        action="issue.message.create",
        entity_type="issue",
        entity_id=issue.id,
        details={"message_preview": payload.message[:120]},
    )
    db.commit()
    db.refresh(message)
    return _issue_message_to_out(message)


@router.put("/issues/{issue_id}", response_model=IssueOut)
def update_issue(
    issue_id: str,
    payload: IssueUpdate,
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> IssueOut:
    issue = db.get(TimetableIssue, issue_id)
    if issue is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Issue not found")

    previous_status = issue.status
    data = payload.model_dump(exclude_unset=True)
    for key, value in data.items():
        setattr(issue, key, value)
    issue.updated_at = datetime.now(timezone.utc)
    db.flush()

    if "status" in data and previous_status != issue.status:
        notify_users(
            db,
            user_ids=[issue.reporter_id],
            title="Issue Status Updated",
            message=f"Your reported issue is now '{issue.status.value}'.",
            notification_type=NotificationType.issue,
            deliver_email=True,
        )

    log_activity(
        db,
        user=current_user,
        action="issue.update",
        entity_type="issue",
        entity_id=issue_id,
        details=data,
    )
    db.commit()
    db.refresh(issue)

    reporter = db.get(User, issue.reporter_id)
    count, preview = _issue_message_stats(db, [issue.id])
    return _issue_to_out(
        issue=issue,
        reporter=reporter,
        message_count=count.get(issue.id, 0),
        latest_message_preview=preview.get(issue.id),
    )
