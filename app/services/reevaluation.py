from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.models.timetable import OfficialTimetable
from app.models.timetable_generation import ReevaluationStatus, TimetableReevaluationEvent
from app.models.user import User


def official_scope_impacted(
    db: Session,
    *,
    program_id: str,
    term_number: int | None,
) -> bool:
    record = db.get(OfficialTimetable, 1)
    if record is None:
        return False
    payload = record.payload or {}
    official_program_id = payload.get("programId") or payload.get("program_id")
    official_term_number = payload.get("termNumber") or payload.get("term_number")
    if official_program_id != program_id:
        return False
    if term_number is None:
        return True
    return official_term_number == term_number


def record_curriculum_change(
    db: Session,
    *,
    program_id: str,
    term_number: int | None,
    change_type: str,
    entity_type: str,
    entity_id: str | None,
    description: str,
    details: dict | None = None,
    triggered_by: User | None = None,
) -> TimetableReevaluationEvent:
    event = TimetableReevaluationEvent(
        program_id=program_id,
        term_number=term_number,
        change_type=change_type,
        entity_type=entity_type,
        entity_id=entity_id,
        description=description,
        details=details or {},
        status=ReevaluationStatus.pending,
        triggered_by_id=triggered_by.id if triggered_by is not None else None,
    )
    db.add(event)
    return event


def list_reevaluation_events(
    db: Session,
    *,
    program_id: str | None = None,
    term_number: int | None = None,
    status: ReevaluationStatus | None = None,
) -> list[TimetableReevaluationEvent]:
    query = select(TimetableReevaluationEvent)
    if program_id is not None:
        query = query.where(TimetableReevaluationEvent.program_id == program_id)
    if term_number is not None:
        query = query.where(TimetableReevaluationEvent.term_number == term_number)
    if status is not None:
        query = query.where(TimetableReevaluationEvent.status == status)
    query = query.order_by(TimetableReevaluationEvent.triggered_at.desc())
    return list(db.execute(query).scalars())


def resolve_reevaluation_events(
    db: Session,
    *,
    program_id: str,
    term_number: int,
    resolved_by: User | None = None,
    resolution_note: str | None = None,
) -> list[TimetableReevaluationEvent]:
    events = list(
        db.execute(
            select(TimetableReevaluationEvent).where(
                TimetableReevaluationEvent.program_id == program_id,
                TimetableReevaluationEvent.status == ReevaluationStatus.pending,
                or_(
                    TimetableReevaluationEvent.term_number == term_number,
                    TimetableReevaluationEvent.term_number.is_(None),
                ),
            )
        ).scalars()
    )
    if not events:
        return []
    now = datetime.now(tz=timezone.utc)
    for event in events:
        event.status = ReevaluationStatus.resolved
        event.resolved_by_id = resolved_by.id if resolved_by is not None else None
        event.resolved_at = now
        event.resolution_note = resolution_note
    return events
