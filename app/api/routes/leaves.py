from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db, require_roles
from app.models.faculty import Faculty
from app.models.institution_settings import InstitutionSettings
from app.models.leave_request import LeaveRequest, LeaveStatus
from app.models.leave_substitute_assignment import LeaveSubstituteAssignment
from app.models.leave_substitute_offer import LeaveSubstituteOffer, LeaveSubstituteOfferStatus
from app.models.notification import NotificationType
from app.models.timetable import OfficialTimetable
from app.models.timetable_version import TimetableVersion
from app.models.user import User, UserRole
from app.schemas.leave import (
    LeaveRequestCreate,
    LeaveRequestOut,
    LeaveRequestStatusUpdate,
    LeaveSwapOfferCreate,
    LeaveSubstituteAssignmentCreate,
    LeaveSubstituteAssignmentOut,
    LeaveSubstituteOfferOut,
    LeaveSubstituteOfferRespond,
)
from app.schemas.settings import DEFAULT_SCHEDULE_POLICY, DEFAULT_WORKING_HOURS, BreakWindowEntry, WorkingHoursEntry
from app.schemas.timetable import FacultyPayload, OfficialTimetablePayload, parse_time_to_minutes
from app.services.audit import log_activity
from app.services.notifications import notify_users

router = APIRouter()

DAY_SHORT_MAP = {
    "Mon": "Monday",
    "Tue": "Tuesday",
    "Wed": "Wednesday",
    "Thu": "Thursday",
    "Fri": "Friday",
    "Sat": "Saturday",
    "Sun": "Sunday",
}
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
OFFER_EXPIRY_MINUTES = 15


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _normalize_text(value: str | None) -> str | None:
    if value is None:
        return None
    trimmed = value.strip()
    return trimmed or None


def _safe_email(value: str | None) -> str:
    return (value or "").strip().lower()


def _normalize_day(value: str) -> str:
    return DAY_SHORT_MAP.get(value, value)


def _slot_overlaps(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    return start_a < end_b and start_b < end_a


def _minutes_to_hhmm(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _faculty_has_overlapping_slot(
    payload: OfficialTimetablePayload,
    *,
    faculty_id: str,
    day: str,
    start_time: int,
    end_time: int,
    ignore_slot_id: str,
) -> bool:
    for item in payload.timetable_data:
        if item.id == ignore_slot_id:
            continue
        if item.facultyId != faculty_id or item.day != day:
            continue
        item_start = parse_time_to_minutes(item.startTime)
        item_end = parse_time_to_minutes(item.endTime)
        if _slot_overlaps(start_time, end_time, item_start, item_end):
            return True
    return False


def _faculty_minutes_assigned(payload: OfficialTimetablePayload, faculty_id: str) -> int:
    total = 0
    for item in payload.timetable_data:
        if item.facultyId != faculty_id:
            continue
        total += parse_time_to_minutes(item.endTime) - parse_time_to_minutes(item.startTime)
    return total


def _faculty_available_for_slot(faculty: Faculty, *, day: str, start_time: int, end_time: int) -> bool:
    availability = {_normalize_day(str(item).strip()) for item in (faculty.availability or []) if str(item).strip()}
    if availability and day not in availability:
        return False

    day_windows: list[tuple[int, int]] = []
    for window in faculty.availability_windows or []:
        window_day = _normalize_day(str(window.get("day", "")).strip())
        if window_day != day:
            continue
        start_raw = window.get("start_time")
        end_raw = window.get("end_time")
        if not start_raw or not end_raw:
            continue
        try:
            day_windows.append((parse_time_to_minutes(str(start_raw)), parse_time_to_minutes(str(end_raw))))
        except ValueError:
            continue

    if day_windows:
        return any(window_start <= start_time and end_time <= window_end for window_start, window_end in day_windows)
    return True


def _build_teaching_segments(
    *,
    day_start: int,
    day_end: int,
    period_minutes: int,
    breaks: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    break_windows = sorted(
        (
            (start, end)
            for start, end in breaks
            if end > day_start and start < day_end
        ),
        key=lambda item: item[0],
    )
    segments: list[tuple[int, int]] = []
    cursor = day_start
    break_index = 0
    while cursor + period_minutes <= day_end:
        while break_index < len(break_windows) and break_windows[break_index][1] <= cursor:
            break_index += 1

        if break_index < len(break_windows):
            break_start, break_end = break_windows[break_index]
            if break_start <= cursor < break_end:
                cursor = break_end
                continue
            if cursor < break_start < cursor + period_minutes:
                cursor = break_end
                continue

        next_cursor = cursor + period_minutes
        segments.append((cursor, next_cursor))
        cursor = next_cursor
    return segments


def _is_window_aligned_with_segments(start_time: int, end_time: int, segments: list[tuple[int, int]]) -> bool:
    if end_time <= start_time:
        return False
    by_start = {start: end for start, end in segments}
    cursor = start_time
    while cursor < end_time:
        next_boundary = by_start.get(cursor)
        if next_boundary is None:
            return False
        cursor = next_boundary
    return cursor == end_time


def _load_schedule_context(db: Session) -> tuple[int, list[tuple[int, int]], dict[str, tuple[int, int]]]:
    record = db.get(InstitutionSettings, 1)

    period_minutes = record.period_minutes if record and record.period_minutes else DEFAULT_SCHEDULE_POLICY.period_minutes

    break_raw = record.break_windows if record and record.break_windows else [
        item.model_dump() for item in DEFAULT_SCHEDULE_POLICY.breaks
    ]
    break_entries: list[BreakWindowEntry] = []
    for raw in break_raw:
        try:
            break_entries.append(BreakWindowEntry.model_validate(raw))
        except Exception:
            continue
    if not break_entries:
        break_entries = list(DEFAULT_SCHEDULE_POLICY.breaks)
    break_minutes = [
        (parse_time_to_minutes(item.start_time), parse_time_to_minutes(item.end_time))
        for item in break_entries
    ]

    working_raw = record.working_hours if record else [item.model_dump() for item in DEFAULT_WORKING_HOURS]
    working_entries: list[WorkingHoursEntry] = []
    for raw in working_raw:
        try:
            working_entries.append(WorkingHoursEntry.model_validate(raw))
        except Exception:
            continue
    if not working_entries:
        working_entries = list(DEFAULT_WORKING_HOURS)

    working_hours: dict[str, tuple[int, int]] = {}
    for item in working_entries:
        if not item.enabled:
            continue
        working_hours[item.day] = (
            parse_time_to_minutes(item.start_time),
            parse_time_to_minutes(item.end_time),
        )
    return period_minutes, break_minutes, working_hours


def _next_timetable_version_label(db: Session) -> str:
    labels = db.execute(select(TimetableVersion.label)).scalars().all()
    numbers: list[int] = []
    for label in labels:
        if not label.startswith("v"):
            continue
        suffix = label[1:]
        if suffix.isdigit():
            numbers.append(int(suffix))
    next_index = (max(numbers) + 1) if numbers else 1
    return f"v{next_index}"


def _ensure_faculty_in_payload(payload: OfficialTimetablePayload, faculty: Faculty) -> None:
    if any(item.id == faculty.id for item in payload.faculty_data):
        return
    payload.faculty_data.append(
        FacultyPayload(
            id=faculty.id,
            name=faculty.name,
            department=faculty.department,
            workloadHours=faculty.workload_hours,
            maxHours=faculty.max_hours,
            availability=faculty.availability or [],
            email=faculty.email,
            currentWorkload=faculty.workload_hours,
        )
    )


def _persist_payload_as_version(
    *,
    db: Session,
    official: OfficialTimetable,
    payload: OfficialTimetablePayload,
    current_user: User,
    summary: dict,
) -> str:
    payload_dict = payload.model_dump(by_alias=True)
    official.payload = payload_dict
    official.updated_by_id = current_user.id

    version_label = _next_timetable_version_label(db)
    db.add(
        TimetableVersion(
            label=version_label,
            payload=payload_dict,
            summary=summary,
            created_by_id=current_user.id,
        )
    )
    return version_label


def _load_official_payload(db: Session) -> tuple[OfficialTimetable | None, OfficialTimetablePayload | None]:
    official = db.get(OfficialTimetable, 1)
    if official is None:
        return None, None
    return official, OfficialTimetablePayload.model_validate(official.payload)


def _find_slot(payload: OfficialTimetablePayload, slot_id: str):
    for slot in payload.timetable_data:
        if slot.id == slot_id:
            return slot
    return None


def _prune_primary_from_slot_assistants(slot: object, primary_faculty_id: str) -> None:
    raw_assistants = getattr(slot, "assistant_faculty_ids", None)
    if not isinstance(raw_assistants, list):
        return
    slot.assistant_faculty_ids = [
        str(item).strip()
        for item in raw_assistants
        if str(item).strip() and str(item).strip() != primary_faculty_id
    ]


def _build_assignment_out(
    assignment: LeaveSubstituteAssignment,
    substitute_faculty: Faculty | None,
) -> LeaveSubstituteAssignmentOut:
    return LeaveSubstituteAssignmentOut(
        id=assignment.id,
        leave_request_id=assignment.leave_request_id,
        substitute_faculty_id=assignment.substitute_faculty_id,
        substitute_faculty_name=substitute_faculty.name if substitute_faculty else None,
        substitute_faculty_email=substitute_faculty.email if substitute_faculty else None,
        assigned_by_id=assignment.assigned_by_id,
        notes=assignment.notes,
        created_at=assignment.created_at,
        updated_at=assignment.updated_at,
    )


def _hydrate_leave_requests(
    db: Session,
    requests: list[LeaveRequest],
) -> list[LeaveRequestOut]:
    if not requests:
        return []

    request_ids = [item.id for item in requests]
    assignments = list(
        db.execute(
            select(LeaveSubstituteAssignment).where(
                LeaveSubstituteAssignment.leave_request_id.in_(request_ids)
            )
        ).scalars()
    )
    assignment_by_leave_id = {item.leave_request_id: item for item in assignments}
    substitute_ids = [item.substitute_faculty_id for item in assignments]
    substitute_faculty = {
        item.id: item
        for item in db.execute(select(Faculty).where(Faculty.id.in_(substitute_ids))).scalars()
    }

    output: list[LeaveRequestOut] = []
    for request in requests:
        assignment = assignment_by_leave_id.get(request.id)
        output.append(
            LeaveRequestOut(
                id=request.id,
                user_id=request.user_id,
                faculty_id=request.faculty_id,
                leave_date=request.leave_date,
                leave_type=request.leave_type,
                reason=request.reason,
                status=request.status,
                admin_comment=request.admin_comment,
                reviewed_by_id=request.reviewed_by_id,
                reviewed_at=request.reviewed_at,
                substitute_assignment=(
                    _build_assignment_out(
                        assignment,
                        substitute_faculty.get(assignment.substitute_faculty_id),
                    )
                    if assignment
                    else None
                ),
                created_at=request.created_at,
            )
        )
    return output


def _hydrate_substitute_offers(
    db: Session,
    offers: list[LeaveSubstituteOffer],
) -> list[LeaveSubstituteOfferOut]:
    if not offers:
        return []

    leave_ids = sorted({item.leave_request_id for item in offers})
    leaves_by_id = {
        item.id: item
        for item in db.execute(select(LeaveRequest).where(LeaveRequest.id.in_(leave_ids))).scalars()
    }

    faculty_ids: set[str] = {item.substitute_faculty_id for item in offers}
    for leave in leaves_by_id.values():
        if leave.faculty_id:
            faculty_ids.add(leave.faculty_id)

    faculty_by_id = {
        item.id: item
        for item in db.execute(select(Faculty).where(Faculty.id.in_(faculty_ids))).scalars()
    }

    _, payload = _load_official_payload(db)
    slot_by_id = {}
    course_by_id = {}
    room_by_id = {}
    if payload is not None:
        slot_by_id = {slot.id: slot for slot in payload.timetable_data}
        course_by_id = {course.id: course for course in payload.course_data}
        room_by_id = {room.id: room for room in payload.room_data}

    output: list[LeaveSubstituteOfferOut] = []
    for offer in offers:
        leave = leaves_by_id.get(offer.leave_request_id)
        substitute = faculty_by_id.get(offer.substitute_faculty_id)
        absent_faculty = faculty_by_id.get(leave.faculty_id) if leave and leave.faculty_id else None
        slot = slot_by_id.get(offer.slot_id)
        course = course_by_id.get(slot.courseId) if slot else None
        room = room_by_id.get(slot.roomId) if slot else None
        output.append(
            LeaveSubstituteOfferOut(
                id=offer.id,
                leave_request_id=offer.leave_request_id,
                slot_id=offer.slot_id,
                substitute_faculty_id=offer.substitute_faculty_id,
                substitute_faculty_name=substitute.name if substitute else None,
                substitute_faculty_email=substitute.email if substitute else None,
                offered_by_id=offer.offered_by_id,
                status=offer.status,
                expires_at=offer.expires_at,
                responded_at=offer.responded_at,
                response_note=offer.response_note,
                created_at=offer.created_at,
                updated_at=offer.updated_at,
                leave_date=leave.leave_date if leave else None,
                absent_faculty_id=leave.faculty_id if leave else None,
                absent_faculty_name=absent_faculty.name if absent_faculty else None,
                day=slot.day if slot else None,
                startTime=slot.startTime if slot else None,
                endTime=slot.endTime if slot else None,
                section=slot.section if slot else None,
                batch=slot.batch if slot else None,
                course_code=course.code if course else None,
                course_name=course.name if course else None,
                room_name=room.name if room else None,
            )
        )
    return output


def _get_faculty_for_user(db: Session, current_user: User) -> Faculty | None:
    email = _safe_email(current_user.email)
    if not email:
        return None
    return db.execute(select(Faculty).where(func.lower(Faculty.email) == email)).scalar_one_or_none()


def _collect_student_ids_for_sections(db: Session, sections: set[str]) -> list[str]:
    normalized = {item.strip().upper() for item in sections if item and item.strip()}
    if not normalized:
        return []
    return [
        item.id
        for item in db.execute(select(User).where(User.role == UserRole.student)).scalars()
        if (item.section_name or "").strip().upper() in normalized
    ]


def _slot_conflicts(
    payload: OfficialTimetablePayload,
    *,
    day: str,
    start_time: int,
    end_time: int,
    section: str | None = None,
    faculty_id: str | None = None,
    room_id: str | None = None,
    ignore_slot_id: str | None = None,
) -> bool:
    for existing in payload.timetable_data:
        if ignore_slot_id and existing.id == ignore_slot_id:
            continue
        if existing.day != day:
            continue
        existing_start = parse_time_to_minutes(existing.startTime)
        existing_end = parse_time_to_minutes(existing.endTime)
        if not _slot_overlaps(start_time, end_time, existing_start, existing_end):
            continue
        if section and existing.section == section:
            return True
        if faculty_id and existing.facultyId == faculty_id:
            return True
        if room_id and existing.roomId == room_id:
            return True
    return False


def _candidate_days_for_leave(leave_day: str) -> list[str]:
    if leave_day not in DAY_ORDER:
        return list(DAY_ORDER)
    index = DAY_ORDER.index(leave_day)
    return DAY_ORDER[index + 1 :] + DAY_ORDER[:index] + [leave_day]


def _eligible_substitute_candidates_for_slot(
    *,
    request: LeaveRequest,
    slot,
    course_code: str,
    leave_faculty: Faculty | None,
    payload: OfficialTimetablePayload,
    faculty_by_id: dict[str, Faculty],
    approved_leave_faculty_ids: set[str],
    minutes_cache: dict[str, int],
) -> list[Faculty]:
    day_name = slot.day
    slot_start = parse_time_to_minutes(slot.startTime)
    slot_end = parse_time_to_minutes(slot.endTime)
    slot_duration = slot_end - slot_start

    ranked: list[tuple[int, Faculty]] = []
    for candidate in faculty_by_id.values():
        if candidate.id == request.faculty_id:
            continue
        if candidate.id in approved_leave_faculty_ids:
            continue

        preferred_codes = {
            code.strip().upper() for code in (candidate.preferred_subject_codes or []) if code.strip()
        }
        if course_code not in preferred_codes:
            continue
        if not _faculty_available_for_slot(
            candidate,
            day=day_name,
            start_time=slot_start,
            end_time=slot_end,
        ):
            continue
        if _faculty_has_overlapping_slot(
            payload,
            faculty_id=candidate.id,
            day=day_name,
            start_time=slot_start,
            end_time=slot_end,
            ignore_slot_id=slot.id,
        ):
            continue

        assigned_minutes = minutes_cache.get(candidate.id)
        if assigned_minutes is None:
            assigned_minutes = _faculty_minutes_assigned(payload, candidate.id)
            minutes_cache[candidate.id] = assigned_minutes

        if assigned_minutes + slot_duration > candidate.max_hours * 60:
            continue

        score = candidate.max_hours * 60 - assigned_minutes
        if leave_faculty and candidate.department == leave_faculty.department:
            score += 120
        ranked.append((score, candidate))

    ranked.sort(key=lambda item: (item[0], item[1].name), reverse=True)
    return [item[1] for item in ranked]


def _reschedule_slot_for_leave(
    *,
    db: Session,
    request: LeaveRequest,
    payload: OfficialTimetablePayload,
    slot,
    course_by_id: dict[str, object],
    faculty_by_id: dict[str, Faculty],
) -> dict | None:
    if not request.faculty_id:
        return None

    course = course_by_id.get(slot.courseId)
    if course is None:
        return None

    leave_faculty = faculty_by_id.get(request.faculty_id)
    slot_start = parse_time_to_minutes(slot.startTime)
    slot_end = parse_time_to_minutes(slot.endTime)
    duration = slot_end - slot_start
    if duration <= 0:
        return None

    period_minutes, break_windows, working_hours = _load_schedule_context(db)
    is_lab = getattr(course, "type", None) == "lab" or slot.sessionType == "lab"
    required_capacity = slot.studentCount or 0

    old_day = slot.day
    old_start = slot.startTime
    old_end = slot.endTime
    old_room_id = slot.roomId

    room_candidates = [room for room in payload.room_data if not is_lab or room.type == "lab"]
    if not is_lab:
        room_candidates = [room for room in room_candidates if room.type != "lab"]
    if required_capacity:
        room_candidates = [room for room in room_candidates if room.capacity >= required_capacity]

    if not room_candidates:
        return None

    for day_name in _candidate_days_for_leave(request.leave_date.strftime("%A")):
        working_window = working_hours.get(day_name)
        if working_window is None:
            continue
        day_start, day_end = working_window
        segments = _build_teaching_segments(
            day_start=day_start,
            day_end=day_end,
            period_minutes=period_minutes,
            breaks=break_windows,
        )
        if not segments:
            continue

        for start_minute, _ in segments:
            end_minute = start_minute + duration
            if not _is_window_aligned_with_segments(start_minute, end_minute, segments):
                continue
            if _slot_conflicts(
                payload,
                day=day_name,
                start_time=start_minute,
                end_time=end_minute,
                section=slot.section,
                ignore_slot_id=slot.id,
            ):
                continue
            if _slot_conflicts(
                payload,
                day=day_name,
                start_time=start_minute,
                end_time=end_minute,
                faculty_id=request.faculty_id,
                ignore_slot_id=slot.id,
            ):
                continue
            if leave_faculty and not _faculty_available_for_slot(
                leave_faculty,
                day=day_name,
                start_time=start_minute,
                end_time=end_minute,
            ):
                continue

            ranked_rooms = sorted(
                room_candidates,
                key=lambda room: (
                    room.id != slot.roomId,
                    abs((room.capacity or 0) - required_capacity),
                    room.name,
                ),
            )
            selected_room = None
            for room in ranked_rooms:
                if _slot_conflicts(
                    payload,
                    day=day_name,
                    start_time=start_minute,
                    end_time=end_minute,
                    room_id=room.id,
                    ignore_slot_id=slot.id,
                ):
                    continue
                selected_room = room
                break
            if selected_room is None:
                continue

            slot.day = day_name
            slot.startTime = _minutes_to_hhmm(start_minute)
            slot.endTime = _minutes_to_hhmm(end_minute)
            slot.roomId = selected_room.id
            slot.facultyId = request.faculty_id
            _prune_primary_from_slot_assistants(slot, request.faculty_id)
            return {
                "slot_id": slot.id,
                "section": slot.section,
                "batch": slot.batch,
                "course_code": getattr(course, "code", slot.courseId),
                "course_name": getattr(course, "name", ""),
                "old_day": old_day,
                "old_start": old_start,
                "old_end": old_end,
                "old_room_id": old_room_id,
                "new_day": slot.day,
                "new_start": slot.startTime,
                "new_end": slot.endTime,
                "new_room_id": slot.roomId,
            }
    return None


def _sync_primary_assignment_from_accepted_offers(
    *,
    db: Session,
    leave_request_id: str,
    assigned_by_id: str,
    payload: OfficialTimetablePayload,
) -> None:
    accepted = list(
        db.execute(
            select(LeaveSubstituteOffer).where(
                LeaveSubstituteOffer.leave_request_id == leave_request_id,
                LeaveSubstituteOffer.status == LeaveSubstituteOfferStatus.accepted,
            )
        ).scalars()
    )
    if not accepted:
        return

    primary_substitute_id = Counter(item.substitute_faculty_id for item in accepted).most_common(1)[0][0]
    slot_by_id = {slot.id: slot for slot in payload.timetable_data}
    course_by_id = {course.id: course for course in payload.course_data}

    fragments: list[str] = []
    for offer in accepted:
        slot = slot_by_id.get(offer.slot_id)
        if slot is None:
            continue
        course = course_by_id.get(slot.courseId)
        fragments.append(
            f"{getattr(course, 'code', slot.courseId)} {slot.day} {slot.startTime}-{slot.endTime} (Section {slot.section})"
        )
    notes = "Accepted substitute offers: " + "; ".join(fragments) if fragments else "Accepted substitute offers"

    assignment = db.execute(
        select(LeaveSubstituteAssignment).where(LeaveSubstituteAssignment.leave_request_id == leave_request_id)
    ).scalar_one_or_none()
    if assignment is None:
        db.add(
            LeaveSubstituteAssignment(
                leave_request_id=leave_request_id,
                substitute_faculty_id=primary_substitute_id,
                assigned_by_id=assigned_by_id,
                notes=notes,
            )
        )
        return

    assignment.substitute_faculty_id = primary_substitute_id
    assignment.assigned_by_id = assigned_by_id
    assignment.notes = notes


def _dispatch_substitute_offers_for_approved_leave(
    *,
    db: Session,
    request: LeaveRequest,
    current_user: User,
) -> dict:
    summary = {
        "offers_created": 0,
        "slots_with_offers": 0,
        "rescheduled_count": 0,
        "unresolved_count": 0,
        "notified_faculty": 0,
        "version_label": None,
    }
    if not request.faculty_id:
        return summary

    official, payload = _load_official_payload(db)
    if official is None or payload is None:
        return summary

    day_name = request.leave_date.strftime("%A")
    affected_slots = [
        item
        for item in payload.timetable_data
        if item.day == day_name and item.facultyId == request.faculty_id
    ]
    if not affected_slots:
        return summary

    now = datetime.now(timezone.utc)
    for stale in db.execute(
        select(LeaveSubstituteOffer).where(
            LeaveSubstituteOffer.leave_request_id == request.id,
            LeaveSubstituteOffer.status == LeaveSubstituteOfferStatus.pending,
        )
    ).scalars():
        stale.status = LeaveSubstituteOfferStatus.cancelled
        stale.responded_at = now
        stale.response_note = "Replaced by updated substitute request run."

    course_by_id = {item.id: item for item in payload.course_data}
    faculty_by_id = {item.id: item for item in db.execute(select(Faculty)).scalars()}
    leave_faculty = faculty_by_id.get(request.faculty_id)
    approved_leave_faculty_ids = set(
        db.execute(
            select(LeaveRequest.faculty_id).where(
                LeaveRequest.leave_date == request.leave_date,
                LeaveRequest.status == LeaveStatus.approved,
                LeaveRequest.id != request.id,
                LeaveRequest.faculty_id.is_not(None),
            )
        ).scalars()
    )
    faculty_users = list(db.execute(select(User).where(User.role == UserRole.faculty)).scalars())
    faculty_user_by_email = {_safe_email(item.email): item for item in faculty_users}

    offers_by_user_id: dict[str, list[dict]] = defaultdict(list)
    rescheduled_events: list[dict] = []
    minutes_cache: dict[str, int] = {}
    for slot in sorted(affected_slots, key=lambda item: (parse_time_to_minutes(item.startTime), item.id)):
        course = course_by_id.get(slot.courseId)
        if course is None:
            continue
        course_code = course.code.strip().upper()
        candidates = _eligible_substitute_candidates_for_slot(
            request=request,
            slot=slot,
            course_code=course_code,
            leave_faculty=leave_faculty,
            payload=payload,
            faculty_by_id=faculty_by_id,
            approved_leave_faculty_ids=approved_leave_faculty_ids,
            minutes_cache=minutes_cache,
        )
        if not candidates:
            event = _reschedule_slot_for_leave(
                db=db,
                request=request,
                payload=payload,
                slot=slot,
                course_by_id=course_by_id,
                faculty_by_id=faculty_by_id,
            )
            if event is None:
                summary["unresolved_count"] += 1
            else:
                summary["rescheduled_count"] += 1
                rescheduled_events.append(event)
            continue

        summary["slots_with_offers"] += 1
        for candidate in candidates:
            db.add(
                LeaveSubstituteOffer(
                    leave_request_id=request.id,
                    slot_id=slot.id,
                    substitute_faculty_id=candidate.id,
                    offered_by_id=current_user.id,
                    status=LeaveSubstituteOfferStatus.pending,
                    expires_at=now + timedelta(minutes=OFFER_EXPIRY_MINUTES),
                )
            )
            summary["offers_created"] += 1
            candidate_user = faculty_user_by_email.get(_safe_email(candidate.email))
            if candidate_user is not None:
                offers_by_user_id[candidate_user.id].append(
                    {
                        "slot_id": slot.id,
                        "course_code": course.code,
                        "section": slot.section,
                        "start_time": slot.startTime,
                        "end_time": slot.endTime,
                    }
                )

    for user_id, slot_items in offers_by_user_id.items():
        summary["notified_faculty"] += 1
        notify_users(
            db,
            user_ids=[user_id],
            title="Substitute Request Pending",
            message=(
                f"You received {len(slot_items)} substitute request(s) for "
                f"{request.leave_date.isoformat()}. Please accept or reject from Leave Management."
            ),
            notification_type=NotificationType.workflow,
            deliver_email=True,
        )

    if rescheduled_events:
        summary["version_label"] = _persist_payload_as_version(
            db=db,
            official=official,
            payload=payload,
            current_user=current_user,
            summary={
                "source": "leave-substitute-auto-reschedule",
                "leave_request_id": request.id,
                "leave_date": request.leave_date.isoformat(),
                "rescheduled_slots": len(rescheduled_events),
            },
        )

        affected_sections = {item["section"] for item in rescheduled_events}
        student_ids = _collect_student_ids_for_sections(db, affected_sections)
        if student_ids:
            notify_users(
                db,
                user_ids=student_ids,
                title="Class Rescheduled",
                message=(
                    f"{len(rescheduled_events)} class(es) were rescheduled due to approved faculty leave on "
                    f"{request.leave_date.isoformat()}."
                ),
                notification_type=NotificationType.timetable,
                deliver_email=True,
            )

    if summary["unresolved_count"] > 0:
        admin_ids = [
            item.id
            for item in db.execute(select(User).where(User.role.in_([UserRole.admin, UserRole.scheduler]))).scalars()
        ]
        notify_users(
            db,
            user_ids=admin_ids,
            title="Substitute Resolution Needed",
            message=(
                f"{summary['unresolved_count']} leave-impacted class(es) on {request.leave_date.isoformat()} "
                "still require manual intervention."
            ),
            notification_type=NotificationType.workflow,
            deliver_email=True,
        )

    log_activity(
        db,
        user=current_user,
        action="leave.substitute.dispatch",
        entity_type="leave_request",
        entity_id=request.id,
        details={
            "leave_date": request.leave_date.isoformat(),
            "offers_created": summary["offers_created"],
            "slots_with_offers": summary["slots_with_offers"],
            "rescheduled_count": summary["rescheduled_count"],
            "unresolved_count": summary["unresolved_count"],
            "version_label": summary["version_label"],
        },
    )
    return summary


def _try_reschedule_uncovered_slot(
    *,
    db: Session,
    request: LeaveRequest,
    slot_id: str,
    current_user: User,
    reason: str,
) -> dict:
    result = {"rescheduled": False, "version_label": None, "event": None}
    if request.status != LeaveStatus.approved or not request.faculty_id:
        return result

    # Test and local DB sessions may run with autoflush disabled; ensure current offer updates are visible.
    db.flush()

    accepted_exists = (
        db.execute(
            select(func.count(LeaveSubstituteOffer.id)).where(
                LeaveSubstituteOffer.leave_request_id == request.id,
                LeaveSubstituteOffer.slot_id == slot_id,
                LeaveSubstituteOffer.status == LeaveSubstituteOfferStatus.accepted,
            )
        ).scalar_one()
        > 0
    )
    if accepted_exists:
        return result

    pending_exists = (
        db.execute(
            select(func.count(LeaveSubstituteOffer.id)).where(
                LeaveSubstituteOffer.leave_request_id == request.id,
                LeaveSubstituteOffer.slot_id == slot_id,
                LeaveSubstituteOffer.status == LeaveSubstituteOfferStatus.pending,
            )
        ).scalar_one()
        > 0
    )
    if pending_exists:
        return result

    official, payload = _load_official_payload(db)
    if official is None or payload is None:
        return result

    slot = _find_slot(payload, slot_id)
    if slot is None or slot.facultyId != request.faculty_id:
        return result

    course_by_id = {item.id: item for item in payload.course_data}
    faculty_by_id = {item.id: item for item in db.execute(select(Faculty)).scalars()}
    event = _reschedule_slot_for_leave(
        db=db,
        request=request,
        payload=payload,
        slot=slot,
        course_by_id=course_by_id,
        faculty_by_id=faculty_by_id,
    )
    if event is None:
        return result

    now = datetime.now(timezone.utc)
    for offer in db.execute(
        select(LeaveSubstituteOffer).where(
            LeaveSubstituteOffer.leave_request_id == request.id,
            LeaveSubstituteOffer.slot_id == slot_id,
            LeaveSubstituteOffer.status.in_(
                [
                    LeaveSubstituteOfferStatus.pending,
                    LeaveSubstituteOfferStatus.expired,
                ]
            ),
        )
    ).scalars():
        offer.status = LeaveSubstituteOfferStatus.rescheduled
        offer.responded_at = now
        if not offer.response_note:
            offer.response_note = "Class was rescheduled because no substitute acceptance was received."

    result["version_label"] = _persist_payload_as_version(
        db=db,
        official=official,
        payload=payload,
        current_user=current_user,
        summary={
            "source": "leave-substitute-fallback-reschedule",
            "leave_request_id": request.id,
            "leave_date": request.leave_date.isoformat(),
            "slot_id": slot_id,
            "reason": reason,
        },
    )
    result["rescheduled"] = True
    result["event"] = event

    absent_user = db.get(User, request.user_id)
    if absent_user is not None:
        notify_users(
            db,
            user_ids=[absent_user.id],
            title="Class Rescheduled After Leave",
            message=(
                f"A class slot impacted by your leave on {request.leave_date.isoformat()} was moved to "
                f"{event['new_day']} {event['new_start']}-{event['new_end']}."
            ),
            notification_type=NotificationType.workflow,
            deliver_email=True,
        )

    student_ids = _collect_student_ids_for_sections(db, {event["section"]})
    if student_ids:
        notify_users(
            db,
            user_ids=student_ids,
            title="Class Rescheduled",
            message=(
                f"{event['course_code']} for section {event['section']} is rescheduled to "
                f"{event['new_day']} {event['new_start']}-{event['new_end']}."
            ),
            notification_type=NotificationType.timetable,
            deliver_email=True,
        )

    log_activity(
        db,
        user=current_user,
        action="leave.substitute.reschedule",
        entity_type="leave_request",
        entity_id=request.id,
        details={
            "slot_id": slot_id,
            "reason": reason,
            "version_label": result["version_label"],
            "event": event,
        },
    )
    return result


def _expire_stale_substitute_offers_and_reschedule(
    *,
    db: Session,
    current_user: User,
) -> dict:
    now = _utc_now()
    pending_with_expiry = list(
        db.execute(
            select(LeaveSubstituteOffer).where(
                LeaveSubstituteOffer.status == LeaveSubstituteOfferStatus.pending,
                LeaveSubstituteOffer.expires_at.is_not(None),
            )
        ).scalars()
    )
    stale_offers = [
        offer
        for offer in pending_with_expiry
        if (_as_utc(offer.expires_at) is not None and _as_utc(offer.expires_at) <= now)
    ]
    if not stale_offers:
        return {"expired_count": 0, "rescheduled_count": 0}

    slot_groups: set[tuple[str, str]] = set()
    for offer in stale_offers:
        offer.status = LeaveSubstituteOfferStatus.expired
        offer.responded_at = now
        offer.response_note = "Offer expired without response."
        slot_groups.add((offer.leave_request_id, offer.slot_id))

    rescheduled_count = 0
    for leave_request_id, slot_id in sorted(slot_groups):
        request = db.get(LeaveRequest, leave_request_id)
        if request is None:
            continue
        fallback = _try_reschedule_uncovered_slot(
            db=db,
            request=request,
            slot_id=slot_id,
            current_user=current_user,
            reason="offer_expired",
        )
        if fallback["rescheduled"]:
            rescheduled_count += 1

    log_activity(
        db,
        user=current_user,
        action="leave.substitute.expire_offers",
        entity_type="leave_request",
        details={
            "expired_count": len(stale_offers),
            "rescheduled_count": rescheduled_count,
        },
    )
    return {
        "expired_count": len(stale_offers),
        "rescheduled_count": rescheduled_count,
    }


@router.get("/leaves", response_model=list[LeaveRequestOut])
def list_leave_requests(
    leave_status: LeaveStatus | None = Query(default=None, alias="status"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[LeaveRequestOut]:
    if current_user.role in {UserRole.admin, UserRole.scheduler}:
        stale_summary = _expire_stale_substitute_offers_and_reschedule(db=db, current_user=current_user)
        if stale_summary["expired_count"] > 0:
            db.commit()

    query = select(LeaveRequest)
    if current_user.role not in {UserRole.admin, UserRole.scheduler}:
        query = query.where(LeaveRequest.user_id == current_user.id)
    if leave_status is not None:
        query = query.where(LeaveRequest.status == leave_status)
    query = query.order_by(LeaveRequest.leave_date.desc(), LeaveRequest.created_at.desc())
    requests = list(db.execute(query).scalars())
    return _hydrate_leave_requests(db, requests)


@router.post(
    "/leaves",
    response_model=LeaveRequestOut,
    status_code=status.HTTP_201_CREATED,
)
def create_leave_request(
    payload: LeaveRequestCreate,
    current_user: User = Depends(require_roles(UserRole.faculty, UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> LeaveRequestOut:
    if payload.leave_date < datetime.now(timezone.utc).date():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Leave date cannot be in the past")

    faculty = _get_faculty_for_user(db, current_user)
    request = LeaveRequest(
        user_id=current_user.id,
        faculty_id=faculty.id if faculty is not None else None,
        leave_date=payload.leave_date,
        leave_type=payload.leave_type,
        reason=payload.reason.strip(),
        status=LeaveStatus.pending,
    )
    db.add(request)
    db.commit()
    db.refresh(request)
    return _hydrate_leave_requests(db, [request])[0]


@router.put("/leaves/{leave_id}/status", response_model=LeaveRequestOut)
def update_leave_status(
    leave_id: str,
    payload: LeaveRequestStatusUpdate,
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> LeaveRequestOut:
    request = db.get(LeaveRequest, leave_id)
    if request is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Leave request not found")

    request.status = payload.status
    request.admin_comment = _normalize_text(payload.admin_comment)
    request.reviewed_by_id = current_user.id
    request.reviewed_at = datetime.now(timezone.utc)

    summary = {
        "offers_created": 0,
        "slots_with_offers": 0,
        "rescheduled_count": 0,
        "unresolved_count": 0,
        "notified_faculty": 0,
        "version_label": None,
    }
    if payload.status == LeaveStatus.approved:
        summary = _dispatch_substitute_offers_for_approved_leave(
            db=db,
            request=request,
            current_user=current_user,
        )

    message = f"Your leave request for {request.leave_date.isoformat()} is now {payload.status.value}."
    if payload.status == LeaveStatus.approved:
        if summary["offers_created"] > 0:
            message += (
                f" Substitute requests were sent to {summary['notified_faculty']} faculty user(s) "
                f"for {summary['slots_with_offers']} class slot(s)."
            )
        if summary["rescheduled_count"] > 0:
            message += f" {summary['rescheduled_count']} slot(s) were auto-rescheduled."
        if summary["unresolved_count"] > 0:
            message += f" {summary['unresolved_count']} slot(s) still need manual action."

    notify_users(
        db,
        user_ids=[request.user_id],
        title="Leave Request Status Updated",
        message=message,
        notification_type=NotificationType.workflow,
        deliver_email=True,
    )
    log_activity(
        db,
        user=current_user,
        action="leave.status.update",
        entity_type="leave_request",
        entity_id=request.id,
        details={
            "status": payload.status.value,
            "reviewed_user_id": request.user_id,
            "offers_created": summary["offers_created"],
            "slots_with_offers": summary["slots_with_offers"],
            "rescheduled_count": summary["rescheduled_count"],
            "unresolved_count": summary["unresolved_count"],
            "version_label": summary["version_label"],
        },
    )

    db.commit()
    db.refresh(request)
    return _hydrate_leave_requests(db, [request])[0]


@router.post("/leaves/substitute-offers/finalize-expired")
def finalize_expired_substitute_offers(
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> dict:
    summary = _expire_stale_substitute_offers_and_reschedule(db=db, current_user=current_user)
    db.commit()
    return summary


@router.post(
    "/leaves/{leave_id}/swap-offers",
    response_model=LeaveSubstituteOfferOut,
    status_code=status.HTTP_201_CREATED,
)
def create_leave_swap_offer(
    leave_id: str,
    payload: LeaveSwapOfferCreate,
    current_user: User = Depends(require_roles(UserRole.faculty, UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> LeaveSubstituteOfferOut:
    request = db.get(LeaveRequest, leave_id)
    if request is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Leave request not found")
    if not request.faculty_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Leave request is not linked to a faculty profile")

    if current_user.role == UserRole.faculty:
        faculty = _get_faculty_for_user(db, current_user)
        if faculty is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Faculty profile not linked")
        if request.user_id != current_user.id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You can only create swap requests for your own leave")

    substitute = db.get(Faculty, payload.substitute_faculty_id)
    if substitute is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Substitute faculty not found")
    if substitute.id == request.faculty_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Substitute faculty must be different from leave faculty")

    official, timetable_payload = _load_official_payload(db)
    if official is None or timetable_payload is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Official timetable not available")

    slot = _find_slot(timetable_payload, payload.slot_id)
    if slot is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Timetable slot not found")
    if slot.facultyId != request.faculty_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Selected slot does not belong to leave faculty")

    leave_day = request.leave_date.strftime("%A")
    if slot.day != leave_day:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Selected slot is not on leave day ({leave_day})",
        )

    same_section_same_day = any(
        other.id != slot.id
        and other.day == slot.day
        and other.section == slot.section
        and other.facultyId == substitute.id
        and other.courseId != slot.courseId
        for other in timetable_payload.timetable_data
    )
    if not same_section_same_day:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Selected faculty must already teach the same section on the same day for class swap workflow",
        )

    course_by_id = {item.id: item for item in timetable_payload.course_data}
    course = course_by_id.get(slot.courseId)
    if course is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Course data for selected slot is missing")

    faculty_by_id = {
        item.id: item
        for item in db.execute(select(Faculty)).scalars()
    }
    leave_faculty = faculty_by_id.get(request.faculty_id)
    approved_leave_faculty_ids = set(
        db.execute(
            select(LeaveRequest.faculty_id).where(
                LeaveRequest.leave_date == request.leave_date,
                LeaveRequest.status == LeaveStatus.approved,
                LeaveRequest.id != request.id,
                LeaveRequest.faculty_id.is_not(None),
            )
        ).scalars()
    )
    eligible = _eligible_substitute_candidates_for_slot(
        request=request,
        slot=slot,
        course_code=course.code.strip().upper(),
        leave_faculty=leave_faculty,
        payload=timetable_payload,
        faculty_by_id=faculty_by_id,
        approved_leave_faculty_ids=approved_leave_faculty_ids,
        minutes_cache={},
    )
    if all(item.id != substitute.id for item in eligible):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Selected faculty is not currently eligible for this slot (availability/workload overlap).",
        )

    now = _utc_now()
    if request.status == LeaveStatus.pending:
        request.status = LeaveStatus.approved
        request.reviewed_by_id = current_user.id
        request.reviewed_at = now
        if not request.admin_comment:
            request.admin_comment = "Auto-approved via faculty class swap workflow."

    for stale in db.execute(
        select(LeaveSubstituteOffer).where(
            LeaveSubstituteOffer.leave_request_id == request.id,
            LeaveSubstituteOffer.slot_id == slot.id,
            LeaveSubstituteOffer.status == LeaveSubstituteOfferStatus.pending,
        )
    ).scalars():
        stale.status = LeaveSubstituteOfferStatus.cancelled
        stale.responded_at = now
        stale.response_note = "Replaced by a newer class swap request."

    existing_offer = db.execute(
        select(LeaveSubstituteOffer).where(
            LeaveSubstituteOffer.leave_request_id == request.id,
            LeaveSubstituteOffer.slot_id == slot.id,
            LeaveSubstituteOffer.substitute_faculty_id == substitute.id,
        )
    ).scalar_one_or_none()

    if existing_offer is None:
        existing_offer = LeaveSubstituteOffer(
            leave_request_id=request.id,
            slot_id=slot.id,
            substitute_faculty_id=substitute.id,
            offered_by_id=current_user.id,
            status=LeaveSubstituteOfferStatus.pending,
            expires_at=now + timedelta(minutes=OFFER_EXPIRY_MINUTES),
        )
        db.add(existing_offer)
    else:
        existing_offer.offered_by_id = current_user.id
        existing_offer.status = LeaveSubstituteOfferStatus.pending
        existing_offer.expires_at = now + timedelta(minutes=OFFER_EXPIRY_MINUTES)
        existing_offer.responded_at = None
        existing_offer.response_note = None

    substitute_user = db.execute(
        select(User).where(
            User.role == UserRole.faculty,
            func.lower(User.email) == substitute.email.lower(),
        )
    ).scalar_one_or_none()
    if substitute_user is not None:
        notify_users(
            db,
            user_ids=[substitute_user.id],
            title="Class Swap Request",
            message=(
                f"You received a class swap request for {course.code} on {slot.day} "
                f"{slot.startTime}-{slot.endTime} (Section {slot.section})."
            ),
            notification_type=NotificationType.workflow,
            deliver_email=True,
        )

    notify_users(
        db,
        user_ids=[request.user_id],
        title="Swap Request Sent",
        message=(
            f"Swap request sent to {substitute.name} for {course.code} on {slot.day} "
            f"{slot.startTime}-{slot.endTime} (Section {slot.section})."
        ),
        notification_type=NotificationType.workflow,
        deliver_email=True,
    )

    log_activity(
        db,
        user=current_user,
        action="leave.swap.offer.create",
        entity_type="leave_request",
        entity_id=request.id,
        details={
            "slot_id": slot.id,
            "substitute_faculty_id": substitute.id,
            "substitute_faculty_email": substitute.email,
            "leave_date": request.leave_date.isoformat(),
            "section": slot.section,
            "auto_approved": request.status == LeaveStatus.approved,
        },
    )

    db.commit()
    refreshed = db.get(LeaveSubstituteOffer, existing_offer.id)
    return _hydrate_substitute_offers(db, [refreshed])[0]


@router.get("/leaves/substitute-offers", response_model=list[LeaveSubstituteOfferOut])
def list_substitute_offers(
    offer_status: LeaveSubstituteOfferStatus | None = Query(default=None, alias="status"),
    leave_id: str | None = Query(default=None),
    scope: str = Query(default="received", pattern="^(received|sent|all)$"),
    current_user: User = Depends(require_roles(UserRole.faculty, UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> list[LeaveSubstituteOfferOut]:
    query = select(LeaveSubstituteOffer)

    if current_user.role == UserRole.faculty:
        faculty = _get_faculty_for_user(db, current_user)
        if faculty is None:
            return []
        if scope == "sent":
            query = query.where(LeaveSubstituteOffer.offered_by_id == current_user.id)
        elif scope == "all":
            query = query.where(
                or_(
                    LeaveSubstituteOffer.substitute_faculty_id == faculty.id,
                    LeaveSubstituteOffer.offered_by_id == current_user.id,
                )
            )
        else:
            query = query.where(LeaveSubstituteOffer.substitute_faculty_id == faculty.id)
    elif leave_id:
        query = query.where(LeaveSubstituteOffer.leave_request_id == leave_id)

    if offer_status is not None:
        query = query.where(LeaveSubstituteOffer.status == offer_status)

    offers = list(
        db.execute(
            query.order_by(LeaveSubstituteOffer.created_at.desc())
        ).scalars()
    )
    return _hydrate_substitute_offers(db, offers)


@router.post(
    "/leaves/substitute-offers/{offer_id}/respond",
    response_model=LeaveSubstituteOfferOut,
)
def respond_to_substitute_offer(
    offer_id: str,
    payload: LeaveSubstituteOfferRespond,
    current_user: User = Depends(require_roles(UserRole.faculty)),
    db: Session = Depends(get_db),
) -> LeaveSubstituteOfferOut:
    faculty = _get_faculty_for_user(db, current_user)
    if faculty is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Faculty profile not linked")

    offer = db.get(LeaveSubstituteOffer, offer_id)
    if offer is None or offer.substitute_faculty_id != faculty.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Substitute offer not found")
    if offer.status != LeaveSubstituteOfferStatus.pending:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Offer is already resolved",
        )

    request = db.get(LeaveRequest, offer.leave_request_id)
    if request is None or request.status != LeaveStatus.approved:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Leave request is not active for substitution")

    now = _utc_now()
    offer_expiry = _as_utc(offer.expires_at)
    if offer_expiry and now > offer_expiry:
        offer.status = LeaveSubstituteOfferStatus.expired
        offer.responded_at = now
        offer.response_note = "Offer expired before response."
        fallback = _try_reschedule_uncovered_slot(
            db=db,
            request=request,
            slot_id=offer.slot_id,
            current_user=current_user,
            reason="offer_expired_before_response",
        )
        db.commit()
        if fallback["rescheduled"]:
            refreshed = db.get(LeaveSubstituteOffer, offer_id)
            return _hydrate_substitute_offers(db, [refreshed])[0]
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Offer expired")

    if payload.decision == "reject":
        offer.status = LeaveSubstituteOfferStatus.rejected
        offer.responded_at = now
        offer.response_note = _normalize_text(payload.response_note)

        fallback = _try_reschedule_uncovered_slot(
            db=db,
            request=request,
            slot_id=offer.slot_id,
            current_user=current_user,
            reason="all_rejected",
        )
        db.commit()
        refreshed = db.get(LeaveSubstituteOffer, offer_id)
        return _hydrate_substitute_offers(db, [refreshed])[0]

    official, timetable_payload = _load_official_payload(db)
    if official is None or timetable_payload is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Official timetable not available")

    slot = _find_slot(timetable_payload, offer.slot_id)
    if slot is None:
        offer.status = LeaveSubstituteOfferStatus.cancelled
        offer.responded_at = now
        offer.response_note = "Referenced timetable slot was removed."
        db.commit()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Referenced timetable slot no longer exists")

    if slot.facultyId != request.faculty_id:
        offer.status = LeaveSubstituteOfferStatus.superseded
        offer.responded_at = now
        offer.response_note = "Another substitute or reschedule has already handled this slot."
        db.commit()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Slot is already covered")

    course_by_id = {item.id: item for item in timetable_payload.course_data}
    course = course_by_id.get(slot.courseId)
    if course is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Offer course is missing from timetable")

    approved_leave_faculty_ids = set(
        db.execute(
            select(LeaveRequest.faculty_id).where(
                LeaveRequest.leave_date == request.leave_date,
                LeaveRequest.status == LeaveStatus.approved,
                LeaveRequest.id != request.id,
                LeaveRequest.faculty_id.is_not(None),
            )
        ).scalars()
    )
    faculty_by_id = {item.id: item for item in db.execute(select(Faculty)).scalars()}
    leave_faculty = faculty_by_id.get(request.faculty_id) if request.faculty_id else None
    candidates = _eligible_substitute_candidates_for_slot(
        request=request,
        slot=slot,
        course_code=course.code.strip().upper(),
        leave_faculty=leave_faculty,
        payload=timetable_payload,
        faculty_by_id=faculty_by_id,
        approved_leave_faculty_ids=approved_leave_faculty_ids,
        minutes_cache={},
    )
    if all(item.id != faculty.id for item in candidates):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You are no longer eligible for this substitute slot",
        )

    slot.facultyId = faculty.id
    _prune_primary_from_slot_assistants(slot, faculty.id)
    _ensure_faculty_in_payload(timetable_payload, faculty)

    offer.status = LeaveSubstituteOfferStatus.accepted
    offer.responded_at = now
    offer.response_note = _normalize_text(payload.response_note)

    for competing in db.execute(
        select(LeaveSubstituteOffer).where(
            LeaveSubstituteOffer.leave_request_id == offer.leave_request_id,
            LeaveSubstituteOffer.slot_id == offer.slot_id,
            LeaveSubstituteOffer.id != offer.id,
            LeaveSubstituteOffer.status == LeaveSubstituteOfferStatus.pending,
        )
    ).scalars():
        competing.status = LeaveSubstituteOfferStatus.superseded
        competing.responded_at = now
        competing.response_note = "Another faculty member accepted first."

    version_label = _persist_payload_as_version(
        db=db,
        official=official,
        payload=timetable_payload,
        current_user=current_user,
        summary={
            "source": "leave-substitute-offer-accepted",
            "leave_request_id": request.id,
            "slot_id": offer.slot_id,
            "substitute_faculty_id": faculty.id,
        },
    )

    _sync_primary_assignment_from_accepted_offers(
        db=db,
        leave_request_id=request.id,
        assigned_by_id=current_user.id,
        payload=timetable_payload,
    )

    absent_user = db.get(User, request.user_id)
    if absent_user is not None:
        notify_users(
            db,
            user_ids=[absent_user.id],
            title="Substitute Confirmed",
            message=(
                f"{faculty.name} accepted substitute coverage for {course.code} on "
                f"{request.leave_date.isoformat()} ({slot.day} {slot.startTime}-{slot.endTime})."
            ),
            notification_type=NotificationType.workflow,
            deliver_email=True,
        )

    student_ids = _collect_student_ids_for_sections(db, {slot.section})
    if student_ids:
        notify_users(
            db,
            user_ids=student_ids,
            title="Class Schedule Updated",
            message=(
                f"{course.code} for section {slot.section} on {slot.day} {slot.startTime}-{slot.endTime} "
                f"is now assigned to substitute faculty {faculty.name}."
            ),
            notification_type=NotificationType.timetable,
            deliver_email=True,
        )

    notify_users(
        db,
        user_ids=[current_user.id],
        title="Substitute Acceptance Recorded",
        message=(
            f"You accepted substitute coverage for {course.code} on {slot.day} "
            f"{slot.startTime}-{slot.endTime}."
        ),
        notification_type=NotificationType.workflow,
        deliver_email=True,
    )

    log_activity(
        db,
        user=current_user,
        action="leave.substitute.accept",
        entity_type="leave_request",
        entity_id=request.id,
        details={
            "offer_id": offer.id,
            "slot_id": offer.slot_id,
            "substitute_faculty_id": faculty.id,
            "version_label": version_label,
        },
    )
    db.commit()
    refreshed = db.get(LeaveSubstituteOffer, offer.id)
    return _hydrate_substitute_offers(db, [refreshed])[0]


@router.post(
    "/leaves/{leave_id}/substitute",
    response_model=LeaveSubstituteAssignmentOut,
)
def assign_leave_substitute(
    leave_id: str,
    payload: LeaveSubstituteAssignmentCreate,
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> LeaveSubstituteAssignmentOut:
    request = db.get(LeaveRequest, leave_id)
    if request is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Leave request not found")
    if request.status != LeaveStatus.approved:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Substitute assignment requires an approved leave request",
        )

    substitute = db.get(Faculty, payload.substitute_faculty_id)
    if substitute is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Substitute faculty not found")

    if request.faculty_id and request.faculty_id == substitute.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Substitute faculty must be different from the faculty on leave",
        )

    day_name = request.leave_date.strftime("%A")

    assignment = db.execute(
        select(LeaveSubstituteAssignment).where(LeaveSubstituteAssignment.leave_request_id == leave_id)
    ).scalar_one_or_none()
    if assignment is None:
        assignment = LeaveSubstituteAssignment(
            leave_request_id=leave_id,
            substitute_faculty_id=substitute.id,
            assigned_by_id=current_user.id,
            notes=_normalize_text(payload.notes),
        )
        db.add(assignment)
    else:
        assignment.substitute_faculty_id = substitute.id
        assignment.assigned_by_id = current_user.id
        assignment.notes = _normalize_text(payload.notes)

    now = datetime.now(timezone.utc)
    for offer in db.execute(
        select(LeaveSubstituteOffer).where(
            LeaveSubstituteOffer.leave_request_id == leave_id,
            LeaveSubstituteOffer.status == LeaveSubstituteOfferStatus.pending,
        )
    ).scalars():
        offer.status = LeaveSubstituteOfferStatus.cancelled
        offer.responded_at = now
        offer.response_note = "Cancelled after manual substitute assignment."

    absent_user = db.get(User, request.user_id)
    substitute_user = db.execute(
        select(User).where(
            User.role == UserRole.faculty,
            func.lower(User.email) == substitute.email.lower(),
        )
    ).scalar_one_or_none()

    if absent_user is not None:
        notify_users(
            db,
            user_ids=[absent_user.id],
            title="Substitute Assigned For Your Leave",
            message=(
                f"{substitute.name} was assigned to cover your classes on "
                f"{request.leave_date.isoformat()}."
            ),
            notification_type=NotificationType.workflow,
            deliver_email=True,
        )
    if substitute_user is not None:
        notify_users(
            db,
            user_ids=[substitute_user.id],
            title="Substitute Class Assignment",
            message=(
                f"You were assigned as substitute faculty on {request.leave_date.isoformat()} "
                f"for {day_name} classes."
            ),
            notification_type=NotificationType.workflow,
            deliver_email=True,
        )

    log_activity(
        db,
        user=current_user,
        action="leave.substitute.assign",
        entity_type="leave_request",
        entity_id=leave_id,
        details={
            "substitute_faculty_id": substitute.id,
            "substitute_faculty_email": substitute.email,
        },
    )

    db.commit()
    db.refresh(assignment)
    return _build_assignment_out(assignment, substitute)
