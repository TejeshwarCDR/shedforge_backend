from collections import defaultdict
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime, timezone
from html import escape
import logging
from math import isclose, sqrt
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db, require_roles
from app.models.faculty import Faculty
from app.models.institution_settings import InstitutionSettings
from app.models.notification import NotificationType
from app.models.program_constraint import ProgramConstraint
from app.models.program_structure import (
    ElectiveConflictPolicy,
    ProgramCourse,
    ProgramElectiveGroup,
    ProgramElectiveGroupMember,
    ProgramSection,
    ProgramSharedLectureGroup,
    ProgramSharedLectureGroupMember,
    ProgramTerm,
)
from app.models.room import Room
from app.models.semester_constraint import SemesterConstraint
from app.models.timetable_conflict_decision import ConflictDecision, TimetableConflictDecision
from app.models.timetable_change_request import (
    TimetableChangeRequest,
    TimetableChangeRequestStatus,
)
from app.models.timetable import OfficialTimetable
from app.models.timetable_generation import TimetableGenerationSettings
from app.models.timetable_version import TimetableVersion
from app.models.user import User, UserRole
from app.schemas.version import TimetableTrendPoint, TimetableVersionCompare, TimetableVersionOut
from app.schemas.insights import (
    ConflictDecisionIn,
    ConflictDecisionOut,
    ConstraintStatus,
    DailyWorkloadEntry,
    OptimizationSummary,
    PerformanceTrendEntry,
    TimetableAnalytics,
    TimetableConflict,
    TimetableConflictResolveAllIn,
    TimetableConflictResolveAllOut,
    TimetableConflictReviewIn,
    TimetableConflictReviewOut,
    WorkloadChartEntry,
)
from app.schemas.settings import (
    DEFAULT_SCHEDULE_POLICY,
    DEFAULT_WORKING_HOURS,
    BreakWindowEntry,
    SchedulePolicyUpdate,
    WorkingHoursEntry,
    parse_time_to_minutes,
)
from app.schemas.timetable import (
    FacultyCourseSectionAssignment,
    FacultyCourseSectionMappingOut,
    OfflinePublishFilters,
    OfflinePublishRequest,
    OfflinePublishResponse,
    OfficialTimetablePayload,
)
from app.schemas.timetable_change_request import (
    TimetableChangeRequestDecisionIn,
    TimetableChangeRequestDecisionOut,
    TimetableChangeRequestOut,
    TimetableChangeRequestProposalIn,
)
from app.services.audit import log_activity
from app.services.email import EmailDeliveryError, send_email
from app.services.notifications import create_notification, notify_all_users, notify_users

router = APIRouter()
logger = logging.getLogger(__name__)

DAY_SHORT_MAP = {
    "Mon": "Monday",
    "Tue": "Tuesday",
    "Wed": "Wednesday",
    "Thu": "Thursday",
    "Fri": "Friday",
    "Sat": "Saturday",
    "Sun": "Sunday",
}

DAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
DAY_INDEX = {day: idx for idx, day in enumerate(DAY_ORDER)}
for short_day, long_day in DAY_SHORT_MAP.items():
    DAY_INDEX[short_day] = DAY_INDEX[long_day]

THREE_SLOT_PRACTICAL_COURSE_CODES = {"23MEE115", "23ECE285"}
THREE_SLOT_PRACTICAL_NAME_MARKERS = {
    "manufacturing practice",
    "digital electronics laboratory",
}

CANONICAL_LUNCH_START_MINUTES = parse_time_to_minutes("13:15")
CANONICAL_LUNCH_END_MINUTES = parse_time_to_minutes("14:05")
REMOVED_LEGACY_SLOT_RANGES: set[tuple[int, int]] = {
    (parse_time_to_minutes("10:45"), parse_time_to_minutes("11:20")),
    (parse_time_to_minutes("11:20"), parse_time_to_minutes("12:10")),
    (parse_time_to_minutes("12:10"), parse_time_to_minutes("13:00")),
    (parse_time_to_minutes("14:40"), parse_time_to_minutes("15:30")),
    (parse_time_to_minutes("15:30"), parse_time_to_minutes("16:20")),
    (parse_time_to_minutes("16:20"), parse_time_to_minutes("16:35")),
}


def normalize_day(value: str) -> str:
    return DAY_SHORT_MAP.get(value, value)


def _minutes_to_time(value: int) -> str:
    hours = value // 60
    minutes = value % 60
    return f"{hours:02d}:{minutes:02d}"


def _is_removed_legacy_slot_range(start: int, end: int) -> bool:
    return (start, end) in REMOVED_LEGACY_SLOT_RANGES


def _is_canonical_lunch_range(start: int, end: int) -> bool:
    return start == CANONICAL_LUNCH_START_MINUTES and end == CANONICAL_LUNCH_END_MINUTES


def _overlaps_canonical_lunch(start: int, end: int) -> bool:
    return start < CANONICAL_LUNCH_END_MINUTES and end > CANONICAL_LUNCH_START_MINUTES


def _day_sort_index(day: str) -> int:
    return DAY_INDEX.get(day, len(DAY_ORDER))


def _normalize_project_phase_text(value: str | None) -> str:
    raw = (value or "").lower().replace("-", " ").replace("_", " ")
    return " ".join(raw.split())


def _is_project_phase_course(course: object | None) -> bool:
    if course is None:
        return False
    name = _normalize_project_phase_text(str(getattr(course, "name", "")))
    code = _normalize_project_phase_text(str(getattr(course, "code", "")))
    return "project phase" in name or "project phase" in code


def _normalize_course_identity_text(value: str | None) -> str:
    return " ".join((value or "").strip().lower().replace("-", " ").replace("_", " ").split())


def _course_practical_hours(course: object | None) -> int:
    if course is None:
        return 0
    return max(
        0,
        int(
            getattr(course, "lab_hours", getattr(course, "labHours", 0))
            or 0
        ),
    )


def _course_batch_segregation_enabled(course: object | None) -> bool:
    if course is None:
        return True
    value = getattr(course, "batch_segregation", getattr(course, "batchSegregation", True))
    return bool(value)


def _course_has_practical_component(course: object | None) -> bool:
    return _course_practical_hours(course) > 0


def _slot_is_practical(slot: object, course: object | None) -> bool:
    session_type = getattr(slot, "sessionType", None)
    if session_type is None:
        session_type = getattr(slot, "session_type", None)
    if isinstance(session_type, str):
        normalized = session_type.strip().lower()
        if normalized in {"theory", "tutorial", "lab"}:
            return normalized == "lab"

    if course is None:
        return False

    # If no explicit session marker is available, only treat the slot as practical
    # when the course is fully practical. Mixed LTP courses default to lecture mode.
    theory = int(getattr(course, "theory_hours", getattr(course, "theoryHours", 0)) or 0)
    tutorial = int(getattr(course, "tutorial_hours", getattr(course, "tutorialHours", 0)) or 0)
    practical = _course_practical_hours(course)
    return practical > 0 and (theory + tutorial) <= 0


def _slot_assistant_faculty_ids(slot: object) -> tuple[str, ...]:
    raw = getattr(slot, "assistant_faculty_ids", None)
    if raw is None:
        raw = getattr(slot, "assistantFacultyIds", None)
    if not isinstance(raw, list):
        return tuple()
    ordered: list[str] = []
    seen: set[str] = set()
    for item in raw:
        faculty_id = str(item or "").strip()
        if not faculty_id or faculty_id in seen:
            continue
        seen.add(faculty_id)
        ordered.append(faculty_id)
    return tuple(ordered)


def _slot_all_faculty_ids(slot: object) -> tuple[str, ...]:
    primary = str(getattr(slot, "facultyId", "") or "").strip()
    ordered: list[str] = []
    seen: set[str] = set()
    if primary:
        ordered.append(primary)
        seen.add(primary)
    for assistant_id in _slot_assistant_faculty_ids(slot):
        if assistant_id in seen:
            continue
        seen.add(assistant_id)
        ordered.append(assistant_id)
    return tuple(ordered)


def _is_virtual_faculty_id(faculty_id: str | None) -> bool:
    return str(faculty_id or "").strip().startswith("nr-f-")


def _is_research_placeholder_course(course: object | None) -> bool:
    if course is None:
        return False
    code = str(getattr(course, "code", "") or "").strip().upper()
    name = str(getattr(course, "name", "") or "").strip().lower()
    course_id = str(getattr(course, "id", "") or "").strip().lower()
    return code.startswith("RS-") or name.startswith("research slot") or course_id.startswith("res-c-")


def _prune_primary_from_slot_assistants(slot: object, primary_faculty_id: str) -> None:
    assistant_ids = [faculty_id for faculty_id in _slot_assistant_faculty_ids(slot) if faculty_id != primary_faculty_id]
    if hasattr(slot, "assistant_faculty_ids"):
        setattr(slot, "assistant_faculty_ids", assistant_ids)
    if hasattr(slot, "assistantFacultyIds"):
        setattr(slot, "assistantFacultyIds", assistant_ids)


def _set_slot_assistant_faculty_ids(slot: object, assistant_ids: list[str]) -> None:
    if hasattr(slot, "assistant_faculty_ids"):
        setattr(slot, "assistant_faculty_ids", assistant_ids)
    if hasattr(slot, "assistantFacultyIds"):
        setattr(slot, "assistantFacultyIds", assistant_ids)


def _is_three_slot_practical_course(course: object | None) -> bool:
    if course is None:
        return False
    code = str(getattr(course, "code", "") or "").strip().upper()
    if code in THREE_SLOT_PRACTICAL_COURSE_CODES:
        return True
    name = _normalize_course_identity_text(str(getattr(course, "name", "")))
    return any(marker in name for marker in THREE_SLOT_PRACTICAL_NAME_MARKERS)


def _slot_semester_number(
    *,
    slot: object,
    course_map: dict[str, object],
    fallback_term_number: int | None,
) -> int | None:
    course = course_map.get(getattr(slot, "courseId", None))
    semester = getattr(course, "semester_number", None) if course is not None else None
    if isinstance(semester, int):
        return semester
    return fallback_term_number


def _slot_semester_section_label(
    *,
    slot: object,
    course_map: dict[str, object],
    fallback_term_number: int | None,
) -> str:
    semester = _slot_semester_number(slot=slot, course_map=course_map, fallback_term_number=fallback_term_number)
    section = str(getattr(slot, "section", "") or "").strip() or "?"
    if semester is None:
        return f"Section {section}"
    return f"Semester {semester} Section {section}"


def _course_computed_credits(course: object) -> float:
    theory = int(getattr(course, "theory_hours", getattr(course, "theoryHours", 0)) or 0)
    tutorial = int(getattr(course, "tutorial_hours", getattr(course, "tutorialHours", 0)) or 0)
    practical = int(getattr(course, "lab_hours", getattr(course, "labHours", 0)) or 0)
    return float(max(0, theory) + max(0, tutorial) + (max(0, practical) / 2.0))


def _course_weekly_period_units(course: object, schedule_policy: SchedulePolicyUpdate) -> int:
    split_units = (
        max(0, int(getattr(course, "theory_hours", getattr(course, "theoryHours", 0)) or 0))
        + max(0, int(getattr(course, "tutorial_hours", getattr(course, "tutorialHours", 0)) or 0))
        + max(0, _course_practical_hours(course))
    )
    if split_units > 0:
        return split_units
    return max(0, int(getattr(course, "hoursPerWeek", 0) or 0))


def _lab_block_slots_for_course(course: object, schedule_policy: SchedulePolicyUpdate) -> int:
    practical_hours = _course_practical_hours(course)
    configured = getattr(course, "practical_contiguous_slots", getattr(course, "practicalContiguousSlots", None))
    if configured is not None:
        try:
            block_size = max(1, int(configured))
        except (TypeError, ValueError):
            block_size = 1
    elif _is_three_slot_practical_course(course):
        block_size = 3
    else:
        block_size = max(1, int(schedule_policy.lab_contiguous_slots or 2))
    if practical_hours > 0:
        block_size = min(block_size, practical_hours)
    return max(1, block_size)


def slots_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    return start_a < end_b and start_b < end_a


def load_working_hours(db: Session) -> dict[str, WorkingHoursEntry]:
    record = db.get(InstitutionSettings, 1)
    if record is None:
        entries = DEFAULT_WORKING_HOURS
    else:
        entries = [WorkingHoursEntry.model_validate(entry) for entry in record.working_hours]
    return {entry.day: entry for entry in entries}


def load_schedule_policy(db: Session) -> SchedulePolicyUpdate:
    record = db.get(InstitutionSettings, 1)
    if record is None:
        return DEFAULT_SCHEDULE_POLICY

    period_minutes = record.period_minutes or DEFAULT_SCHEDULE_POLICY.period_minutes
    lab_contiguous_slots = record.lab_contiguous_slots or DEFAULT_SCHEDULE_POLICY.lab_contiguous_slots
    break_windows = record.break_windows or [item.model_dump() for item in DEFAULT_SCHEDULE_POLICY.breaks]
    return SchedulePolicyUpdate(
        period_minutes=period_minutes,
        lab_contiguous_slots=lab_contiguous_slots,
        breaks=break_windows,
    )


def load_program_constraint(db: Session, program_id: str) -> ProgramConstraint | None:
    return (
        db.execute(select(ProgramConstraint).where(ProgramConstraint.program_id == program_id))
        .scalars()
        .first()
    )


def normalize_program_daily_slots(raw_slots: list[dict] | None) -> list[tuple[int, int, str, str]]:
    normalized_by_key: dict[tuple[int, int], tuple[int, int, str, str]] = {}
    has_canonical_lunch = False
    has_teaching_slots = False
    for item in raw_slots or []:
        try:
            start_time = str(item.get("start_time", "")).strip()
            end_time = str(item.get("end_time", "")).strip()
            start = parse_time_to_minutes(start_time)
            end = parse_time_to_minutes(end_time)
        except Exception:
            continue
        if end <= start:
            continue
        if _is_removed_legacy_slot_range(start, end):
            continue
        if _overlaps_canonical_lunch(start, end) and not _is_canonical_lunch_range(start, end):
            continue
        tag = str(item.get("tag", "teaching")).strip().lower() or "teaching"
        if tag not in {"teaching", "block", "break", "lunch"}:
            tag = "teaching"
        label = str(item.get("label", "")).strip() or tag.title()
        if _is_canonical_lunch_range(start, end):
            tag = "lunch"
            label = "Lunch Break"
            has_canonical_lunch = True
        elif tag == "lunch":
            continue
        elif tag == "teaching":
            has_teaching_slots = True

        normalized_by_key[(start, end)] = (start, end, tag, label)

    if has_teaching_slots and not has_canonical_lunch:
        normalized_by_key[(CANONICAL_LUNCH_START_MINUTES, CANONICAL_LUNCH_END_MINUTES)] = (
            CANONICAL_LUNCH_START_MINUTES,
            CANONICAL_LUNCH_END_MINUTES,
            "lunch",
            "Lunch Break",
        )

    normalized = sorted(normalized_by_key.values(), key=lambda slot: slot[0])
    return normalized


def build_teaching_segments_from_program_slots(
    slots: list[tuple[int, int, str, str]],
) -> tuple[list[tuple[int, int]], list[tuple[int, int, str]]]:
    teaching_segments: list[tuple[int, int]] = []
    blocked_segments: list[tuple[int, int, str]] = []
    for start, end, tag, label in slots:
        if tag == "teaching":
            teaching_segments.append((start, end))
        else:
            blocked_segments.append((start, end, label))
    return teaching_segments, blocked_segments


def slot_overlaps_break(slot_start: int, slot_end: int, breaks: list[BreakWindowEntry]) -> BreakWindowEntry | None:
    for break_entry in breaks:
        break_start = parse_time_to_minutes(break_entry.start_time)
        break_end = parse_time_to_minutes(break_entry.end_time)
        if slot_start < break_end and slot_end > break_start:
            return break_entry
    return None


def build_teaching_segments(
    day_start: int,
    day_end: int,
    period_minutes: int,
    breaks: list[BreakWindowEntry],
) -> list[tuple[int, int]]:
    break_windows = sorted(
        (
            (parse_time_to_minutes(item.start_time), parse_time_to_minutes(item.end_time))
            for item in breaks
            if parse_time_to_minutes(item.end_time) > day_start and parse_time_to_minutes(item.start_time) < day_end
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


def is_slot_aligned_with_segments(slot_start: int, slot_end: int, segments: list[tuple[int, int]]) -> bool:
    if slot_end <= slot_start:
        return False
    by_start = {start: end for start, end in segments}
    cursor = slot_start
    while cursor < slot_end:
        next_boundary = by_start.get(cursor)
        if next_boundary is None:
            return False
        cursor = next_boundary
    return cursor == slot_end


def load_semester_constraint(db: Session, term_number: int) -> SemesterConstraint | None:
    return (
        db.execute(select(SemesterConstraint).where(SemesterConstraint.term_number == term_number))
        .scalars()
        .first()
    )


def enforce_semester_constraints(
    payload: OfficialTimetablePayload,
    constraint: SemesterConstraint,
    force: bool = False,
) -> None:
    allowed_start = parse_time_to_minutes(constraint.earliest_start_time)
    allowed_end = parse_time_to_minutes(constraint.latest_end_time)
    max_day_minutes = constraint.max_hours_per_day * 60
    max_week_minutes = constraint.max_hours_per_week * 60
    max_consecutive_minutes = constraint.max_consecutive_hours * 60

    section_windows: dict[str, set[tuple[str, int, int]]] = defaultdict(set)

    for slot in payload.timetable_data:
        slot_start = parse_time_to_minutes(slot.startTime)
        slot_end = parse_time_to_minutes(slot.endTime)
        if slot_start < allowed_start or slot_end > allowed_end:
            if not force:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"Timeslot {slot.id} on {slot.day} must be within "
                        f"{constraint.earliest_start_time}-{constraint.latest_end_time}"
                    ),
                )
        section_windows[slot.section].add((slot.day, slot_start, slot_end))

    for section_name, windows in section_windows.items():
        total_week_minutes = sum(end - start for _, start, end in windows)
        if total_week_minutes > max_week_minutes:
            if not force:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        "Weekly scheduled hours exceed semester constraints "
                        f"for section {section_name}"
                    ),
                )

        day_slots: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for day, start, end in windows:
            day_slots[day].append((start, end))

        for day, slots in day_slots.items():
            slots.sort(key=lambda item: item[0])
            day_total = sum(end - start for start, end in slots)
            if day_total > max_day_minutes:
                if not force:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Daily scheduled hours exceed semester constraints on {day} for section {section_name}",
                    )

            prev_end = None
            consecutive_start = None
            consecutive_end = None
            for start, end in slots:
                if prev_end is not None:
                    gap = start - prev_end
                    if gap < 0:
                        if not force:
                            raise HTTPException(
                                status_code=status.HTTP_400_BAD_REQUEST,
                                detail=f"Timeslots overlap on {day} for section {section_name}",
                            )
                    if gap < constraint.min_break_minutes:
                        if not force:
                            raise HTTPException(
                                status_code=status.HTTP_400_BAD_REQUEST,
                                detail=(
                                    f"Section {section_name} on {day} must allow at least "
                                    f"{constraint.min_break_minutes} minutes break between classes"
                                ),
                            )
                if consecutive_start is None or prev_end is None or start != prev_end:
                    consecutive_start = start
                    consecutive_end = end
                else:
                    consecutive_end = end

                if consecutive_end - consecutive_start > max_consecutive_minutes:
                    if not force:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Consecutive classes exceed limit on {day} for section {section_name}",
                        )

                prev_end = end


def load_shared_lecture_groups(
    *,
    db: Session,
    program_id: str,
    term_number: int,
) -> list[tuple[str, str, set[str]]]:
    groups = (
        db.execute(
            select(ProgramSharedLectureGroup).where(
                ProgramSharedLectureGroup.program_id == program_id,
                ProgramSharedLectureGroup.term_number == term_number,
            )
        )
        .scalars()
        .all()
    )
    if not groups:
        return []

    group_ids = [group.id for group in groups]
    members_by_group: dict[str, set[str]] = defaultdict(set)
    for member in db.execute(
        select(ProgramSharedLectureGroupMember).where(
            ProgramSharedLectureGroupMember.group_id.in_(group_ids)
        )
    ).scalars():
        members_by_group[member.group_id].add(member.section_name)

    result: list[tuple[str, str, set[str]]] = []
    for group in groups:
        sections = members_by_group.get(group.id, set())
        if len(sections) >= 2:
            result.append((group.name, group.course_id, sections))
    return result


def build_shared_group_lookup(
    groups: list[tuple[str, str, set[str]]],
) -> dict[str, list[set[str]]]:
    lookup: dict[str, list[set[str]]] = defaultdict(list)
    for _, course_id, sections in groups:
        lookup[course_id].append(set(sections))
    return lookup


def sections_share_shared_lecture(
    *,
    course_id: str,
    section_a: str,
    section_b: str,
    shared_groups_by_course: dict[str, list[set[str]]],
) -> bool:
    for sections in shared_groups_by_course.get(course_id, []):
        if section_a in sections and section_b in sections:
            return True
    return False


def is_shared_lecture_overlap_event(
    *,
    slot: object,
    other: object,
    slot_start: int,
    slot_end: int,
    other_start: int,
    other_end: int,
    shared_groups_by_course: dict[str, list[set[str]]],
) -> bool:
    if slot.courseId != other.courseId:
        return False
    if slot.section == other.section:
        return False
    if slot.roomId != other.roomId:
        return False
    if slot.facultyId != other.facultyId:
        return False
    if (slot.batch or "") != (other.batch or ""):
        return False
    if slot_start != other_start or slot_end != other_end:
        return False
    return sections_share_shared_lecture(
        course_id=slot.courseId,
        section_a=slot.section,
        section_b=other.section,
        shared_groups_by_course=shared_groups_by_course,
    )


def enforce_resource_conflicts(
    payload: OfficialTimetablePayload,
    course_by_id: dict[str, object],
    shared_groups_by_course: dict[str, list[set[str]]],
    force: bool = False,
) -> None:
    slots_by_day: dict[str, list] = defaultdict(list)
    for slot in payload.timetable_data:
        slots_by_day[slot.day].append(slot)

    for day, slots in slots_by_day.items():
        for i, slot in enumerate(slots):
            slot_start = parse_time_to_minutes(slot.startTime)
            slot_end = parse_time_to_minutes(slot.endTime)
            for other in slots[i + 1 :]:
                other_start = parse_time_to_minutes(other.startTime)
                other_end = parse_time_to_minutes(other.endTime)
                if slot_start >= other_end or other_start >= slot_end:
                    continue

                allow_shared_lecture = is_shared_lecture_overlap_event(
                    slot=slot,
                    other=other,
                    slot_start=slot_start,
                    slot_end=slot_end,
                    other_start=other_start,
                    other_end=other_end,
                    shared_groups_by_course=shared_groups_by_course,
                )

                if slot.roomId == other.roomId and not allow_shared_lecture:
                    if not force:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Room conflict on {day} for room {slot.roomId}",
                        )
                slot_faculty_ids = {
                    faculty_id
                    for faculty_id in _slot_all_faculty_ids(slot)
                    if not _is_virtual_faculty_id(faculty_id)
                }
                other_faculty_ids = {
                    faculty_id
                    for faculty_id in _slot_all_faculty_ids(other)
                    if not _is_virtual_faculty_id(faculty_id)
                }
                shared_faculty_ids = sorted(slot_faculty_ids.intersection(other_faculty_ids))
                if shared_faculty_ids and not allow_shared_lecture:
                    if not force:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Faculty conflict on {day} for faculty {', '.join(shared_faculty_ids)}",
                        )

                course = course_by_id[slot.courseId]
                other_course = course_by_id[other.courseId]
                slot_semester = getattr(course, "semester_number", None)
                other_semester = getattr(other_course, "semester_number", None)
                slot_semester = slot_semester if slot_semester is not None else payload.term_number
                other_semester = other_semester if other_semester is not None else payload.term_number

                same_section_scope = (
                    slot.section == other.section
                    and (
                        slot_semester is None
                        or other_semester is None
                        or slot_semester == other_semester
                    )
                )

                if same_section_scope:
                    allow_parallel_lab = (
                        _slot_is_practical(slot, course)
                        and _slot_is_practical(other, other_course)
                        and slot.courseId == other.courseId
                        and slot.batch
                        and other.batch
                        and slot.batch != other.batch
                    )
                    if not allow_parallel_lab:
                        if not force:
                            semester_label = (
                                f"Semester {slot_semester}"
                                if slot_semester is not None
                                else "Unknown semester"
                            )
                            raise HTTPException(
                                status_code=status.HTTP_400_BAD_REQUEST,
                                detail=f"Section conflict on {day} for section {slot.section} ({semester_label})",
                            )


def enforce_single_faculty_per_course_sections(
    payload: OfficialTimetablePayload,
    course_by_id: dict[str, object],
    faculty_by_id: dict[str, object],
    force: bool = False,
) -> None:
    faculty_ids_by_course_section: dict[tuple[str, str], set[str]] = defaultdict(set)

    for slot in payload.timetable_data:
        course = course_by_id.get(slot.courseId)
        if _slot_is_practical(slot, course):
            continue
        if course is not None and not bool(getattr(course, "assign_faculty", True)):
            continue
        if _is_virtual_faculty_id(slot.facultyId):
            continue
        faculty_ids_by_course_section[(slot.courseId, slot.section)].add(slot.facultyId)

    violations: list[str] = []
    for (course_id, section_name), faculty_ids in faculty_ids_by_course_section.items():
        if len(faculty_ids) <= 1:
            continue
        course = course_by_id.get(course_id)
        course_label = getattr(course, "code", course_id)
        faculty_labels = ", ".join(
            sorted(getattr(faculty_by_id.get(faculty_id), "name", faculty_id) for faculty_id in faculty_ids)
        )
        violations.append(f"{course_label} [section {section_name}] -> {faculty_labels}")

    if violations:
        if not force:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Each non-lab course must be assigned to one faculty within each section: "
                    + " | ".join(violations)
                ),
            )


def enforce_course_scheduling(
    payload: OfficialTimetablePayload,
    course_by_id: dict[str, object],
    room_by_id: dict[str, object],
    schedule_policy: SchedulePolicyUpdate,
    force: bool = False,
) -> None:
    period_minutes = schedule_policy.period_minutes
    grouped: dict[tuple[str, str, str, str | None], list] = defaultdict(list)
    for slot in payload.timetable_data:
        course = course_by_id[slot.courseId]
        requires_batch_segregation = _course_batch_segregation_enabled(course)
        slot_is_practical = _slot_is_practical(slot, course)
        if slot_is_practical:
            if requires_batch_segregation and not slot.batch:
                if not force:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Lab timeslot {slot.id} must include a batch identifier",
                    )
            if not requires_batch_segregation and slot.batch:
                if not force:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Lab timeslot {slot.id} must not include a batch when batch segregation is disabled",
                    )
            if slot.studentCount is None:
                if not force:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Lab timeslot {slot.id} must include studentCount for batch sizing",
                    )
            room = room_by_id[slot.roomId]
            if getattr(room, "type", None) != "lab":
                if not force:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Lab timeslot {slot.id} must be scheduled in a lab room",
                    )
            slot_duration = parse_time_to_minutes(slot.endTime) - parse_time_to_minutes(slot.startTime)
            if slot_duration != period_minutes:
                if not force:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=(
                            f"Lab timeslot {slot.id} must span exactly one period "
                            f"({period_minutes} minutes)"
                        ),
                    )
        else:
            if slot.batch:
                if not force:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Non-practical timeslot {slot.id} must not include a batch identifier",
                    )
        session_bucket = "practical" if slot_is_practical else "lecture"
        batch_key = slot.batch if (slot_is_practical and requires_batch_segregation) else None
        group_key = (slot.courseId, slot.section, session_bucket, batch_key)
        grouped[group_key].append(slot)

    for (course_id, section, session_bucket, batch), slots in grouped.items():
        course = course_by_id[course_id]
        theory_units = max(0, int(getattr(course, "theory_hours", getattr(course, "theoryHours", 0)) or 0))
        tutorial_units = max(0, int(getattr(course, "tutorial_hours", getattr(course, "tutorialHours", 0)) or 0))
        practical_units = _course_practical_hours(course)
        lecture_units = theory_units + tutorial_units
        required_units = practical_units if session_bucket == "practical" else lecture_units
        required_minutes = required_units * period_minutes
        total_minutes = 0
        slots_sorted = sorted(slots, key=lambda s: (s.day, parse_time_to_minutes(s.startTime)))
        for slot in slots_sorted:
            total_minutes += parse_time_to_minutes(slot.endTime) - parse_time_to_minutes(slot.startTime)

        if required_minutes and total_minutes != required_minutes:
            label = f"{course_id} section {section}"
            if batch:
                label += f" batch {batch}"
            if not force:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Scheduled duration for {label} must equal {required_minutes} minutes per week",
                )

        if session_bucket == "practical" and required_units > 0:
            preferred_block_slots = _lab_block_slots_for_course(course, schedule_policy)
            full_blocks, remainder = divmod(required_units, preferred_block_slots)
            expected_block_lengths: list[int] = [preferred_block_slots * period_minutes] * full_blocks
            if remainder > 0:
                expected_block_lengths.append(remainder * period_minutes)

            blocks: list[int] = []
            current_day: str | None = None
            current_start: int | None = None
            current_end: int | None = None

            for slot in slots_sorted:
                slot_start = parse_time_to_minutes(slot.startTime)
                slot_end = parse_time_to_minutes(slot.endTime)
                if current_day != slot.day or current_end is None or slot_start != current_end:
                    if current_day is not None and current_start is not None and current_end is not None:
                        blocks.append(current_end - current_start)
                    current_day = slot.day
                    current_start = slot_start
                    current_end = slot_end
                else:
                    current_end = slot_end

            if current_day is not None and current_start is not None and current_end is not None:
                blocks.append(current_end - current_start)

            blocks_sorted = sorted(blocks)
            expected_sorted = sorted(expected_block_lengths)
            if expected_sorted and len(blocks_sorted) != len(expected_sorted):
                if not force:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Practical sessions for {course_id} must be scheduled in contiguous blocks",
                    )
            elif expected_sorted and blocks_sorted != expected_sorted:
                if not force:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=(
                            f"Practical sessions for {course_id} must use contiguous blocks of "
                            f"{preferred_block_slots} period(s) by default"
                        ),
                    )


def enforce_room_capacity(
    payload: OfficialTimetablePayload,
    room_by_id: dict[str, object],
    db: Session,
    force: bool = False,
) -> dict[str, int]:
    resolved_counts: dict[str, int] = {}
    for slot in payload.timetable_data:
        student_count = slot.studentCount
        if student_count is None and payload.program_id and payload.term_number is not None:
            section = (
                db.execute(
                    select(ProgramSection).where(
                        ProgramSection.program_id == payload.program_id,
                        ProgramSection.term_number == payload.term_number,
                        ProgramSection.name == slot.section,
                    )
                )
                .scalars()
                .first()
            )
            if section is not None:
                student_count = section.capacity

        if student_count is None:
            if not force:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"studentCount is required to validate room capacity for timeslot {slot.id}",
                )
            student_count = 0

        room = room_by_id[slot.roomId]
        if getattr(room, "capacity", 0) < student_count:
            if not force:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Room {room.name} capacity is insufficient for timeslot {slot.id}",
                )
        resolved_counts[slot.id] = student_count
    return resolved_counts


def enforce_program_credit_requirements(
    payload: OfficialTimetablePayload,
    course_by_id: dict[str, object],
    db: Session,
    force: bool = False,
) -> None:
    if not payload.program_id or payload.term_number is None:
        return

    term = (
        db.execute(
            select(ProgramTerm).where(
                ProgramTerm.program_id == payload.program_id,
                ProgramTerm.term_number == payload.term_number,
            )
        )
        .scalars()
        .first()
    )
    if term is None:
        if not force:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Program term not configured for credit requirements",
            )
        return

    program_courses = (
        db.execute(
            select(ProgramCourse).where(
                ProgramCourse.program_id == payload.program_id,
                ProgramCourse.term_number == payload.term_number,
            )
        )
        .scalars()
        .all()
    )
    if not program_courses:
        return

    program_course_ids = {course.course_id for course in program_courses}
    required_course_ids = {course.course_id for course in program_courses if course.is_required}
    scheduled_course_ids = {
        slot.courseId
        for slot in payload.timetable_data
        if not _is_research_placeholder_course(course_by_id.get(slot.courseId))
    }

    missing_required = required_course_ids - scheduled_course_ids
    if missing_required:
        if not force:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Missing required courses for term: {', '.join(sorted(missing_required))}",
            )

    extra_courses = scheduled_course_ids - program_course_ids
    if extra_courses:
        if not force:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Scheduled courses not part of program term: {', '.join(sorted(extra_courses))}",
            )

    total_credits = 0.0
    for course_id in scheduled_course_ids:
        course = course_by_id.get(course_id)
        if course is not None:
            total_credits += _course_computed_credits(course)

    if term.credits_required > 0 and not isclose(total_credits, float(term.credits_required), abs_tol=0.01):
        if not force:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Scheduled credits ({total_credits:.2f}, computed as L+T+P/2) "
                    f"must exactly match program term requirement ({term.credits_required:.2f})"
                ),
            )


def enforce_section_credit_aligned_minutes(
    payload: OfficialTimetablePayload,
    db: Session,
    schedule_policy: SchedulePolicyUpdate,
    force: bool = False,
) -> None:
    if not payload.program_id or payload.term_number is None:
        return

    term = (
        db.execute(
            select(ProgramTerm).where(
                ProgramTerm.program_id == payload.program_id,
                ProgramTerm.term_number == payload.term_number,
            )
        )
        .scalars()
        .first()
    )
    if term is None or term.credits_required <= 0:
        return

    mapped_course_ids = (
        db.execute(
            select(ProgramCourse.course_id).where(
                ProgramCourse.program_id == payload.program_id,
                ProgramCourse.term_number == payload.term_number,
            )
        )
        .scalars()
        .all()
    )
    payload_hours_by_course = {
        course.id: _course_weekly_period_units(course, schedule_policy)
        for course in payload.course_data
    }
    configured_hours = sum(payload_hours_by_course.get(course_id, 0) for course_id in mapped_course_ids)
    if configured_hours <= 0:
        configured_hours = sum(payload_hours_by_course.values())

    expected_hours = configured_hours
    if term.credits_required > 0 and term.credits_required == configured_hours:
        expected_hours = term.credits_required
    elif expected_hours <= 0 and term.credits_required > 0:
        expected_hours = term.credits_required
    if expected_hours <= 0:
        return

    expected_minutes = expected_hours * schedule_policy.period_minutes
    course_by_id = {course.id: course for course in payload.course_data}
    section_windows: dict[str, set[tuple[str, int, int]]] = defaultdict(set)
    for slot in payload.timetable_data:
        if _is_research_placeholder_course(course_by_id.get(slot.courseId)):
            continue
        start = parse_time_to_minutes(slot.startTime)
        end = parse_time_to_minutes(slot.endTime)
        if end <= start:
            continue
        section_windows[slot.section].add((slot.day, start, end))

    configured_sections = (
        db.execute(
            select(ProgramSection.name).where(
                ProgramSection.program_id == payload.program_id,
                ProgramSection.term_number == payload.term_number,
            )
        )
        .scalars()
        .all()
    )
    section_names = set(configured_sections) if configured_sections else set(section_windows.keys())
    if not section_names:
        return

    for section_name in sorted(section_names):
        minutes = sum(end - start for _, start, end in section_windows.get(section_name, set()))
        if minutes != expected_minutes:
            if not force:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"Section {section_name} has {minutes} scheduled minutes per week, "
                        f"but semester target is {expected_minutes} minutes "
                        f"({expected_hours} hourly credits x {schedule_policy.period_minutes} minutes)."
                    ),
                )


def enforce_shared_lecture_constraints(
    payload: OfficialTimetablePayload,
    shared_groups: list[tuple[str, str, set[str]]],
    shared_groups_by_course: dict[str, list[set[str]]],
    room_by_id: dict[str, object],
    student_counts_by_slot: dict[str, int],
    force: bool = False,
) -> None:
    if not shared_groups:
        return

    slots_by_course_section: dict[tuple[str, str], list[tuple[str, str, str, str, str]]] = defaultdict(list)
    for slot in payload.timetable_data:
        slots_by_course_section[(slot.courseId, slot.section)].append(
            (slot.day, slot.startTime, slot.endTime, slot.roomId, slot.facultyId)
        )

    for group_name, course_id, sections in shared_groups:
        baseline_signatures: list[tuple[str, str, str, str, str]] | None = None
        for section_name in sorted(sections):
            signatures = sorted(slots_by_course_section.get((course_id, section_name), []))
            if baseline_signatures is None:
                baseline_signatures = signatures
                continue
            if signatures != baseline_signatures:
                if not force:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=(
                            f"Shared lecture group '{group_name}' requires synchronized slots for course {course_id} "
                            f"across sections: {', '.join(sorted(sections))}"
                        ),
                    )

    grouped_events: dict[tuple[str, str, str, str, str], list] = defaultdict(list)
    for slot in payload.timetable_data:
        if slot.courseId not in shared_groups_by_course:
            continue
        grouped_events[(slot.day, slot.startTime, slot.endTime, slot.courseId, slot.roomId)].append(slot)

    for event_slots in grouped_events.values():
        if len(event_slots) < 2:
            continue
        sample = event_slots[0]
        matched_sections = set()
        for slot in event_slots[1:]:
            sample_start = parse_time_to_minutes(sample.startTime)
            sample_end = parse_time_to_minutes(sample.endTime)
            slot_start = parse_time_to_minutes(slot.startTime)
            slot_end = parse_time_to_minutes(slot.endTime)
            if is_shared_lecture_overlap_event(
                slot=sample,
                other=slot,
                slot_start=sample_start,
                slot_end=sample_end,
                other_start=slot_start,
                other_end=slot_end,
                shared_groups_by_course=shared_groups_by_course,
            ):
                matched_sections.add(sample.section)
                matched_sections.add(slot.section)

        if len(matched_sections) < 2:
            continue

        room = room_by_id[sample.roomId]
        total_students = sum(
            student_counts_by_slot.get(slot.id, 0)
            for slot in event_slots
            if slot.section in matched_sections
        )
        if total_students > getattr(room, "capacity", 0):
            if not force:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"Shared lecture event for course {sample.courseId} exceeds room capacity in room {sample.roomId}"
                    ),
                )


def load_elective_overlap_pairs(
    *,
    db: Session,
    program_id: str,
    term_number: int,
) -> set[tuple[str, str]]:
    groups = (
        db.execute(
            select(ProgramElectiveGroup).where(
                ProgramElectiveGroup.program_id == program_id,
                ProgramElectiveGroup.term_number == term_number,
                ProgramElectiveGroup.conflict_policy == ElectiveConflictPolicy.no_overlap,
            )
        )
        .scalars()
        .all()
    )
    if not groups:
        return set()

    group_ids = [group.id for group in groups]
    rows = (
        db.execute(
            select(ProgramElectiveGroupMember.group_id, ProgramCourse.course_id)
            .join(ProgramCourse, ProgramCourse.id == ProgramElectiveGroupMember.program_course_id)
            .where(ProgramElectiveGroupMember.group_id.in_(group_ids))
        )
        .all()
    )
    courses_by_group: dict[str, set[str]] = defaultdict(set)
    for group_id, course_id in rows:
        courses_by_group[group_id].add(course_id)

    conflict_pairs: set[tuple[str, str]] = set()
    for course_ids in courses_by_group.values():
        ordered = sorted(course_ids)
        for index, left in enumerate(ordered):
            for right in ordered[index + 1 :]:
                conflict_pairs.add((left, right))
    return conflict_pairs


def courses_conflict_in_elective_group(
    course_a: str,
    course_b: str,
    conflict_pairs: set[tuple[str, str]],
) -> bool:
    left, right = sorted((course_a, course_b))
    return (left, right) in conflict_pairs


def enforce_elective_overlap_constraints(
    payload: OfficialTimetablePayload,
    db: Session,
    force: bool = False,
) -> None:
    if not payload.program_id or payload.term_number is None:
        return

    conflict_pairs = load_elective_overlap_pairs(
        db=db,
        program_id=payload.program_id,
        term_number=payload.term_number,
    )
    if not conflict_pairs:
        return

    violations: set[str] = set()
    slots_by_day: dict[str, list] = defaultdict(list)
    for slot in payload.timetable_data:
        slots_by_day[slot.day].append(slot)

    for day, slots in slots_by_day.items():
        for index, slot in enumerate(slots):
            start = parse_time_to_minutes(slot.startTime)
            end = parse_time_to_minutes(slot.endTime)
            for other in slots[index + 1 :]:
                if slot.courseId == other.courseId:
                    continue
                other_start = parse_time_to_minutes(other.startTime)
                other_end = parse_time_to_minutes(other.endTime)
                if not slots_overlap(start, end, other_start, other_end):
                    continue
                if not courses_conflict_in_elective_group(slot.courseId, other.courseId, conflict_pairs):
                    continue
                left, right = sorted((slot.courseId, other.courseId))
                violations.add(
                    f"{day} {slot.startTime}-{slot.endTime}: {left}({slot.section}) vs {right}({other.section})"
                )

    if violations:
        if not force:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Elective overlap constraints violated: " + " | ".join(sorted(violations)),
            )


def enforce_prerequisite_constraints(
    payload: OfficialTimetablePayload,
    db: Session,
    force: bool = False,
) -> None:
    if not payload.program_id or payload.term_number is None:
        return

    current_program_courses = (
        db.execute(
            select(ProgramCourse).where(
                ProgramCourse.program_id == payload.program_id,
                ProgramCourse.term_number == payload.term_number,
            )
        )
        .scalars()
        .all()
    )
    if not current_program_courses:
        return

    completed_course_ids = set(
        db.execute(
            select(ProgramCourse.course_id).where(
                ProgramCourse.program_id == payload.program_id,
                ProgramCourse.term_number < payload.term_number,
            )
        )
        .scalars()
        .all()
    )

    violations: list[str] = []
    for program_course in current_program_courses:
        prerequisite_ids = set(program_course.prerequisite_course_ids or [])
        missing = sorted(prerequisite_ids - completed_course_ids)
        if missing:
            violations.append(f"{program_course.course_id} -> {', '.join(missing)}")

    if violations:
        if not force:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Prerequisite constraints are not satisfied: " + " | ".join(violations),
            )


def enforce_faculty_overload_preferences(
    payload: OfficialTimetablePayload,
    db: Session,
    force: bool = False,
) -> None:
    faculty_ids: set[str] = set()
    for slot in payload.timetable_data:
        for faculty_id in _slot_all_faculty_ids(slot):
            if _is_virtual_faculty_id(faculty_id):
                continue
            faculty_ids.add(faculty_id)
    if not faculty_ids:
        return

    faculty_records = {
        item.id: item
        for item in db.execute(select(Faculty).where(Faculty.id.in_(faculty_ids))).scalars().all()
    }

    slots_by_faculty_day: dict[tuple[str, str], list] = defaultdict(list)
    for slot in payload.timetable_data:
        for faculty_id in _slot_all_faculty_ids(slot):
            if _is_virtual_faculty_id(faculty_id):
                continue
            slots_by_faculty_day[(faculty_id, slot.day)].append(slot)

    for (faculty_id, day), slots in slots_by_faculty_day.items():
        faculty = faculty_records.get(faculty_id)
        if faculty is None:
            continue
        ordered = sorted(slots, key=lambda item: parse_time_to_minutes(item.startTime))
        previous_end: int | None = None
        consecutive_count = 0
        for slot in ordered:
            start = parse_time_to_minutes(slot.startTime)
            end = parse_time_to_minutes(slot.endTime)
            if previous_end is not None:
                gap = start - previous_end
                if gap == 0:
                    consecutive_count += 1
                else:
                    consecutive_count = 0
                if gap < 0:
                    if not force:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=(
                                f"Faculty {faculty.name} has overlapping sessions on {day}; "
                                "minimum break validation failed because slot timings overlap."
                            ),
                        )
                elif faculty.preferred_min_break_minutes > 0 and gap < faculty.preferred_min_break_minutes:
                    if not force:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=(
                                f"Faculty {faculty.name} requires at least "
                                f"{faculty.preferred_min_break_minutes} minutes break on {day}"
                            ),
                        )
                if faculty.avoid_back_to_back and gap == 0 and consecutive_count >= 1:
                    if not force:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Faculty {faculty.name} is configured to avoid back-to-back classes on {day}",
                        )
            previous_end = end


def _slot_fingerprints(payload: OfficialTimetablePayload) -> set[tuple[str, str, str, str, str, str, str, str, str]]:
    return {
        (
            slot.day,
            slot.startTime,
            slot.endTime,
            slot.courseId,
            slot.roomId,
            slot.facultyId,
            ",".join(sorted(_slot_assistant_faculty_ids(slot))),
            slot.section,
            slot.batch or "",
        )
        for slot in payload.timetable_data
    }


def _resolve_impacted_schedule_users(
    db: Session,
    old_payload: OfficialTimetablePayload | None,
    new_payload: OfficialTimetablePayload,
) -> tuple[set[str], set[str]]:
    old_slots = _slot_fingerprints(old_payload) if old_payload else set()
    new_slots = _slot_fingerprints(new_payload)
    changed_slots = old_slots.symmetric_difference(new_slots)
    if not changed_slots:
        return set(), set()

    affected_sections = {
        section.strip().upper()
        for _, _, _, _, _, _, _, section, _ in changed_slots
        if section and section.strip()
    }
    affected_faculty_ids: set[str] = set()
    for _, _, _, _, _, faculty_id, assistant_csv, _, _ in changed_slots:
        if faculty_id:
            affected_faculty_ids.add(faculty_id)
        if assistant_csv:
            for assistant_id in str(assistant_csv).split(","):
                cleaned = assistant_id.strip()
                if cleaned:
                    affected_faculty_ids.add(cleaned)

    faculty_emails: set[str] = set()
    for p_load in [p for p in (old_payload, new_payload) if p]:
        faculty_emails.update(
            item.email.strip().lower()
            for item in p_load.faculty_data
            if item.id in affected_faculty_ids and item.email and item.email.strip()
        )

    faculty_user_ids: set[str] = set()
    if faculty_emails:
        faculty_user_ids = set(
            db.execute(
                select(User.id).where(
                    User.role == UserRole.faculty,
                    func.lower(User.email).in_(faculty_emails),
                )
            ).scalars()
        )

    student_user_ids: set[str] = set()
    if affected_sections:
        students = list(
            db.execute(
                select(User.id, User.section_name).where(User.role == UserRole.student)
            ).all()
        )
        for user_id, section_name in students:
            normalized = (section_name or "").strip().upper()
            if normalized in affected_sections:
                student_user_ids.add(user_id)

    return faculty_user_ids, student_user_ids


def _version_summary(payload: OfficialTimetablePayload, conflicts: list[TimetableConflict]) -> dict:
    return {
        "program_id": payload.program_id,
        "term_number": payload.term_number,
        "slots": len(payload.timetable_data),
        "conflicts": len(conflicts),
    }


def _next_version_label(db: Session) -> str:
    versions = db.execute(select(TimetableVersion.label)).scalars().all()
    numeric = []
    for label in versions:
        if not label.startswith("v"):
            continue
        suffix = label[1:]
        if suffix.isdigit():
            numeric.append(int(suffix))
    next_index = (max(numeric) + 1) if numeric else 1
    return f"v{next_index}"


def _availability_windows_by_day(windows: list[dict]) -> dict[str, list[tuple[int, int]]]:
    normalized: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for window in windows:
        day = normalize_day(str(window.get("day", "")).strip())
        start_time = window.get("start_time")
        end_time = window.get("end_time")
        if not day or not start_time or not end_time:
            continue
        try:
            start_min = parse_time_to_minutes(start_time)
            end_min = parse_time_to_minutes(end_time)
        except ValueError:
            continue
        if end_min <= start_min:
            continue
        normalized[day].append((start_min, end_min))
    return normalized


def _build_conflicts(payload: OfficialTimetablePayload, db: Session | None = None) -> list[TimetableConflict]:
    conflicts: list[TimetableConflict] = []
    seen_pairs: set[tuple[str, str, str]] = set()
    seen_single: set[tuple[str, str]] = set()

    course_map = {course.id: course for course in payload.course_data}
    faculty_map = {faculty.id: faculty for faculty in payload.faculty_data}
    room_map = {room.id: room for room in payload.room_data}
    shared_groups_by_course: dict[str, list[set[str]]] = {}
    if db is not None and payload.program_id and payload.term_number is not None:
        shared_groups = load_shared_lecture_groups(
            db=db,
            program_id=payload.program_id,
            term_number=payload.term_number,
        )
        shared_groups_by_course = build_shared_group_lookup(shared_groups)

    slots_by_day: dict[str, list] = defaultdict(list)
    for slot in payload.timetable_data:
        slots_by_day[slot.day].append(slot)

    for day, slots in slots_by_day.items():
        for index, slot in enumerate(slots):
            start = parse_time_to_minutes(slot.startTime)
            end = parse_time_to_minutes(slot.endTime)
            for other in slots[index + 1 :]:
                other_start = parse_time_to_minutes(other.startTime)
                other_end = parse_time_to_minutes(other.endTime)
                if not slots_overlap(start, end, other_start, other_end):
                    continue

                slot_pair = tuple(sorted((slot.id, other.id)))
                allow_shared_lecture = is_shared_lecture_overlap_event(
                    slot=slot,
                    other=other,
                    slot_start=start,
                    slot_end=end,
                    other_start=other_start,
                    other_end=other_end,
                    shared_groups_by_course=shared_groups_by_course,
                )

                if slot.roomId == other.roomId and not allow_shared_lecture:
                    conflict_key = ("room-overlap", slot_pair[0], slot_pair[1])
                    if conflict_key not in seen_pairs:
                        seen_pairs.add(conflict_key)
                        room_name = room_map.get(slot.roomId).name if slot.roomId in room_map else slot.roomId
                        slot_scope = _slot_semester_section_label(
                            slot=slot,
                            course_map=course_map,
                            fallback_term_number=payload.term_number,
                        )
                        other_scope = _slot_semester_section_label(
                            slot=other,
                            course_map=course_map,
                            fallback_term_number=payload.term_number,
                        )
                        conflicts.append(
                            TimetableConflict(
                                id=f"room-{slot_pair[0]}-{slot_pair[1]}",
                                type="room-overlap",
                                severity="high",
                                description=f"Room {room_name} is double-booked on {day} ({slot_scope} and {other_scope}).",
                                affectedSlots=list(slot_pair),
                                resolution="Move one class to another room or non-overlapping time slot.",
                            )
                        )

                slot_faculty_ids = {
                    faculty_id
                    for faculty_id in _slot_all_faculty_ids(slot)
                    if not _is_virtual_faculty_id(faculty_id)
                }
                other_faculty_ids = {
                    faculty_id
                    for faculty_id in _slot_all_faculty_ids(other)
                    if not _is_virtual_faculty_id(faculty_id)
                }
                overlapping_faculty_ids = sorted(slot_faculty_ids.intersection(other_faculty_ids))
                if overlapping_faculty_ids and not allow_shared_lecture:
                    conflict_key = ("faculty-overlap", slot_pair[0], slot_pair[1])
                    if conflict_key not in seen_pairs:
                        seen_pairs.add(conflict_key)
                        faculty_name = ", ".join(
                            (
                                faculty_map.get(faculty_id).name
                                if faculty_id in faculty_map and faculty_map.get(faculty_id) is not None
                                else faculty_id
                            )
                            for faculty_id in overlapping_faculty_ids
                        )
                        slot_scope = _slot_semester_section_label(
                            slot=slot,
                            course_map=course_map,
                            fallback_term_number=payload.term_number,
                        )
                        other_scope = _slot_semester_section_label(
                            slot=other,
                            course_map=course_map,
                            fallback_term_number=payload.term_number,
                        )
                        conflicts.append(
                            TimetableConflict(
                                id=f"faculty-{slot_pair[0]}-{slot_pair[1]}",
                                type="faculty-overlap",
                                severity="high",
                                description=(
                                    f"{faculty_name} is assigned to overlapping sessions on {day} "
                                    f"({slot_scope} and {other_scope})."
                                ),
                                affectedSlots=list(slot_pair),
                                resolution="Reassign one session to another faculty member or time slot.",
                            )
                        )

                slot_course = course_map.get(slot.courseId)
                other_course = course_map.get(other.courseId)
                slot_semester = getattr(slot_course, "semester_number", None) if slot_course is not None else None
                other_semester = getattr(other_course, "semester_number", None) if other_course is not None else None
                slot_semester = slot_semester if slot_semester is not None else payload.term_number
                other_semester = other_semester if other_semester is not None else payload.term_number
                same_section_scope = (
                    slot.section == other.section
                    and (
                        slot_semester is None
                        or other_semester is None
                        or slot_semester == other_semester
                    )
                )

                if same_section_scope:
                    course_a = course_map.get(slot.courseId)
                    course_b = course_map.get(other.courseId)
                    is_parallel_lab = (
                        _slot_is_practical(slot, course_a)
                        and _slot_is_practical(other, course_b)
                        and slot.courseId == other.courseId
                        and slot.batch
                        and other.batch
                        and slot.batch != other.batch
                    )
                    if not is_parallel_lab:
                        conflict_key = ("section-overlap", slot_pair[0], slot_pair[1])
                        if conflict_key not in seen_pairs:
                            seen_pairs.add(conflict_key)
                            semester_label = (
                                f"Semester {slot_semester}"
                                if slot_semester is not None
                                else "Unknown semester"
                            )
                            conflicts.append(
                                TimetableConflict(
                                    id=f"section-{slot_pair[0]}-{slot_pair[1]}",
                                    type="section-overlap",
                                    severity="high",
                                    description=(
                                        f"Section {slot.section} ({semester_label}) has overlapping classes on {day}."
                                    ),
                                    affectedSlots=list(slot_pair),
                                    resolution="Move one class so section sessions do not overlap.",
                                )
                            )

    if db is not None and payload.program_id and payload.term_number is not None:
        conflict_pairs = load_elective_overlap_pairs(
            db=db,
            program_id=payload.program_id,
            term_number=payload.term_number,
        )
        if conflict_pairs:
            for day, slots in slots_by_day.items():
                for index, slot in enumerate(slots):
                    start = parse_time_to_minutes(slot.startTime)
                    end = parse_time_to_minutes(slot.endTime)
                    for other in slots[index + 1 :]:
                        if slot.courseId == other.courseId:
                            continue
                        if not courses_conflict_in_elective_group(slot.courseId, other.courseId, conflict_pairs):
                            continue
                        other_start = parse_time_to_minutes(other.startTime)
                        other_end = parse_time_to_minutes(other.endTime)
                        if not slots_overlap(start, end, other_start, other_end):
                            continue

                        slot_pair = tuple(sorted((slot.id, other.id)))
                        conflict_key = ("elective-overlap", slot_pair[0], slot_pair[1])
                        if conflict_key not in seen_pairs:
                            seen_pairs.add(conflict_key)
                            conflicts.append(
                                TimetableConflict(
                                    id=f"elective-{slot_pair[0]}-{slot_pair[1]}",
                                    type="elective-overlap",
                                    severity="medium",
                                    description=(
                                        f"Elective courses {slot.courseId} and {other.courseId} overlap on {day} "
                                        "for a configured elective group."
                                    ),
                                    affectedSlots=list(slot_pair),
                                    resolution=(
                                        "Move one elective to a different time slot to avoid overlap "
                                        "for eligible student groups."
                                    ),
                                )
                            )

    faculty_ids_by_course_section: dict[tuple[str, str], set[str]] = defaultdict(set)
    slot_ids_by_course_section: dict[tuple[str, str], list[str]] = defaultdict(list)
    for slot in payload.timetable_data:
        course = course_map.get(slot.courseId)
        if _slot_is_practical(slot, course):
            continue
        if course is not None and not bool(getattr(course, "assign_faculty", True)):
            continue
        if _is_virtual_faculty_id(slot.facultyId):
            continue
        key = (slot.courseId, slot.section)
        faculty_ids_by_course_section[key].add(slot.facultyId)
        slot_ids_by_course_section[key].append(slot.id)

    for (course_id, section_name), faculty_ids in faculty_ids_by_course_section.items():
        if len(faculty_ids) <= 1:
            continue
        key = ("course-faculty-inconsistency", course_id, section_name)
        if key in seen_single:
            continue
        seen_single.add(key)
        course = course_map.get(course_id)
        course_label = course.code if course is not None else course_id
        faculty_labels = ", ".join(
            sorted(faculty_map.get(faculty_id).name if faculty_id in faculty_map else faculty_id for faculty_id in faculty_ids)
        )
        conflicts.append(
            TimetableConflict(
                id=f"course-faculty-{course_id}-{section_name}",
                type="course-faculty-inconsistency",
                severity="high",
                description=(
                    f"Course {course_label} in section {section_name} is assigned to multiple faculty: "
                    f"{faculty_labels}."
                ),
                affectedSlots=sorted(slot_ids_by_course_section.get((course_id, section_name), [])),
                resolution="Assign one faculty member to this course within the section.",
            )
        )

    for slot in payload.timetable_data:
        room = room_map.get(slot.roomId)
        if room is not None and slot.studentCount is not None and slot.studentCount > room.capacity:
            key = ("capacity", slot.id)
            if key not in seen_single:
                seen_single.add(key)
                conflicts.append(
                    TimetableConflict(
                        id=f"capacity-{slot.id}",
                        type="capacity",
                        severity="medium",
                        description=(
                            f"Room {room.name} capacity ({room.capacity}) is below "
                            f"student count ({slot.studentCount}) for slot {slot.id}."
                        ),
                        affectedSlots=[slot.id],
                        resolution="Assign a larger room or reduce section/batch size for this slot.",
                    )
                )

    faculty_windows = {item.id: _availability_windows_by_day(getattr(item, "availability_windows", [])) for item in payload.faculty_data}
    room_windows = {item.id: _availability_windows_by_day(getattr(item, "availability_windows", [])) for item in payload.room_data}

    for slot in payload.timetable_data:
        start = parse_time_to_minutes(slot.startTime)
        end = parse_time_to_minutes(slot.endTime)
        day = slot.day

        faculty = faculty_map.get(slot.facultyId)
        if faculty is not None:
            if faculty.availability:
                allowed_days = {normalize_day(item) for item in faculty.availability}
                if day not in allowed_days:
                    key = ("availability", f"faculty-day-{slot.id}")
                    if key not in seen_single:
                        seen_single.add(key)
                        conflicts.append(
                            TimetableConflict(
                                id=f"availability-faculty-day-{slot.id}",
                                type="availability",
                                severity="medium",
                                description=f"{faculty.name} is scheduled on unavailable day {day}.",
                                affectedSlots=[slot.id],
                                resolution="Move the class to a day marked available by the faculty member.",
                            )
                        )

            day_windows = faculty_windows.get(faculty.id, {}).get(day, [])
            if day_windows and not any(window_start <= start and end <= window_end for window_start, window_end in day_windows):
                key = ("availability", f"faculty-window-{slot.id}")
                if key not in seen_single:
                    seen_single.add(key)
                    conflicts.append(
                        TimetableConflict(
                            id=f"availability-faculty-window-{slot.id}",
                            type="availability",
                            severity="medium",
                            description=f"{faculty.name} is scheduled outside configured availability window on {day}.",
                            affectedSlots=[slot.id],
                            resolution="Shift class timing to match the faculty availability window.",
                        )
                    )

        room = room_map.get(slot.roomId)
        if room is not None:
            day_windows = room_windows.get(room.id, {}).get(day, [])
            if day_windows and not any(window_start <= start and end <= window_end for window_start, window_end in day_windows):
                key = ("availability", f"room-window-{slot.id}")
                if key not in seen_single:
                    seen_single.add(key)
                    conflicts.append(
                        TimetableConflict(
                            id=f"availability-room-window-{slot.id}",
                            type="availability",
                            severity="medium",
                            description=f"Room {room.name} is scheduled outside configured availability window on {day}.",
                            affectedSlots=[slot.id],
                            resolution="Move session to a room-available window or a different room.",
                        )
                    )

    if db is not None:
        schedule_policy = load_schedule_policy(db)
        period_minutes = schedule_policy.period_minutes
        grouped_courses: dict[tuple[str, str, str, str | None], list] = defaultdict(list)

        for slot in payload.timetable_data:
            course = course_map.get(slot.courseId)
            if course is None:
                continue
            requires_batch_segregation = _course_batch_segregation_enabled(course)
            slot_is_practical = _slot_is_practical(slot, course)
            session_bucket = "practical" if slot_is_practical else "lecture"
            grouped_courses[
                (
                    slot.courseId,
                    slot.section,
                    session_bucket,
                    slot.batch if (slot_is_practical and requires_batch_segregation) else None,
                )
            ].append(slot)

        for (course_id, section, session_bucket, batch), slots in grouped_courses.items():
            course = course_map.get(course_id)
            if course is None:
                continue

            theory_units = max(0, int(getattr(course, "theory_hours", getattr(course, "theoryHours", 0)) or 0))
            tutorial_units = max(0, int(getattr(course, "tutorial_hours", getattr(course, "tutorialHours", 0)) or 0))
            practical_units = _course_practical_hours(course)
            lecture_units = theory_units + tutorial_units
            required_units = practical_units if session_bucket == "practical" else lecture_units

            required_minutes = required_units * period_minutes
            slots_sorted = sorted(slots, key=lambda s: (s.day, parse_time_to_minutes(s.startTime)))
            total_minutes = sum(parse_time_to_minutes(slot.endTime) - parse_time_to_minutes(slot.startTime) for slot in slots_sorted)
            slot_ids = [slot.id for slot in slots_sorted]

            key_suffix = f"{course_id}-{section}-{session_bucket}-{batch or 'all'}"
            if required_minutes and total_minutes != required_minutes:
                key = ("availability", f"course-duration-{key_suffix}")
                if key not in seen_single:
                    seen_single.add(key)
                    label = f"{course_id} section {section}" + (f" batch {batch}" if batch else "")
                    conflicts.append(
                        TimetableConflict(
                            id=f"availability-course-duration-{key_suffix}",
                            type="availability",
                            severity="high",
                            description=(
                                f"Scheduled duration for {label} is {total_minutes} minutes; "
                                f"expected {required_minutes} minutes per week."
                            ),
                            affectedSlots=slot_ids,
                            resolution="Adjust slot count/duration to match required weekly course minutes.",
                        )
                    )

            if session_bucket != "practical" or required_units <= 0:
                continue

            preferred_block_slots = _lab_block_slots_for_course(course, schedule_policy)
            full_blocks, remainder = divmod(required_units, preferred_block_slots)
            expected_block_lengths: list[int] = [preferred_block_slots * period_minutes] * full_blocks
            if remainder > 0:
                expected_block_lengths.append(remainder * period_minutes)

            blocks: list[int] = []
            current_day: str | None = None
            current_start: int | None = None
            current_end: int | None = None

            for slot in slots_sorted:
                slot_start = parse_time_to_minutes(slot.startTime)
                slot_end = parse_time_to_minutes(slot.endTime)
                if current_day != slot.day or current_end is None or slot_start != current_end:
                    if current_day is not None and current_start is not None and current_end is not None:
                        blocks.append(current_end - current_start)
                    current_day = slot.day
                    current_start = slot_start
                    current_end = slot_end
                else:
                    current_end = slot_end

            if current_day is not None and current_start is not None and current_end is not None:
                blocks.append(current_end - current_start)

            blocks_sorted = sorted(blocks)
            expected_sorted = sorted(expected_block_lengths)

            if expected_sorted and len(blocks_sorted) != len(expected_sorted):
                key = ("availability", f"practical-block-count-{key_suffix}")
                if key not in seen_single:
                    seen_single.add(key)
                    conflicts.append(
                        TimetableConflict(
                            id=f"availability-practical-block-count-{key_suffix}",
                            type="availability",
                            severity="high",
                            description=f"Practical sessions for {course_id} must be scheduled in contiguous blocks.",
                            affectedSlots=slot_ids,
                            resolution="Merge fragmented practical periods into contiguous blocks.",
                        )
                    )
            elif expected_sorted and blocks_sorted != expected_sorted:
                key = ("availability", f"practical-block-size-{key_suffix}")
                if key not in seen_single:
                    seen_single.add(key)
                    conflicts.append(
                        TimetableConflict(
                            id=f"availability-practical-block-size-{key_suffix}",
                            type="availability",
                            severity="high",
                            description=(
                                f"Practical sessions for {course_id} must use contiguous blocks of "
                                f"{preferred_block_slots} period(s) by default."
                            ),
                            affectedSlots=slot_ids,
                            resolution="Restructure practical slots to match configured contiguous block size.",
                        )
                    )

    return conflicts


def _slot_duration_minutes(slot: object) -> int:
    return parse_time_to_minutes(slot.endTime) - parse_time_to_minutes(slot.startTime)


def _room_matches_course_type(room: object, course: object | None, *, session_type: str | None = None) -> bool:
    if course is None:
        return True
    if session_type == "lab":
        return getattr(room, "type", None) == "lab"
    if session_type is None and getattr(course, "type", None) == "lab":
        return getattr(room, "type", None) == "lab"
    return True


def _faculty_available_for_window(
    *,
    faculty_payload: object,
    faculty_record: Faculty | None,
    day: str,
    start: int,
    end: int,
) -> bool:
    availability = (
        list(getattr(faculty_record, "availability", []))
        if faculty_record is not None
        else list(getattr(faculty_payload, "availability", []))
    )
    if availability:
        allowed_days = {normalize_day(item) for item in availability}
        if day not in allowed_days:
            return False

    windows = list(getattr(faculty_record, "availability_windows", [])) if faculty_record is not None else []
    if windows:
        by_day = _availability_windows_by_day(windows)
        day_windows = by_day.get(day, [])
        if not day_windows:
            return False
        if not any(window_start <= start and end <= window_end for window_start, window_end in day_windows):
            return False
    return True


def _room_available_for_window(
    *,
    room_record: object | None,
    day: str,
    start: int,
    end: int,
) -> bool:
    windows = list(getattr(room_record, "availability_windows", [])) if room_record is not None else []
    if not windows:
        return True
    by_day = _availability_windows_by_day(windows)
    day_windows = by_day.get(day, [])
    if not day_windows:
        return False
    return any(window_start <= start and end <= window_end for window_start, window_end in day_windows)


def _is_parallel_lab_allowed(slot: object, other: object, course_map: dict[str, object]) -> bool:
    course_a = course_map.get(slot.courseId)
    course_b = course_map.get(other.courseId)
    return (
        _slot_is_practical(slot, course_a)
        and _slot_is_practical(other, course_b)
        and slot.courseId == other.courseId
        and slot.batch
        and other.batch
        and slot.batch != other.batch
    )


def _resource_placement_conflicts(
    *,
    payload: OfficialTimetablePayload,
    slot_id: str,
    course_id: str,
    section: str,
    batch: str | None,
    day: str,
    start: int,
    end: int,
    room_id: str,
    faculty_id: str,
    course_map: dict[str, object],
    elective_pairs: set[tuple[str, str]],
    moving_assistant_ids: tuple[str, ...] = (),
    ignore_slot_ids: set[str] | None = None,
) -> bool:
    ignored_ids = ignore_slot_ids or set()
    moving_faculty_ids = {faculty_id}
    moving_faculty_ids.update(item for item in moving_assistant_ids if item and not _is_virtual_faculty_id(item))
    for other in payload.timetable_data:
        if other.id == slot_id or other.id in ignored_ids:
            continue
        if other.day != day:
            continue

        other_start = parse_time_to_minutes(other.startTime)
        other_end = parse_time_to_minutes(other.endTime)
        if not slots_overlap(start, end, other_start, other_end):
            continue

        if other.roomId == room_id:
            return True
        other_faculty_ids = {
            other_faculty_id
            for other_faculty_id in _slot_all_faculty_ids(other)
            if not _is_virtual_faculty_id(other_faculty_id)
        }
        if moving_faculty_ids.intersection(other_faculty_ids):
            return True
        if other.section == section:
            probe = deepcopy(other.model_dump())
            probe["courseId"] = course_id
            probe["batch"] = batch
            probe_slot = type(other).model_validate(probe)
            if not _is_parallel_lab_allowed(probe_slot, other, course_map):
                return True
        if courses_conflict_in_elective_group(course_id, other.courseId, elective_pairs):
            return True
    return False


def _build_time_block_candidates(
    *,
    payload: OfficialTimetablePayload,
    db: Session,
    duration_minutes: int,
) -> list[tuple[str, int, int]]:
    if duration_minutes <= 0:
        return []
    schedule_policy = load_schedule_policy(db)
    if duration_minutes % schedule_policy.period_minutes != 0:
        return []

    working_hours = load_working_hours(db)
    candidates: list[tuple[str, int, int]] = []
    for day, entry in working_hours.items():
        if not entry.enabled:
            continue
        segments = build_teaching_segments(
            day_start=parse_time_to_minutes(entry.start_time),
            day_end=parse_time_to_minutes(entry.end_time),
            period_minutes=schedule_policy.period_minutes,
            breaks=schedule_policy.breaks,
        )
        for segment_start, _ in segments:
            segment_end = segment_start + duration_minutes
            if is_slot_aligned_with_segments(segment_start, segment_end, segments):
                candidates.append((day, segment_start, segment_end))
    return candidates


def _find_room_candidate(
    *,
    payload: OfficialTimetablePayload,
    slot: object,
    course_map: dict[str, object],
    db_room_map: dict[str, object],
    elective_pairs: set[tuple[str, str]],
    day: str,
    start: int,
    end: int,
    current_faculty_id: str,
    moving_assistant_ids: tuple[str, ...] = (),
    ignore_slot_ids: set[str] | None = None,
) -> str | None:
    course = course_map.get(slot.courseId)
    ranked: list[tuple[int, int, str]] = []
    for room in payload.room_data:
        if room.id == slot.roomId:
            continue
        if not _room_matches_course_type(room, course, session_type=getattr(slot, "sessionType", None)):
            continue
        if slot.studentCount is not None and room.capacity < slot.studentCount:
            continue
        if not _room_available_for_window(
            room_record=db_room_map.get(room.id),
            day=day,
            start=start,
            end=end,
        ):
            continue
        if _resource_placement_conflicts(
            payload=payload,
            slot_id=slot.id,
            course_id=slot.courseId,
            section=slot.section,
            batch=slot.batch,
            day=day,
            start=start,
            end=end,
            room_id=room.id,
            faculty_id=current_faculty_id,
            course_map=course_map,
            elective_pairs=elective_pairs,
            moving_assistant_ids=moving_assistant_ids,
            ignore_slot_ids=ignore_slot_ids,
        ):
            continue
        capacity_delta = room.capacity - (slot.studentCount or 0)
        ranked.append((capacity_delta if capacity_delta >= 0 else 10_000, room.capacity, room.id))
    ranked.sort(key=lambda item: (item[0], item[1]))
    return ranked[0][2] if ranked else None


def _find_faculty_candidate(
    *,
    payload: OfficialTimetablePayload,
    slot: object,
    course_map: dict[str, object],
    db_faculty_map: dict[str, Faculty],
    elective_pairs: set[tuple[str, str]],
    day: str,
    start: int,
    end: int,
    current_room_id: str,
    moving_assistant_ids: tuple[str, ...] = (),
    ignore_slot_ids: set[str] | None = None,
) -> str | None:
    course = course_map.get(slot.courseId)
    course_code = str(getattr(course, "code", "")).strip().upper()
    faculty_payload_map = {item.id: item for item in payload.faculty_data}
    assigned_minutes: dict[str, int] = defaultdict(int)
    for item in payload.timetable_data:
        assigned_minutes[item.facultyId] += _slot_duration_minutes(item)

    ranked: list[tuple[float, str]] = []
    for faculty_id, faculty_payload in faculty_payload_map.items():
        if faculty_id == slot.facultyId:
            continue
        faculty_record = db_faculty_map.get(faculty_id)
        if not _faculty_available_for_window(
            faculty_payload=faculty_payload,
            faculty_record=faculty_record,
            day=day,
            start=start,
            end=end,
        ):
            continue

        max_hours = (
            faculty_record.max_hours
            if faculty_record is not None
            else int(getattr(faculty_payload, "maxHours", 0))
        )
        projected_minutes = assigned_minutes.get(faculty_id, 0) + (end - start)
        if max_hours and projected_minutes > (max_hours * 60):
            continue

        if _resource_placement_conflicts(
            payload=payload,
            slot_id=slot.id,
            course_id=slot.courseId,
            section=slot.section,
            batch=slot.batch,
            day=day,
            start=start,
            end=end,
            room_id=current_room_id,
            faculty_id=faculty_id,
            course_map=course_map,
            elective_pairs=elective_pairs,
            moving_assistant_ids=moving_assistant_ids,
            ignore_slot_ids=ignore_slot_ids,
        ):
            continue

        preferred_codes = {
            str(item).strip().upper()
            for item in (
                faculty_record.preferred_subject_codes if faculty_record is not None else []
            )
            if str(item).strip()
        }
        preference_bonus = 100.0 if course_code and course_code in preferred_codes else 0.0
        department_bonus = 20.0 if getattr(faculty_payload, "department", None) == getattr(
            faculty_payload_map.get(slot.facultyId), "department", None
        ) else 0.0
        workload_balance = max(0.0, float(max_hours * 60 - assigned_minutes.get(faculty_id, 0)) / 60.0)
        score = preference_bonus + department_bonus + workload_balance
        ranked.append((score, faculty_id))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[0][1] if ranked else None


def _append_unique_identifier(items: list[str], candidate: str | None) -> None:
    if not candidate:
        return
    if candidate in items:
        return
    items.append(candidate)


def _resolve_course_faculty_inconsistency_conflict(
    *,
    payload: OfficialTimetablePayload,
    conflict: TimetableConflict,
    slots_by_id: dict[str, object],
    db_faculty_map: dict[str, Faculty],
    course_map: dict[str, object],
    elective_pairs: set[tuple[str, str]],
) -> tuple[OfficialTimetablePayload | None, str]:
    affected_slot_ids = [slot_id for slot_id in conflict.affected_slots if slot_id in slots_by_id]
    if not affected_slot_ids:
        return None, "No actionable slots found for faculty consistency repair."

    target_slots = [slots_by_id[slot_id] for slot_id in affected_slot_ids]
    if not target_slots:
        return None, "No actionable slots found for faculty consistency repair."

    faculty_payload_map = {item.id: item for item in payload.faculty_data}
    if not faculty_payload_map:
        return None, "Faculty metadata is unavailable for consistency repair."

    course = course_map.get(target_slots[0].courseId)
    course_code = str(getattr(course, "code", "")).strip().upper()
    faculty_usage_by_slot: dict[str, int] = defaultdict(int)
    assigned_minutes: dict[str, int] = defaultdict(int)
    for item in payload.timetable_data:
        assigned_minutes[item.facultyId] += _slot_duration_minutes(item)
    for slot in target_slots:
        faculty_usage_by_slot[slot.facultyId] += 1

    prioritized_candidates: list[str] = []
    for faculty_id, _count in sorted(
        faculty_usage_by_slot.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        _append_unique_identifier(prioritized_candidates, faculty_id)
    for faculty_id in sorted(faculty_payload_map):
        _append_unique_identifier(prioritized_candidates, faculty_id)

    best_faculty_id: str | None = None
    best_score: tuple[float, float, float] | None = None
    for faculty_id in prioritized_candidates:
        faculty_payload = faculty_payload_map.get(faculty_id)
        if faculty_payload is None:
            continue
        faculty_record = db_faculty_map.get(faculty_id)

        additional_minutes = sum(
            _slot_duration_minutes(slot)
            for slot in target_slots
            if slot.facultyId != faculty_id
        )
        max_hours = (
            faculty_record.max_hours
            if faculty_record is not None
            else int(getattr(faculty_payload, "maxHours", 0))
        )
        projected_minutes = assigned_minutes.get(faculty_id, 0) + additional_minutes
        if max_hours and projected_minutes > (max_hours * 60):
            continue

        feasible = True
        for slot in target_slots:
            start = parse_time_to_minutes(slot.startTime)
            end = parse_time_to_minutes(slot.endTime)
            if not _faculty_available_for_window(
                faculty_payload=faculty_payload,
                faculty_record=faculty_record,
                day=slot.day,
                start=start,
                end=end,
            ):
                feasible = False
                break
            if _resource_placement_conflicts(
                payload=payload,
                slot_id=slot.id,
                course_id=slot.courseId,
                section=slot.section,
                batch=slot.batch,
                day=slot.day,
                start=start,
                end=end,
                room_id=slot.roomId,
                faculty_id=faculty_id,
                course_map=course_map,
                elective_pairs=elective_pairs,
                moving_assistant_ids=_slot_assistant_faculty_ids(slot),
            ):
                feasible = False
                break
        if not feasible:
            continue

        preferred_codes = {
            str(item).strip().upper()
            for item in (faculty_record.preferred_subject_codes if faculty_record is not None else [])
            if str(item).strip()
        }
        semester_preferences = (
            dict(faculty_record.semester_preferences or {})
            if faculty_record is not None
            else {}
        )
        term_specific = semester_preferences.get(str(payload.term_number), [])
        preferred_codes.update(str(item).strip().upper() for item in term_specific if str(item).strip())
        preferred_bonus = 1.0 if course_code and course_code in preferred_codes else 0.0
        continuity_bonus = float(faculty_usage_by_slot.get(faculty_id, 0))
        remaining_capacity = (
            float((max_hours * 60) - projected_minutes)
            if max_hours
            else 1_000_000.0
        )
        candidate_score = (continuity_bonus, preferred_bonus, remaining_capacity)
        if best_score is None or candidate_score > best_score:
            best_faculty_id = faculty_id
            best_score = candidate_score

    if best_faculty_id is None:
        return None, "No common feasible faculty found to harmonize this course within the section."

    if all(slot.facultyId == best_faculty_id for slot in target_slots):
        return None, "Course-section already uses a consistent faculty assignment."

    for slot in target_slots:
        slot.facultyId = best_faculty_id
        _prune_primary_from_slot_assistants(slot, best_faculty_id)
    faculty_name = db_faculty_map.get(best_faculty_id).name if best_faculty_id in db_faculty_map else best_faculty_id
    return payload, f"Resolved by harmonizing all related slots to faculty {faculty_name}."


def _faculty_minutes_with_assistants(payload: OfficialTimetablePayload) -> dict[str, int]:
    minutes_by_faculty: dict[str, int] = defaultdict(int)
    for slot in payload.timetable_data:
        duration = _slot_duration_minutes(slot)
        for faculty_id in _slot_all_faculty_ids(slot):
            if _is_virtual_faculty_id(faculty_id):
                continue
            minutes_by_faculty[faculty_id] += duration
    return minutes_by_faculty


def _faculty_has_timeslot_overlap(
    *,
    payload: OfficialTimetablePayload,
    faculty_id: str,
    day: str,
    start: int,
    end: int,
    ignore_slot_ids: set[str] | None = None,
) -> bool:
    ignored = ignore_slot_ids or set()
    for slot in payload.timetable_data:
        if slot.id in ignored:
            continue
        if slot.day != day:
            continue
        other_start = parse_time_to_minutes(slot.startTime)
        other_end = parse_time_to_minutes(slot.endTime)
        if not slots_overlap(start, end, other_start, other_end):
            continue
        faculty_ids = {
            item
            for item in _slot_all_faculty_ids(slot)
            if not _is_virtual_faculty_id(item)
        }
        if faculty_id in faculty_ids:
            return True
    return False


def _resolve_assistant_faculty_overlap_conflict(
    *,
    payload: OfficialTimetablePayload,
    conflict: TimetableConflict,
    slots_by_id: dict[str, object],
    db_faculty_map: dict[str, Faculty],
    faculty_payload_map: dict[str, object],
) -> tuple[OfficialTimetablePayload | None, str]:
    slot_ids = [slot_id for slot_id in conflict.affected_slots if slot_id in slots_by_id]
    if len(slot_ids) < 2:
        return None, "Assistant overlap repair requires at least two affected slots."

    slot_a = slots_by_id[slot_ids[0]]
    slot_b = slots_by_id[slot_ids[1]]
    shared_faculty_ids = {
        faculty_id
        for faculty_id in _slot_all_faculty_ids(slot_a)
        if not _is_virtual_faculty_id(faculty_id)
    }.intersection(
        {
            faculty_id
            for faculty_id in _slot_all_faculty_ids(slot_b)
            if not _is_virtual_faculty_id(faculty_id)
        }
    )
    if not shared_faculty_ids:
        return None, "No assistant-driven overlap detected for this conflict pair."

    assigned_minutes = _faculty_minutes_with_assistants(payload)

    for overlapping_faculty_id in sorted(shared_faculty_ids):
        for target_slot, other_slot in ((slot_a, slot_b), (slot_b, slot_a)):
            assistant_ids = list(_slot_assistant_faculty_ids(target_slot))
            if overlapping_faculty_id not in assistant_ids:
                continue

            slot_start = parse_time_to_minutes(target_slot.startTime)
            slot_end = parse_time_to_minutes(target_slot.endTime)
            slot_duration = slot_end - slot_start
            if slot_duration <= 0:
                continue

            ranked_candidates: list[tuple[float, str]] = []
            for candidate_id, faculty_payload in faculty_payload_map.items():
                if candidate_id == target_slot.facultyId:
                    continue
                if candidate_id in assistant_ids:
                    continue
                if candidate_id == overlapping_faculty_id:
                    continue
                if _is_virtual_faculty_id(candidate_id):
                    continue
                if candidate_id in _slot_all_faculty_ids(other_slot):
                    continue

                faculty_record = db_faculty_map.get(candidate_id)
                if not _faculty_available_for_window(
                    faculty_payload=faculty_payload,
                    faculty_record=faculty_record,
                    day=target_slot.day,
                    start=slot_start,
                    end=slot_end,
                ):
                    continue

                max_hours = (
                    int(faculty_record.max_hours)
                    if faculty_record is not None and getattr(faculty_record, "max_hours", 0)
                    else int(getattr(faculty_payload, "maxHours", 0))
                )
                projected_minutes = assigned_minutes.get(candidate_id, 0) + slot_duration
                if max_hours and projected_minutes > (max_hours * 60):
                    continue

                if _faculty_has_timeslot_overlap(
                    payload=payload,
                    faculty_id=candidate_id,
                    day=target_slot.day,
                    start=slot_start,
                    end=slot_end,
                    ignore_slot_ids={target_slot.id},
                ):
                    continue

                preferred_codes = {
                    str(item).strip().upper()
                    for item in (
                        faculty_record.preferred_subject_codes if faculty_record is not None else []
                    )
                    if str(item).strip()
                }
                course_code = ""
                for candidate_course in payload.course_data:
                    if candidate_course.id == target_slot.courseId:
                        course_code = str(candidate_course.code or "").strip().upper()
                        break
                preference_bonus = 100.0 if course_code and course_code in preferred_codes else 0.0
                remaining_capacity = float((max_hours * 60) - projected_minutes) if max_hours else 1000000.0
                ranked_candidates.append((preference_bonus + remaining_capacity, candidate_id))

            ranked_candidates.sort(key=lambda item: item[0], reverse=True)
            if not ranked_candidates:
                continue

            replacement_id = ranked_candidates[0][1]
            replaced_assistants = [replacement_id if item == overlapping_faculty_id else item for item in assistant_ids]
            deduped: list[str] = []
            for assistant_id in replaced_assistants:
                if assistant_id == target_slot.facultyId or assistant_id in deduped:
                    continue
                deduped.append(assistant_id)
            _set_slot_assistant_faculty_ids(target_slot, deduped)

            overlapping_name = (
                db_faculty_map.get(overlapping_faculty_id).name
                if db_faculty_map.get(overlapping_faculty_id) is not None
                else overlapping_faculty_id
            )
            replacement_name = (
                db_faculty_map.get(replacement_id).name
                if db_faculty_map.get(replacement_id) is not None
                else replacement_id
            )
            return (
                payload,
                f"Resolved by replacing assistant faculty {overlapping_name} with {replacement_name}.",
            )

    return None, "No assistant reassignment candidate could resolve this overlap."


def _resolve_practical_contiguity_conflict(
    *,
    payload: OfficialTimetablePayload,
    conflict: TimetableConflict,
    slots_by_id: dict[str, object],
    course_map: dict[str, object],
    db_faculty_map: dict[str, Faculty],
    db_room_map: dict[str, object],
    faculty_payload_map: dict[str, object],
    room_payload_map: dict[str, object],
    elective_pairs: set[tuple[str, str]],
    db: Session,
) -> tuple[OfficialTimetablePayload | None, str]:
    slot_ids = [slot_id for slot_id in conflict.affected_slots if slot_id in slots_by_id]
    if len(slot_ids) < 2:
        return None, "Practical contiguity resolver needs at least two slots."

    slots = [slots_by_id[slot_id] for slot_id in slot_ids]
    base_slot = slots[0]
    base_course = course_map.get(base_slot.courseId)
    if base_course is None or not _slot_is_practical(base_slot, base_course):
        return None, "Affected slots are not practical sessions."

    for slot in slots:
        if slot.courseId != base_slot.courseId or slot.section != base_slot.section or (slot.batch or None) != (base_slot.batch or None):
            return None, "Practical contiguity repair supports one course-section-batch at a time."
        if not _slot_is_practical(slot, base_course):
            return None, "Affected slots include non-practical entries."

    schedule_policy = load_schedule_policy(db)
    period_minutes = schedule_policy.period_minutes
    total_units = 0
    for slot in slots:
        duration = _slot_duration_minutes(slot)
        if duration <= 0 or duration % period_minutes != 0:
            return None, "Practical slot durations are invalid for contiguous block repair."
        total_units += duration // period_minutes
    if total_units <= 1:
        return None, "Practical block already too small for contiguity repair."

    preferred_block_slots = _lab_block_slots_for_course(base_course, schedule_policy)
    full_blocks, remainder = divmod(total_units, preferred_block_slots)
    block_units = [preferred_block_slots] * full_blocks
    if remainder > 0:
        block_units.append(remainder)
    if not block_units:
        return None, "No practical blocks computed for repair."

    primary_faculty_candidates: list[str] = []
    for slot in slots:
        if slot.facultyId and slot.facultyId not in primary_faculty_candidates:
            primary_faculty_candidates.append(slot.facultyId)
    if not primary_faculty_candidates:
        primary_faculty_candidates = [base_slot.facultyId]

    room_candidates: list[str] = []
    for slot in slots:
        if slot.roomId and slot.roomId not in room_candidates:
            room_candidates.append(slot.roomId)
    if not room_candidates:
        room_candidates = [base_slot.roomId]

    sorted_slots = sorted(slots, key=lambda item: (_day_sort_index(item.day), parse_time_to_minutes(item.startTime), item.id))
    unplaced_slots = list(sorted_slots)
    pending_slot_ids = {slot.id for slot in sorted_slots}

    for units in block_units:
        duration_minutes = units * period_minutes
        time_candidates = _build_time_block_candidates(
            payload=payload,
            db=db,
            duration_minutes=duration_minutes,
        )
        if not time_candidates:
            return None, "No timetable block candidates available for practical contiguity repair."

        sorted_time_candidates = sorted(
            time_candidates,
            key=lambda item: (
                0 if item[0] == base_slot.day else 1,
                abs(_day_sort_index(item[0]) - _day_sort_index(base_slot.day)),
                abs(item[1] - parse_time_to_minutes(base_slot.startTime)),
            ),
        )

        selected_assignment: tuple[str, int, int, str, str] | None = None
        for day, block_start, block_end in sorted_time_candidates:
            for candidate_room_id in room_candidates:
                room_record = db_room_map.get(candidate_room_id)
                if not _room_available_for_window(room_record=room_record, day=day, start=block_start, end=block_end):
                    continue
                for candidate_faculty_id in primary_faculty_candidates:
                    faculty_payload = faculty_payload_map.get(candidate_faculty_id)
                    if faculty_payload is None:
                        continue
                    if not _faculty_available_for_window(
                        faculty_payload=faculty_payload,
                        faculty_record=db_faculty_map.get(candidate_faculty_id),
                        day=day,
                        start=block_start,
                        end=block_end,
                    ):
                        continue
                    if _resource_placement_conflicts(
                        payload=payload,
                        slot_id="__practical-block__",
                        course_id=base_slot.courseId,
                        section=base_slot.section,
                        batch=base_slot.batch,
                        day=day,
                        start=block_start,
                        end=block_end,
                        room_id=candidate_room_id,
                        faculty_id=candidate_faculty_id,
                        course_map=course_map,
                        elective_pairs=elective_pairs,
                        moving_assistant_ids=_slot_assistant_faculty_ids(base_slot),
                        ignore_slot_ids=pending_slot_ids,
                    ):
                        continue
                    selected_assignment = (day, block_start, block_end, candidate_room_id, candidate_faculty_id)
                    break
                if selected_assignment is not None:
                    break
            if selected_assignment is not None:
                break

        if selected_assignment is None:
            return None, "Could not find a conflict-free contiguous practical block assignment."

        day, block_start, _block_end, candidate_room_id, candidate_faculty_id = selected_assignment
        for offset in range(units):
            if not unplaced_slots:
                break
            slot = unplaced_slots.pop(0)
            start_minutes = block_start + (offset * period_minutes)
            end_minutes = start_minutes + period_minutes
            slot.day = day
            slot.startTime = _minutes_to_time(start_minutes)
            slot.endTime = _minutes_to_time(end_minutes)
            slot.roomId = candidate_room_id
            slot.facultyId = candidate_faculty_id
            _prune_primary_from_slot_assistants(slot, candidate_faculty_id)
            pending_slot_ids.discard(slot.id)

    if unplaced_slots:
        return None, "Practical contiguity repair could not place all affected slots."

    return payload, "Resolved by moving practical sessions into contiguous block(s)."


def _apply_best_effort_resolution(
    *,
    payload: OfficialTimetablePayload,
    conflict: TimetableConflict,
    db: Session,
) -> tuple[OfficialTimetablePayload | None, str]:
    slots_by_id = {slot.id: slot for slot in payload.timetable_data}
    course_map = {course.id: course for course in payload.course_data}
    db_faculty_map = {item.id: item for item in db.execute(select(Faculty)).scalars().all()}
    db_room_map = {item.id: item for item in db.execute(select(Room)).scalars().all()}
    faculty_payload_map = {item.id: item for item in payload.faculty_data}
    room_payload_map = {item.id: item for item in payload.room_data}
    elective_pairs: set[tuple[str, str]] = set()
    if payload.program_id and payload.term_number is not None:
        elective_pairs = load_elective_overlap_pairs(
            db=db,
            program_id=payload.program_id,
            term_number=payload.term_number,
        )

    if conflict.type == "course-faculty-inconsistency":
        resolved_payload, message = _resolve_course_faculty_inconsistency_conflict(
            payload=payload,
            conflict=conflict,
            slots_by_id=slots_by_id,
            db_faculty_map=db_faculty_map,
            course_map=course_map,
            elective_pairs=elective_pairs,
        )
        if resolved_payload is not None:
            return resolved_payload, message
        return None, message

    if conflict.type == "faculty-overlap":
        resolved_payload, message = _resolve_assistant_faculty_overlap_conflict(
            payload=payload,
            conflict=conflict,
            slots_by_id=slots_by_id,
            db_faculty_map=db_faculty_map,
            faculty_payload_map=faculty_payload_map,
        )
        if resolved_payload is not None:
            return resolved_payload, message

    if (
        conflict.type == "availability"
        and "practical sessions for" in conflict.description.lower()
        and "contiguous block" in conflict.description.lower()
    ):
        resolved_payload, message = _resolve_practical_contiguity_conflict(
            payload=payload,
            conflict=conflict,
            slots_by_id=slots_by_id,
            course_map=course_map,
            db_faculty_map=db_faculty_map,
            db_room_map=db_room_map,
            faculty_payload_map=faculty_payload_map,
            room_payload_map=room_payload_map,
            elective_pairs=elective_pairs,
            db=db,
        )
        if resolved_payload is not None:
            return resolved_payload, message

    target_slot_id = conflict.affected_slots[-1] if conflict.affected_slots else None
    if target_slot_id is None or target_slot_id not in slots_by_id:
        return None, "Conflict has no actionable slot reference."

    slot = slots_by_id[target_slot_id]
    start = parse_time_to_minutes(slot.startTime)
    end = parse_time_to_minutes(slot.endTime)
    duration = end - start

    # 1) Prefer non-temporal changes first.
    if conflict.type in {"room-overlap", "capacity", "availability"}:
        replacement_room = _find_room_candidate(
            payload=payload,
            slot=slot,
            course_map=course_map,
            db_room_map=db_room_map,
            elective_pairs=elective_pairs,
            day=slot.day,
            start=start,
            end=end,
            current_faculty_id=slot.facultyId,
            moving_assistant_ids=_slot_assistant_faculty_ids(slot),
        )
        if replacement_room:
            slot.roomId = replacement_room
            return payload, "Resolved by assigning an alternate compatible room."

    if conflict.type in {"faculty-overlap", "availability", "course-faculty-inconsistency"}:
        replacement_faculty = _find_faculty_candidate(
            payload=payload,
            slot=slot,
            course_map=course_map,
            db_faculty_map=db_faculty_map,
            elective_pairs=elective_pairs,
            day=slot.day,
            start=start,
            end=end,
            current_room_id=slot.roomId,
            moving_assistant_ids=_slot_assistant_faculty_ids(slot),
        )
        if replacement_faculty:
            slot.facultyId = replacement_faculty
            _prune_primary_from_slot_assistants(slot, replacement_faculty)
            return payload, "Resolved by assigning an available faculty substitute."

    # 2) If still unresolved, search relocation candidates and select the least disruptive valid placement.
    candidate_blocks = _build_time_block_candidates(payload=payload, db=db, duration_minutes=duration)
    candidate_blocks.sort(
        key=lambda item: (
            0 if item[0] == slot.day else 1,
            abs(_day_sort_index(item[0]) - _day_sort_index(slot.day)),
            abs(item[1] - start),
            _day_sort_index(item[0]),
            item[1],
        )
    )
    best_placement: tuple[tuple[int, int, int, int, int, int], str, int, int, str, str] | None = None

    for day, candidate_start, candidate_end in candidate_blocks:
        if day == slot.day and candidate_start == start:
            continue

        room_candidates: list[str] = []
        if _room_available_for_window(
            room_record=db_room_map.get(slot.roomId),
            day=day,
            start=candidate_start,
            end=candidate_end,
        ):
            _append_unique_identifier(room_candidates, slot.roomId)
        _append_unique_identifier(
            room_candidates,
            _find_room_candidate(
                payload=payload,
                slot=slot,
                course_map=course_map,
                db_room_map=db_room_map,
                elective_pairs=elective_pairs,
                day=day,
                start=candidate_start,
                end=candidate_end,
                current_faculty_id=slot.facultyId,
            ),
        )

        faculty_candidates: list[str] = []
        current_faculty_payload = faculty_payload_map.get(slot.facultyId)
        if current_faculty_payload is not None and _faculty_available_for_window(
            faculty_payload=current_faculty_payload,
            faculty_record=db_faculty_map.get(slot.facultyId),
            day=day,
            start=candidate_start,
            end=candidate_end,
        ):
            _append_unique_identifier(faculty_candidates, slot.facultyId)
        _append_unique_identifier(
            faculty_candidates,
            _find_faculty_candidate(
                payload=payload,
                slot=slot,
                course_map=course_map,
                db_faculty_map=db_faculty_map,
                elective_pairs=elective_pairs,
                day=day,
                start=candidate_start,
                end=candidate_end,
                current_room_id=slot.roomId,
            ),
        )

        for candidate_room_id in list(room_candidates):
            _append_unique_identifier(
                faculty_candidates,
                _find_faculty_candidate(
                    payload=payload,
                    slot=slot,
                    course_map=course_map,
                    db_faculty_map=db_faculty_map,
                    elective_pairs=elective_pairs,
                    day=day,
                    start=candidate_start,
                    end=candidate_end,
                    current_room_id=candidate_room_id,
                ),
            )

        for candidate_faculty_id in list(faculty_candidates):
            _append_unique_identifier(
                room_candidates,
                _find_room_candidate(
                    payload=payload,
                    slot=slot,
                    course_map=course_map,
                    db_room_map=db_room_map,
                    elective_pairs=elective_pairs,
                    day=day,
                    start=candidate_start,
                    end=candidate_end,
                    current_faculty_id=candidate_faculty_id,
                ),
            )

        for candidate_room_id in room_candidates[:6]:
            room_record = db_room_map.get(candidate_room_id)
            if not _room_available_for_window(
                room_record=room_record,
                day=day,
                start=candidate_start,
                end=candidate_end,
            ):
                continue
            room_payload = room_payload_map.get(candidate_room_id)
            capacity_delta = (
                max(0, int((room_payload.capacity if room_payload is not None else 0) - (slot.studentCount or 0)))
                if slot.studentCount is not None
                else 0
            )
            for candidate_faculty_id in faculty_candidates[:6]:
                faculty_payload = faculty_payload_map.get(candidate_faculty_id)
                if faculty_payload is None:
                    continue
                if not _faculty_available_for_window(
                    faculty_payload=faculty_payload,
                    faculty_record=db_faculty_map.get(candidate_faculty_id),
                    day=day,
                    start=candidate_start,
                    end=candidate_end,
                ):
                    continue

                if _resource_placement_conflicts(
                    payload=payload,
                    slot_id=slot.id,
                    course_id=slot.courseId,
                    section=slot.section,
                    batch=slot.batch,
                    day=day,
                    start=candidate_start,
                    end=candidate_end,
                    room_id=candidate_room_id,
                    faculty_id=candidate_faculty_id,
                    course_map=course_map,
                    elective_pairs=elective_pairs,
                    moving_assistant_ids=_slot_assistant_faculty_ids(slot),
                ):
                    continue

                placement_score = (
                    0 if day == slot.day else 1,
                    abs(_day_sort_index(day) - _day_sort_index(slot.day)),
                    abs(candidate_start - start),
                    0 if candidate_room_id == slot.roomId else 1,
                    0 if candidate_faculty_id == slot.facultyId else 1,
                    capacity_delta,
                )
                if best_placement is None or placement_score < best_placement[0]:
                    best_placement = (
                        placement_score,
                        day,
                        candidate_start,
                        candidate_end,
                        candidate_room_id,
                        candidate_faculty_id,
                    )

    if best_placement is not None:
        _, day, candidate_start, candidate_end, candidate_room_id, candidate_faculty_id = best_placement
        moved = day != slot.day or candidate_start != start
        room_changed = candidate_room_id != slot.roomId
        faculty_changed = candidate_faculty_id != slot.facultyId
        slot.day = day
        slot.startTime = f"{candidate_start // 60:02d}:{candidate_start % 60:02d}"
        slot.endTime = f"{candidate_end // 60:02d}:{candidate_end % 60:02d}"
        slot.roomId = candidate_room_id
        slot.facultyId = candidate_faculty_id
        _prune_primary_from_slot_assistants(slot, candidate_faculty_id)
        if moved and room_changed and faculty_changed:
            return payload, "Resolved by relocating the slot and reassigning room/faculty."
        if moved and room_changed:
            return payload, "Resolved by moving the slot and assigning a compatible room."
        if moved and faculty_changed:
            return payload, "Resolved by moving the slot and assigning an available faculty member."
        if moved:
            return payload, "Resolved by moving the slot to a conflict-free teaching block."
        if room_changed:
            return payload, "Resolved by assigning an alternate compatible room."
        if faculty_changed:
            return payload, "Resolved by assigning an available faculty substitute."
        return payload, "Resolved by applying a validated slot re-placement."

    return None, "No safe automatic resolution found; apply the recommendation manually."


def _load_conflict_decision_map(db: Session) -> dict[str, TimetableConflictDecision]:
    rows = db.execute(select(TimetableConflictDecision)).scalars().all()
    return {item.conflict_id: item for item in rows}


def _decision_snapshot_to_conflict(snapshot: dict) -> TimetableConflict | None:
    try:
        return TimetableConflict(
            id=str(snapshot.get("id", "")),
            type=str(snapshot.get("type", "availability")),
            severity=str(snapshot.get("severity", "low")),
            description=str(snapshot.get("description", "Resolved conflict")),
            affectedSlots=list(snapshot.get("affectedSlots", [])),
            resolution=str(snapshot.get("resolution", "Resolved")),
            resolved=True,
        )
    except Exception:
        return None


def _decision_is_auto_resolved(decision: TimetableConflictDecision) -> bool:
    note = (decision.note or "").strip().lower()
    return note.startswith("[auto-resolved")


def _apply_decision_metadata(
    *,
    conflict: TimetableConflict,
    decision: TimetableConflictDecision | None,
) -> None:
    if decision is None:
        if not conflict.resolved:
            conflict.resolution_mode = "pending"
        return

    conflict.decision = decision.decision.value
    conflict.decision_note = decision.note
    if decision.decision == ConflictDecision.no:
        conflict.resolution_mode = "ignored"
        return

    if decision.resolved:
        conflict.resolved = True
        conflict.resolution_mode = "auto" if _decision_is_auto_resolved(decision) else "manual"
        return

    conflict.resolution_mode = "pending"


def _merge_conflicts_with_decisions(
    *,
    conflicts: list[TimetableConflict],
    decisions: dict[str, TimetableConflictDecision],
) -> list[TimetableConflict]:
    merged: list[TimetableConflict] = []
    existing_ids: set[str] = set()
    for conflict in conflicts:
        decision = decisions.get(conflict.id)
        _apply_decision_metadata(conflict=conflict, decision=decision)
        merged.append(conflict)
        existing_ids.add(conflict.id)

    for decision in decisions.values():
        if decision.decision != ConflictDecision.yes or not decision.resolved:
            continue
        if decision.conflict_id in existing_ids:
            continue
        recovered = _decision_snapshot_to_conflict(decision.conflict_snapshot or {})
        if recovered is not None:
            _apply_decision_metadata(conflict=recovered, decision=decision)
            merged.append(recovered)

    return merged


def _categorize_conflicts_for_review(
    conflicts: list[TimetableConflict],
) -> tuple[list[TimetableConflict], list[TimetableConflict], list[TimetableConflict], list[TimetableConflict]]:
    auto_resolved: list[TimetableConflict] = []
    manually_resolved: list[TimetableConflict] = []
    ignored: list[TimetableConflict] = []
    pending: list[TimetableConflict] = []

    for conflict in conflicts:
        mode = (conflict.resolution_mode or "").strip().lower()
        if mode == "ignored":
            ignored.append(conflict)
            continue
        if conflict.resolved:
            if mode == "auto":
                auto_resolved.append(conflict)
            else:
                manually_resolved.append(conflict)
            continue
        pending.append(conflict)

    def _sort_key(item: TimetableConflict) -> tuple[int, str]:
        severity_rank = {"high": 0, "medium": 1, "low": 2}
        return (severity_rank.get(item.severity, 3), item.type)

    auto_resolved.sort(key=_sort_key)
    manually_resolved.sort(key=_sort_key)
    ignored.sort(key=_sort_key)
    pending.sort(key=_sort_key)
    return auto_resolved, manually_resolved, ignored, pending


def _conflict_resolution_priority(conflict: TimetableConflict) -> tuple[int, int, int, str]:
    severity_rank = {"high": 0, "medium": 1, "low": 2}
    type_rank = {
        "section-overlap": 0,
        "faculty-overlap": 1,
        "room-overlap": 2,
        "elective-overlap": 3,
        "capacity": 4,
        "availability": 5,
        "course-faculty-inconsistency": 6,
    }
    return (
        severity_rank.get(conflict.severity, 3),
        type_rank.get(conflict.type, 99),
        -len(conflict.affected_slots),
        conflict.id,
    )


def _payload_conflict_resolution_signature(payload: OfficialTimetablePayload) -> tuple[tuple[str, str, str, str, str, str, str], ...]:
    return tuple(
        sorted(
            (
                slot.id,
                slot.day,
                slot.startTime,
                slot.endTime,
                slot.roomId,
                slot.facultyId,
                ",".join(sorted(_slot_assistant_faculty_ids(slot))),
            )
            for slot in payload.timetable_data
        )
    )


def _upsert_auto_resolved_decision(
    *,
    db: Session,
    conflict: TimetableConflict,
    current_user: User,
    resolution_message: str,
    note: str,
) -> None:
    decision = db.execute(
        select(TimetableConflictDecision).where(TimetableConflictDecision.conflict_id == conflict.id)
    ).scalar_one_or_none()
    if decision is None:
        decision = TimetableConflictDecision(
            conflict_id=conflict.id,
            decision=ConflictDecision.yes,
            resolved=True,
        )
        db.add(decision)

    snapshot = conflict.model_dump(by_alias=True)
    snapshot["resolved"] = True
    snapshot["resolution"] = resolution_message

    decision.decision = ConflictDecision.yes
    decision.resolved = True
    decision.note = note
    decision.decided_by_id = current_user.id
    decision.conflict_snapshot = snapshot


def _collect_constraint_mismatches(payload: OfficialTimetablePayload, db: Session) -> list[str]:
    messages: list[str] = []
    seen: set[str] = set()

    def add_message(raw: str | None) -> None:
        text = str(raw or "").strip()
        if not text or text in seen:
            return
        seen.add(text)
        messages.append(text)

    def capture(action: Callable[[], None]) -> None:
        try:
            action()
        except HTTPException as exc:
            detail = exc.detail
            if isinstance(detail, list):
                for item in detail:
                    add_message(str(item))
            else:
                add_message(str(detail))

    course_by_id = {course.id: course for course in payload.course_data}
    faculty_by_id = {faculty.id: faculty for faculty in payload.faculty_data}
    room_by_id = {room.id: room for room in payload.room_data}
    shared_groups: list[tuple[str, str, set[str]]] = []
    shared_groups_by_course: dict[str, list[set[str]]] = {}
    if payload.program_id and payload.term_number is not None:
        shared_groups = load_shared_lecture_groups(
            db=db,
            program_id=payload.program_id,
            term_number=payload.term_number,
        )
        shared_groups_by_course = build_shared_group_lookup(shared_groups)

    working_hours = load_working_hours(db)
    schedule_policy = load_schedule_policy(db)
    program_constraint = load_program_constraint(db, payload.program_id) if payload.program_id else None
    program_daily_slots = normalize_program_daily_slots(
        program_constraint.daily_time_slots if program_constraint is not None else None
    )
    period_minutes = schedule_policy.period_minutes
    day_segments: dict[str, list[tuple[int, int]]] = {}
    day_blocked_segments: dict[str, list[tuple[int, int, str]]] = {}
    for day, hours_entry in working_hours.items():
        if not hours_entry.enabled:
            continue
        if program_daily_slots:
            teaching_segments, blocked_segments = build_teaching_segments_from_program_slots(program_daily_slots)
            if teaching_segments:
                day_segments[day] = teaching_segments
            if blocked_segments:
                day_blocked_segments[day] = blocked_segments
            continue
        day_start = parse_time_to_minutes(hours_entry.start_time)
        day_end = parse_time_to_minutes(hours_entry.end_time)
        day_segments[day] = build_teaching_segments(
            day_start=day_start,
            day_end=day_end,
            period_minutes=period_minutes,
            breaks=schedule_policy.breaks,
        )
        day_blocked_segments[day] = [
            (
                parse_time_to_minutes(item.start_time),
                parse_time_to_minutes(item.end_time),
                item.name,
            )
            for item in schedule_policy.breaks
        ]

    for slot in payload.timetable_data:
        hours_entry = working_hours.get(slot.day)
        segments = day_segments.get(slot.day, [])
        slot_start = parse_time_to_minutes(slot.startTime)
        slot_end = parse_time_to_minutes(slot.endTime)
        if hours_entry is None or not hours_entry.enabled or not segments:
            add_message(f"Timeslot {slot.id} occurs on a non-working day ({slot.day}).")
            continue
        allowed_start = min(start for start, _end in segments)
        allowed_end = max(end for _start, end in segments)
        if slot_start < allowed_start or slot_end > allowed_end:
            add_message(
                (
                    f"Timeslot {slot.id} on {slot.day} must be within working hours "
                    f"{hours_entry.start_time}-{hours_entry.end_time}."
                )
            )
        slot_duration = slot_end - slot_start
        if slot_duration % period_minutes != 0:
            add_message(f"Timeslot {slot.id} must be a multiple of {period_minutes} minutes.")
        if not is_slot_aligned_with_segments(slot_start, slot_end, segments):
            add_message(f"Timeslot {slot.id} must align to configured teaching slot boundaries.")
        blocked_overlap = next(
            (
                (start, end, label)
                for start, end, label in day_blocked_segments.get(slot.day, [])
                if slot_start < end and slot_end > start
            ),
            None,
        )
        if blocked_overlap is not None:
            add_message(
                (
                    f"Timeslot {slot.id} overlaps non-teaching slot '{blocked_overlap[2]}' "
                    f"({_minutes_to_time(blocked_overlap[0])}-{_minutes_to_time(blocked_overlap[1])})."
                )
            )

    if payload.term_number is None:
        has_constraints = db.execute(select(SemesterConstraint.id)).first() is not None
        if has_constraints:
            add_message("termNumber is required to validate semester constraints.")
    else:
        constraint = load_semester_constraint(db, payload.term_number)
        if constraint is not None:
            capture(lambda: enforce_semester_constraints(payload, constraint, force=False))

    capture(lambda: enforce_resource_conflicts(payload, course_by_id, shared_groups_by_course, force=False))
    capture(lambda: enforce_course_scheduling(payload, course_by_id, room_by_id, schedule_policy, force=False))
    student_counts_by_slot: dict[str, int] = {}
    try:
        student_counts_by_slot = enforce_room_capacity(payload, room_by_id, db, force=False)
    except HTTPException as exc:
        detail = exc.detail
        if isinstance(detail, list):
            for item in detail:
                add_message(str(item))
        else:
            add_message(str(detail))
    capture(
        lambda: enforce_shared_lecture_constraints(
            payload,
            shared_groups,
            shared_groups_by_course,
            room_by_id,
            student_counts_by_slot,
            force=False,
        )
    )
    capture(lambda: enforce_section_credit_aligned_minutes(payload, db, schedule_policy, force=False))
    capture(lambda: enforce_program_credit_requirements(payload, course_by_id, db, force=False))
    capture(lambda: enforce_elective_overlap_constraints(payload, db, force=False))
    capture(lambda: enforce_prerequisite_constraints(payload, db, force=False))
    capture(lambda: enforce_faculty_overload_preferences(payload, db, force=False))
    capture(lambda: enforce_single_faculty_per_course_sections(payload, course_by_id, faculty_by_id, force=False))

    live_conflicts = _build_conflicts(payload, db)
    metrics = _build_constraint_metrics(payload, live_conflicts)
    for status_item in metrics:
        if status_item.status == "satisfied":
            continue
        add_message(
            f"{status_item.name}: {status_item.description} (satisfaction {status_item.satisfaction:.1f}%)."
        )

    return messages


def _status_from_score(score: float) -> str:
    if score >= 95:
        return "satisfied"
    if score >= 70:
        return "partial"
    return "violated"


def _build_constraint_metrics(payload: OfficialTimetablePayload, conflicts: list[TimetableConflict]) -> list[ConstraintStatus]:
    total_slots = max(1, len(payload.timetable_data))
    by_type: dict[str, int] = defaultdict(int)
    for conflict in conflicts:
        by_type[conflict.type] += 1

    availability_score = max(0.0, 100.0 - (by_type.get("availability", 0) * 100.0 / total_slots))
    capacity_score = max(0.0, 100.0 - (by_type.get("capacity", 0) * 100.0 / total_slots))
    overlap_score = max(
        0.0,
        100.0
        - (
            (
                by_type.get("faculty-overlap", 0)
                + by_type.get("room-overlap", 0)
                + by_type.get("section-overlap", 0)
                + by_type.get("elective-overlap", 0)
                + by_type.get("course-faculty-inconsistency", 0)
            )
            * 100.0
            / total_slots
        ),
    )

    course_by_id = {course.id: course for course in payload.course_data}
    lab_slots = [
        slot
        for slot in payload.timetable_data
        if _slot_is_practical(slot, course_by_id.get(slot.courseId))
    ]
    lab_groups: dict[tuple[str, str, str], list] = defaultdict(list)
    for slot in lab_slots:
        batch = slot.batch or "default"
        lab_groups[(slot.courseId, slot.section, batch)].append(slot)
    lab_violations = 0
    for slots in lab_groups.values():
        by_day: dict[str, list] = defaultdict(list)
        for slot in slots:
            by_day[slot.day].append(slot)
        for day_slots in by_day.values():
            if len(day_slots) <= 1:
                continue
            ordered = sorted(day_slots, key=lambda item: parse_time_to_minutes(item.startTime))
            for left, right in zip(ordered, ordered[1:]):
                if parse_time_to_minutes(left.endTime) != parse_time_to_minutes(right.startTime):
                    lab_violations += 1
    lab_score = max(0.0, 100.0 - (lab_violations * 100.0 / max(1, len(lab_slots))))

    faculty_minutes: dict[str, int] = defaultdict(int)
    faculty_max: dict[str, int] = {}
    for faculty in payload.faculty_data:
        if _is_virtual_faculty_id(faculty.id):
            continue
        faculty_max[faculty.id] = faculty.maxHours * 60
    for slot in payload.timetable_data:
        duration = parse_time_to_minutes(slot.endTime) - parse_time_to_minutes(slot.startTime)
        for faculty_id in _slot_all_faculty_ids(slot):
            if _is_virtual_faculty_id(faculty_id):
                continue
            faculty_minutes[faculty_id] += duration

    workload_hours = [minutes / 60.0 for minutes in faculty_minutes.values()]
    if workload_hours:
        average = sum(workload_hours) / len(workload_hours)
        std_dev = sqrt(sum((value - average) ** 2 for value in workload_hours) / len(workload_hours))
    else:
        std_dev = 0.0
    overload_penalty = 0.0
    for faculty_id, minutes in faculty_minutes.items():
        max_minutes = faculty_max.get(faculty_id, 0)
        if max_minutes and minutes > max_minutes:
            overload_penalty += (minutes - max_minutes) / 60.0
    workload_score = max(0.0, 100.0 - (std_dev * 10.0) - (overload_penalty * 4.0))

    metrics = [
        ConstraintStatus(
            name="Faculty Availability",
            description="Sessions are assigned within faculty availability settings.",
            satisfaction=round(availability_score, 1),
            status=_status_from_score(availability_score),
        ),
        ConstraintStatus(
            name="Room Capacity",
            description="Room assignments satisfy expected student capacity.",
            satisfaction=round(capacity_score, 1),
            status=_status_from_score(capacity_score),
        ),
        ConstraintStatus(
            name="Conflict-Free Allocation",
            description="No faculty, room, or section overlaps exist in the timetable.",
            satisfaction=round(overlap_score, 1),
            status=_status_from_score(overlap_score),
        ),
        ConstraintStatus(
            name="Lab Continuity",
            description="Lab sessions remain contiguous and unsplit in scheduled blocks.",
            satisfaction=round(lab_score, 1),
            status=_status_from_score(lab_score),
        ),
        ConstraintStatus(
            name="Workload Balance",
            description="Faculty workload remains balanced and under configured limits.",
            satisfaction=round(workload_score, 1),
            status=_status_from_score(workload_score),
        ),
    ]
    return metrics


def _build_workload_chart(payload: OfficialTimetablePayload) -> list[WorkloadChartEntry]:
    faculty_minutes: dict[str, int] = defaultdict(int)
    for slot in payload.timetable_data:
        duration = parse_time_to_minutes(slot.endTime) - parse_time_to_minutes(slot.startTime)
        for faculty_id in _slot_all_faculty_ids(slot):
            if _is_virtual_faculty_id(faculty_id):
                continue
            faculty_minutes[faculty_id] += duration

    entries: list[WorkloadChartEntry] = []
    for faculty in payload.faculty_data:
        if _is_virtual_faculty_id(faculty.id):
            continue
        assigned_hours = faculty_minutes.get(faculty.id, 0) / 60.0
        short_name = faculty.name.split(" ")[-1] if faculty.name else faculty.id
        entries.append(
            WorkloadChartEntry(
                id=faculty.id,
                name=short_name,
                fullName=faculty.name,
                department=faculty.department,
                workload=round(assigned_hours, 2),
                max=float(faculty.maxHours),
                overloaded=assigned_hours > faculty.maxHours,
            )
        )
    return entries


def _build_daily_workload(payload: OfficialTimetablePayload) -> list[DailyWorkloadEntry]:
    day_faculty_minutes: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for slot in payload.timetable_data:
        duration = parse_time_to_minutes(slot.endTime) - parse_time_to_minutes(slot.startTime)
        for faculty_id in _slot_all_faculty_ids(slot):
            if _is_virtual_faculty_id(faculty_id):
                continue
            day_faculty_minutes[slot.day][faculty_id] += duration

    ordered_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    daily: list[DailyWorkloadEntry] = []
    for day in ordered_days:
        faculty_map = day_faculty_minutes.get(day)
        if not faculty_map:
            continue
        loads = {faculty_id: round(minutes / 60.0, 2) for faculty_id, minutes in faculty_map.items()}
        total = round(sum(loads.values()), 2)
        daily.append(DailyWorkloadEntry(day=day, loads=loads, total=total))
    return daily


def _semester_label(payload: OfficialTimetablePayload) -> str:
    if payload.term_number is None:
        return "Current"
    return f"Term {payload.term_number}"


def _build_analytics(payload: OfficialTimetablePayload, conflicts: list[TimetableConflict], db: Session) -> TimetableAnalytics:
    constraint_data = _build_constraint_metrics(payload, conflicts)
    workload_chart_data = _build_workload_chart(payload)
    daily_workload_data = _build_daily_workload(payload)
    overall_satisfaction = round(
        sum(item.satisfaction for item in constraint_data) / max(1, len(constraint_data)),
        1,
    )

    generation_settings = db.get(TimetableGenerationSettings, 1)
    total_iterations = 0
    if generation_settings is not None:
        total_iterations = generation_settings.population_size * generation_settings.generations
    compute_time = "N/A"
    if total_iterations:
        # Coarse estimate for dashboard visibility only.
        compute_time = f"~{max(1, total_iterations // 2500)} sec"

    record = db.get(OfficialTimetable, 1)
    last_generated = record.updated_at.isoformat() if record is not None and record.updated_at is not None else None

    optimization_summary = OptimizationSummary(
        constraintSatisfaction=overall_satisfaction,
        conflictsDetected=len(conflicts),
        optimizationTechnique="Evolutionary Algorithm",
        alternativesGenerated=1,
        lastGenerated=last_generated,
        totalIterations=total_iterations,
        computeTime=compute_time,
    )

    performance_trend_data = [
        PerformanceTrendEntry(
            semester=_semester_label(payload),
            satisfaction=overall_satisfaction,
            conflicts=len(conflicts),
        )
    ]

    return TimetableAnalytics(
        optimizationSummary=optimization_summary,
        constraintData=constraint_data,
        workloadChartData=workload_chart_data,
        dailyWorkloadData=daily_workload_data,
        performanceTrendData=performance_trend_data,
    )


def _load_official_payload(db: Session) -> OfficialTimetablePayload:
    record = db.get(OfficialTimetable, 1)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Official timetable not found")
    return OfficialTimetablePayload.model_validate(record.payload)


def _slice_payload_by_slots(
    payload: OfficialTimetablePayload,
    slots: list[object],
) -> OfficialTimetablePayload:
    selected_slots = list(slots)
    course_ids = {slot.courseId for slot in selected_slots}
    room_ids = {slot.roomId for slot in selected_slots}
    faculty_ids: set[str] = set()
    for slot in selected_slots:
        faculty_ids.update(_slot_all_faculty_ids(slot))

    return OfficialTimetablePayload.model_validate(
        {
            "programId": payload.program_id,
            "termNumber": payload.term_number,
            "facultyData": [
                item.model_dump(by_alias=True)
                for item in payload.faculty_data
                if item.id in faculty_ids
            ],
            "courseData": [
                item.model_dump(by_alias=True)
                for item in payload.course_data
                if item.id in course_ids
            ],
            "roomData": [
                item.model_dump(by_alias=True)
                for item in payload.room_data
                if item.id in room_ids
            ],
            "timetableData": [slot.model_dump(by_alias=True) for slot in selected_slots],
        }
    )


def _scope_official_payload_for_user(
    payload: OfficialTimetablePayload,
    user: User,
    db: Session,
) -> OfficialTimetablePayload:
    if user.role in {UserRole.admin, UserRole.scheduler}:
        return payload

    if user.role == UserRole.student:
        section = (user.section_name or "").strip().upper()
        if not section:
            return _slice_payload_by_slots(payload, [])
        return _slice_payload_by_slots(
            payload,
            [slot for slot in payload.timetable_data if slot.section.strip().upper() == section],
        )

    if user.role == UserRole.faculty:
        user_email = (user.email or "").strip().lower()
        faculty_ids = {
            item.id
            for item in payload.faculty_data
            if (item.email or "").strip().lower() == user_email
        }
        if not faculty_ids and user_email:
            faculty_match = (
                db.execute(
                    select(Faculty.id).where(func.lower(Faculty.email) == user_email)
                )
                .scalars()
                .first()
            )
            if faculty_match:
                faculty_ids.add(faculty_match)

        if not faculty_ids:
            return _slice_payload_by_slots(payload, [])

        return _slice_payload_by_slots(
            payload,
            [
                slot
                for slot in payload.timetable_data
                if any(faculty_id in faculty_ids for faculty_id in _slot_all_faculty_ids(slot))
            ],
        )

    return _slice_payload_by_slots(payload, [])




ORDERED_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DAY_ORDER = {day: index for index, day in enumerate(ORDERED_DAYS)}


def _matches_offline_filters(payload: OfficialTimetablePayload, filters: OfflinePublishFilters | None) -> bool:
    if filters is None:
        return True
    if filters.program_id:
        if payload.program_id is None:
            return False
        if filters.program_id != payload.program_id:
            return False
    if filters.term_number is not None:
        if payload.term_number is None:
            return False
        if filters.term_number != payload.term_number:
            return False
    return True


def _filter_payload_for_offline_publish(
    payload: OfficialTimetablePayload,
    filters: OfflinePublishFilters | None,
) -> OfficialTimetablePayload:
    if filters is None:
        return payload
    if not _matches_offline_filters(payload, filters):
        return _slice_payload_by_slots(payload, [])

    department = filters.department.strip().lower() if filters.department else None
    section_name = filters.section_name.strip().upper() if filters.section_name else None
    faculty_id = filters.faculty_id
    faculty_department = {
        item.id: item.department.strip().lower()
        for item in payload.faculty_data
        if item.department and item.department.strip()
    }

    scoped_slots: list[object] = []
    for slot in payload.timetable_data:
        if section_name and slot.section.strip().upper() != section_name:
            continue
        slot_faculty_ids = _slot_all_faculty_ids(slot)
        if faculty_id and faculty_id not in slot_faculty_ids:
            continue
        if department:
            if not any(faculty_department.get(item_id) == department for item_id in slot_faculty_ids):
                continue
        scoped_slots.append(slot)
    return _slice_payload_by_slots(payload, scoped_slots)


def _sort_timetable_slots(slots: list[object]) -> list[object]:
    return sorted(
        slots,
        key=lambda slot: (
            _day_sort_index(slot.day),
            parse_time_to_minutes(slot.startTime),
            slot.section,
            slot.batch or "",
            slot.id,
        ),
    )


def _build_timetable_email_content(
    *,
    user: User,
    payload: OfficialTimetablePayload,
    scope_label: str,
) -> tuple[str, str]:
    sorted_slots = _sort_timetable_slots(payload.timetable_data)
    course_by_id = {item.id: item for item in payload.course_data}
    room_by_id = {item.id: item for item in payload.room_data}
    faculty_by_id = {item.id: item for item in payload.faculty_data}

    lines: list[str] = [
        f"Hello {user.name},",
        "",
        f"Your ShedForge timetable is attached below ({scope_label}).",
        "",
    ]
    row_html: list[str] = []
    for slot in sorted_slots:
        course = course_by_id.get(slot.courseId)
        room = room_by_id.get(slot.roomId)
        faculty = faculty_by_id.get(slot.facultyId)
        assistant_labels = [
            faculty_by_id.get(assistant_id).name if faculty_by_id.get(assistant_id) is not None else assistant_id
            for assistant_id in _slot_assistant_faculty_ids(slot)
        ]
        assistant_text = f" | Assist: {', '.join(assistant_labels)}" if assistant_labels else ""
        assistant_suffix = f" (Assist: {', '.join(assistant_labels)})" if assistant_labels else ""
        batch = f" Batch {slot.batch}" if slot.batch else ""
        lines.append(
            (
                f"- {slot.day} {slot.startTime}-{slot.endTime} | "
                f"{course.code if course else slot.courseId} {course.name if course else ''} | "
                f"Section {slot.section}{batch} | "
                f"Room {room.name if room else slot.roomId} | "
                f"Faculty {faculty.name if faculty else slot.facultyId}"
                f"{assistant_text}"
            )
        )
        row_html.append(
            "<tr>"
            f"<td>{escape(slot.day)}</td>"
            f"<td>{escape(slot.startTime)} - {escape(slot.endTime)}</td>"
            f"<td>{escape(course.code if course else slot.courseId)}</td>"
            f"<td>{escape(course.name if course else '')}</td>"
            f"<td>{escape(slot.section)}</td>"
            f"<td>{escape(slot.batch or '')}</td>"
            f"<td>{escape(room.name if room else slot.roomId)}</td>"
            f"<td>{escape(faculty.name if faculty else slot.facultyId)}"
            f"{escape(assistant_suffix)}</td>"
            "</tr>"
        )

    if not sorted_slots:
        lines.append("- No timetable entries found for your profile in this scope.")

    lines.extend(
        [
            "",
            "Regards,",
            "ShedForge Scheduler",
        ]
    )

    html_content = (
        "<html><body>"
        f"<p>Hello {escape(user.name)},</p>"
        f"<p>Your ShedForge timetable is available for <strong>{escape(scope_label)}</strong>.</p>"
        "<table border='1' cellpadding='6' cellspacing='0' style='border-collapse:collapse;font-family:Arial,sans-serif;font-size:13px;'>"
        "<thead><tr>"
        "<th>Day</th><th>Time</th><th>Course Code</th><th>Course</th>"
        "<th>Section</th><th>Batch</th><th>Room</th><th>Faculty</th>"
        "</tr></thead>"
        "<tbody>"
        + ("".join(row_html) if row_html else "<tr><td colspan='8'>No timetable entries in this scope.</td></tr>")
        + "</tbody></table>"
        "<p>Regards,<br/>ShedForge Scheduler</p>"
        "</body></html>"
    )

    return "\n".join(lines), html_content


def _send_offline_timetable_emails(
    *,
    db: Session,
    payload: OfficialTimetablePayload,
    filters: OfflinePublishFilters | None,
) -> OfflinePublishResponse:
    filtered_payload = _filter_payload_for_offline_publish(payload, filters)
    users = list(
        db.execute(
            select(User).where(
                User.is_active.is_(True),
                User.role.in_([UserRole.faculty, UserRole.student]),
            )
        ).scalars()
    )

    scope_label_parts: list[str] = []
    if filters and filters.term_number is not None:
        scope_label_parts.append(f"Semester {filters.term_number}")
    elif filtered_payload.term_number is not None:
        scope_label_parts.append(f"Semester {filtered_payload.term_number}")
    if filters and filters.section_name:
        scope_label_parts.append(f"Section {filters.section_name}")
    if filters and filters.department:
        scope_label_parts.append(filters.department)
    if filters and filters.faculty_id:
        scope_label_parts.append("Faculty-specific")
    scope_label = ", ".join(scope_label_parts) if scope_label_parts else "Current timetable"

    attempted = 0
    sent = 0
    skipped = 0
    failed = 0
    recipients: list[str] = []
    failed_recipients: list[str] = []

    for user in users:
        if not user.email:
            skipped += 1
            continue

        user_payload = _scope_official_payload_for_user(filtered_payload, user, db)
        if not user_payload.timetable_data:
            skipped += 1
            continue

        attempted += 1
        text_content, html_content = _build_timetable_email_content(
            user=user,
            payload=user_payload,
            scope_label=scope_label,
        )
        try:
            send_email(
                to_email=user.email,
                subject="ShedForge Timetable (Offline Copy)",
                text_content=text_content,
                html_content=html_content,
            )
            recipients.append(user.email)
            sent += 1
            create_notification(
                db,
                user_id=user.id,
                title="Timetable Sent by Email",
                message=f"Your timetable was emailed ({scope_label}).",
                notification_type=NotificationType.timetable,
                recipient=user,
                deliver_email=False,
            )
        except EmailDeliveryError:
            failed += 1
            failed_recipients.append(user.email)

    message = (
        f"Offline publish completed. Sent: {sent}, Failed: {failed}, Skipped: {skipped}."
    )
    return OfflinePublishResponse(
        attempted=attempted,
        sent=sent,
        skipped=skipped,
        failed=failed,
        recipients=recipients,
        failed_recipients=failed_recipients,
        message=message,
    )


ROOM_ONLY_CONFLICT_TYPES = {
    "room_conflict",
    "room-overlap",
    "room_capacity",
    "capacity",
    "room_type",
}


def _slot_has_room_overlap(slot: object, other: object) -> bool:
    if slot.day != other.day:
        return False
    start_a = parse_time_to_minutes(slot.startTime)
    end_a = parse_time_to_minutes(slot.endTime)
    start_b = parse_time_to_minutes(other.startTime)
    end_b = parse_time_to_minutes(other.endTime)
    return slots_overlap(start_a, end_a, start_b, end_b)


def _resolve_faculty_user_for_slot(payload: OfficialTimetablePayload, slot: object, db: Session) -> User | None:
    faculty_ids = _slot_all_faculty_ids(slot)
    faculty_map = {item.id: item for item in payload.faculty_data}
    for faculty_id in faculty_ids:
        faculty = faculty_map.get(faculty_id)
        if faculty is None or not faculty.email:
            continue
        matched = (
            db.execute(
                select(User).where(
                    func.lower(User.email) == faculty.email.strip().lower(),
                    User.role == UserRole.faculty,
                    User.is_active.is_(True),
                )
            )
            .scalars()
            .first()
        )
        if matched is not None:
            return matched
    return None


def _resolve_faculty_user_by_id(payload: OfficialTimetablePayload, faculty_id: str, db: Session) -> User | None:
    normalized_id = str(faculty_id or "").strip()
    if not normalized_id:
        return None

    faculty_email: str | None = None
    faculty_map = {item.id: item for item in payload.faculty_data}
    mapped = faculty_map.get(normalized_id)
    if mapped is not None and mapped.email:
        faculty_email = mapped.email.strip().lower()
    else:
        faculty_record = db.get(Faculty, normalized_id)
        if faculty_record is not None and faculty_record.email:
            faculty_email = faculty_record.email.strip().lower()

    if not faculty_email:
        return None

    return (
        db.execute(
            select(User).where(
                func.lower(User.email) == faculty_email,
                User.role == UserRole.faculty,
                User.is_active.is_(True),
            )
        )
        .scalars()
        .first()
    )


def _resolve_cr_student_for_section(
    *,
    db: Session,
    program_id: str | None,
    term_number: int | None,
    section_name: str,
) -> User | None:
    section = section_name.strip()
    if not section:
        return None

    statement = select(User).where(
        User.role == UserRole.student,
        User.is_active.is_(True),
        func.lower(User.section_name) == section.lower(),
    )
    if program_id:
        statement = statement.where(User.program_id == program_id)
    if term_number is not None:
        statement = statement.where(User.semester_number == term_number)

    candidates = list(
        db.execute(
            statement.order_by(
                User.roll_number.asc(),
                User.created_at.asc(),
            )
        ).scalars()
    )
    if candidates:
        return candidates[0]
    return None


def _apply_change_proposal_to_payload(
    payload: OfficialTimetablePayload,
    proposal: dict,
) -> tuple[OfficialTimetablePayload, object, list[str]]:
    proposed_slot_id = str(proposal.get("slotId") or "").strip()
    if not proposed_slot_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing slotId in proposal")
    day = str(proposal.get("day") or "").strip()
    start_time = str(proposal.get("startTime") or "").strip()
    end_time = str(proposal.get("endTime") or "").strip()
    room_id = str(proposal.get("roomId") or "").strip() or None
    faculty_id = str(proposal.get("facultyId") or "").strip() or None
    section_name = str(proposal.get("section") or "").strip() or None
    request_kind = str(proposal.get("requestKind") or "slot_move").strip().lower() or "slot_move"
    raw_assistants = proposal.get("assistantFacultyIds")
    assistant_faculty_ids: list[str] | None = None
    if isinstance(raw_assistants, list):
        seen_assistants: set[str] = set()
        normalized_assistants: list[str] = []
        for item in raw_assistants:
            assistant_id = str(item or "").strip()
            if not assistant_id or assistant_id in seen_assistants:
                continue
            seen_assistants.add(assistant_id)
            normalized_assistants.append(assistant_id)
        assistant_faculty_ids = normalized_assistants
    if not day or not start_time or not end_time:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Proposal requires day/startTime/endTime")

    updated = payload.model_copy(deep=True)
    target_slot = next((item for item in updated.timetable_data if item.id == proposed_slot_id), None)
    if target_slot is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Requested slot not found in official timetable")

    if request_kind == "extra_class":
        new_slot = target_slot.model_copy(deep=True)
        new_slot.id = str(uuid.uuid4())
        new_slot.day = day
        new_slot.startTime = start_time
        new_slot.endTime = end_time
        if room_id:
            new_slot.roomId = room_id
        if faculty_id:
            new_slot.facultyId = faculty_id
            _prune_primary_from_slot_assistants(new_slot, faculty_id)
        if assistant_faculty_ids is not None:
            new_slot.assistantFacultyIds = assistant_faculty_ids
        if section_name:
            new_slot.section = section_name
        updated.timetable_data.append(new_slot)
        return updated, new_slot, [target_slot.id, new_slot.id]

    target_slot.day = day
    target_slot.startTime = start_time
    target_slot.endTime = end_time
    if room_id:
        target_slot.roomId = room_id
    if faculty_id:
        target_slot.facultyId = faculty_id
        _prune_primary_from_slot_assistants(target_slot, faculty_id)
    if assistant_faculty_ids is not None:
        target_slot.assistantFacultyIds = assistant_faculty_ids
    if section_name:
        target_slot.section = section_name

    return updated, target_slot, [target_slot.id]


def _find_alternative_room_id_for_slot(payload: OfficialTimetablePayload, slot: object) -> str | None:
    room_by_id = {item.id: item for item in payload.room_data}
    current_room = room_by_id.get(slot.roomId)
    student_count = int(slot.studentCount or 0)

    candidates = sorted(
        payload.room_data,
        key=lambda room: (
            max(0, room.capacity - student_count),
            room.name.lower(),
        ),
    )

    for room in candidates:
        if room.id == slot.roomId:
            continue
        if current_room is not None and room.type != current_room.type:
            continue
        if student_count > 0 and room.capacity < student_count:
            continue
        has_overlap = any(
            other.id != slot.id
            and other.roomId == room.id
            and _slot_has_room_overlap(slot, other)
            for other in payload.timetable_data
        )
        if has_overlap:
            continue
        return room.id
    return None


def _send_timetable_distribution_emails(
    *,
    db: Session,
    payload: OfficialTimetablePayload,
) -> OfflinePublishResponse:
    users = list(
        db.execute(
            select(User).where(
                User.is_active.is_(True),
                User.role.in_([UserRole.admin, UserRole.scheduler, UserRole.faculty, UserRole.student]),
            )
        ).scalars()
    )

    attempted = 0
    sent = 0
    skipped = 0
    failed = 0
    recipients: list[str] = []
    failed_recipients: list[str] = []

    for user in users:
        if not user.email:
            skipped += 1
            continue

        if user.role in {UserRole.admin, UserRole.scheduler}:
            user_payload = payload
            scope_label = "Classroom Master Timetable"
            subject = "ShedForge Room Utilization Timetable"
        elif user.role == UserRole.faculty:
            user_payload = _scope_official_payload_for_user(payload, user, db)
            scope_label = "Faculty Timetable"
            subject = "ShedForge Faculty Timetable"
        else:
            user_payload = _scope_official_payload_for_user(payload, user, db)
            scope_label = "Class Timetable"
            subject = "ShedForge Class Timetable"

        if not user_payload.timetable_data:
            skipped += 1
            continue

        attempted += 1
        text_content, html_content = _build_timetable_email_content(
            user=user,
            payload=user_payload,
            scope_label=scope_label,
        )
        try:
            send_email(
                to_email=user.email,
                subject=subject,
                text_content=text_content,
                html_content=html_content,
            )
            sent += 1
            recipients.append(user.email)
            create_notification(
                db,
                user_id=user.id,
                title="Timetable Distribution",
                message=f"{scope_label} was distributed to your inbox.",
                notification_type=NotificationType.timetable,
                recipient=user,
                deliver_email=False,
            )
        except EmailDeliveryError:
            failed += 1
            failed_recipients.append(user.email)

    return OfflinePublishResponse(
        attempted=attempted,
        sent=sent,
        skipped=skipped,
        failed=failed,
        recipients=recipients,
        failed_recipients=failed_recipients,
        message=(
            "Role-wise distribution completed. "
            f"Sent: {sent}, Failed: {failed}, Skipped: {skipped}. "
            "Faculty received faculty timetables, students received class timetables, "
            "and admin office received room master timetable."
        ),
    )


@router.get("/official", response_model=OfficialTimetablePayload)
def get_official_timetable(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> OfficialTimetablePayload:
    """
    Retrieves the currently published 'Official' timetable.
    
    Data returned is scoped to the requesting user's role:
    - **Admins**: See the full timetable.
    - **Students**: See only their enrolled sections/courses.
    - **Faculty**: See only their assigned classes.
    """
    payload = _load_official_payload(db)
    return _scope_official_payload_for_user(payload, current_user, db)


@router.get("/official/full", response_model=OfficialTimetablePayload)
def get_official_timetable_full(
    current_user: User = Depends(
        require_roles(UserRole.admin, UserRole.scheduler, UserRole.faculty, UserRole.student)
    ),
    db: Session = Depends(get_db),
) -> OfficialTimetablePayload:
    """
    Returns the full official timetable payload to all authenticated roles.

    Used for collaborative timetable browsing where users need to inspect class/faculty/room views.
    """
    del current_user
    return _load_official_payload(db)


@router.get("/official/faculty-mapping", response_model=list[FacultyCourseSectionMappingOut])
def get_official_faculty_mapping(
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler, UserRole.faculty)),
    db: Session = Depends(get_db),
) -> list[FacultyCourseSectionMappingOut]:
    """
    Returns a mapping of faculty members to their assigned courses and sections.
    
    Used for generating faculty-specific view of the official timetable.
    """
    payload = _load_official_payload(db)
    if current_user.role == UserRole.faculty:
        scoped_payload = _scope_official_payload_for_user(payload, current_user, db)
    else:
        scoped_payload = payload

    faculty_by_id = {item.id: item for item in scoped_payload.faculty_data}
    course_by_id = {item.id: item for item in scoped_payload.course_data}
    room_by_id = {item.id: item for item in scoped_payload.room_data}
    allowed_faculty_ids: set[str] | None = None
    if current_user.role == UserRole.faculty:
        faculty_email = (current_user.email or "").strip().lower()
        allowed_faculty_ids = {
            item.id
            for item in scoped_payload.faculty_data
            if (item.email or "").strip().lower() == faculty_email
        }
        if not allowed_faculty_ids and faculty_email:
            faculty_match = (
                db.execute(select(Faculty.id).where(func.lower(Faculty.email) == faculty_email))
                .scalars()
                .first()
            )
            if faculty_match:
                allowed_faculty_ids.add(faculty_match)

    assignments_by_faculty: dict[str, list[FacultyCourseSectionAssignment]] = defaultdict(list)
    assigned_minutes_by_faculty: dict[str, int] = defaultdict(int)

    for slot in scoped_payload.timetable_data:
        faculty = faculty_by_id.get(slot.facultyId)
        course = course_by_id.get(slot.courseId)
        room = room_by_id.get(slot.roomId)
        if faculty is None or course is None or room is None:
            continue

        start_min = parse_time_to_minutes(slot.startTime)
        end_min = parse_time_to_minutes(slot.endTime)
        duration_minutes = max(0, end_min - start_min)
        if allowed_faculty_ids is None or slot.facultyId in allowed_faculty_ids:
            assigned_minutes_by_faculty[slot.facultyId] += duration_minutes
            assignments_by_faculty[slot.facultyId].append(
                FacultyCourseSectionAssignment(
                    course_id=course.id,
                    course_code=course.code,
                    course_name=course.name,
                    section=slot.section,
                    batch=slot.batch,
                    day=slot.day,
                    startTime=slot.startTime,
                    endTime=slot.endTime,
                    room_id=room.id,
                    room_name=room.name,
                    assignmentRole="primary",
                )
            )
        for assistant_faculty_id in _slot_assistant_faculty_ids(slot):
            assistant = faculty_by_id.get(assistant_faculty_id)
            if assistant is None:
                continue
            if allowed_faculty_ids is not None and assistant_faculty_id not in allowed_faculty_ids:
                continue
            assigned_minutes_by_faculty[assistant_faculty_id] += duration_minutes
            assignments_by_faculty[assistant_faculty_id].append(
                FacultyCourseSectionAssignment(
                    course_id=course.id,
                    course_code=course.code,
                    course_name=course.name,
                    section=slot.section,
                    batch=slot.batch,
                    day=slot.day,
                    startTime=slot.startTime,
                    endTime=slot.endTime,
                    room_id=room.id,
                    room_name=room.name,
                    assignmentRole="assistant",
                )
            )

    output: list[FacultyCourseSectionMappingOut] = []
    for faculty_id, assignments in assignments_by_faculty.items():
        faculty = faculty_by_id.get(faculty_id)
        if faculty is None:
            continue
        assignments.sort(
            key=lambda item: (
                _day_sort_index(item.day),
                parse_time_to_minutes(item.start_time),
                item.course_code,
                item.section,
                item.batch or "",
            )
        )
        output.append(
            FacultyCourseSectionMappingOut(
                faculty_id=faculty.id,
                faculty_name=faculty.name,
                faculty_email=faculty.email,
                total_assigned_hours=round(assigned_minutes_by_faculty.get(faculty.id, 0) / 60.0, 2),
                assignments=assignments,
            )
        )

    output.sort(key=lambda item: item.faculty_name.lower())
    return output


@router.post("/publish-offline", response_model=OfflinePublishResponse)
def publish_offline_timetable(
    payload: OfflinePublishRequest,
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> OfflinePublishResponse:
    official_payload = _load_official_payload(db)
    result = _send_offline_timetable_emails(
        db=db,
        payload=official_payload,
        filters=payload.filters,
    )
    log_activity(
        db,
        user=current_user,
        action="timetable.publish_offline",
        entity_type="official_timetable",
        entity_id="1",
        details={
            "attempted": result.attempted,
            "sent": result.sent,
            "failed": result.failed,
            "skipped": result.skipped,
            "filters": payload.model_dump(by_alias=True),
        },
    )
    db.commit()
    return result


@router.post("/publish-offline/all", response_model=OfflinePublishResponse)
def publish_offline_timetable_all(
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> OfflinePublishResponse:
    official_payload = _load_official_payload(db)
    result = _send_offline_timetable_emails(
        db=db,
        payload=official_payload,
        filters=None,
    )
    log_activity(
        db,
        user=current_user,
        action="timetable.publish_offline_all",
        entity_type="official_timetable",
        entity_id="1",
        details={
            "attempted": result.attempted,
            "sent": result.sent,
            "failed": result.failed,
            "skipped": result.skipped,
        },
    )
    db.commit()
    return result


@router.post("/publish-distribution", response_model=OfflinePublishResponse)
def publish_timetable_distribution(
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> OfflinePublishResponse:
    official_payload = _load_official_payload(db)
    result = _send_timetable_distribution_emails(db=db, payload=official_payload)
    log_activity(
        db,
        user=current_user,
        action="timetable.publish_distribution",
        entity_type="official_timetable",
        entity_id="1",
        details={
            "attempted": result.attempted,
            "sent": result.sent,
            "failed": result.failed,
            "skipped": result.skipped,
        },
    )
    db.commit()
    return result


@router.get("/conflicts", response_model=list[TimetableConflict])
def get_timetable_conflicts(
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> list[TimetableConflict]:
    """
    Returns all detected conflicts in the current official timetable.
    
    Includes both unresolved conflicts and those that have been marked with a decision (ignored/resolved).
    """
    payload = _load_official_payload(db)
    conflicts = _build_conflicts(payload, db)
    decisions = _load_conflict_decision_map(db)
    return _merge_conflicts_with_decisions(conflicts=conflicts, decisions=decisions)


@router.post("/conflicts/analyze", response_model=list[TimetableConflict])
def analyze_timetable_conflicts(
    payload: OfficialTimetablePayload,
    current_user: User = Depends(
        require_roles(UserRole.admin, UserRole.scheduler, UserRole.faculty, UserRole.student)
    ),
    db: Session = Depends(get_db),
) -> list[TimetableConflict]:
    """
    Analyzes a *proposed* timetable payload for conflicts without saving it.
    
    Useful for previewing the impact of changes before committing to the official schedule.
    """
    del current_user
    return _build_conflicts(payload, db)


@router.post("/conflicts/review", response_model=TimetableConflictReviewOut)
def review_timetable_conflicts(
    request: TimetableConflictReviewIn,
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> TimetableConflictReviewOut:
    del current_user
    if request.payload is not None:
        payload = request.payload
        source = "provided"
    else:
        payload = _load_official_payload(db)
        source = "official"

    decisions = _load_conflict_decision_map(db)
    merged_conflicts = _merge_conflicts_with_decisions(
        conflicts=_build_conflicts(payload, db),
        decisions=decisions,
    )
    auto_resolved, manually_resolved, ignored_conflicts, pending_conflicts = _categorize_conflicts_for_review(
        merged_conflicts,
    )

    unresolved_required = [
        item for item in pending_conflicts if (item.resolution_mode or "").strip().lower() != "ignored"
    ]
    unresolved_hard_count = sum(1 for item in unresolved_required if item.severity == "high")
    constraint_mismatches = _collect_constraint_mismatches(payload, db)
    can_publish = unresolved_hard_count == 0 and not constraint_mismatches

    return TimetableConflictReviewOut(
        source=source,
        auto_resolved_conflicts=auto_resolved,
        manually_resolved_conflicts=manually_resolved,
        ignored_conflicts=ignored_conflicts,
        pending_conflicts=unresolved_required,
        unresolved_required_count=len(unresolved_required),
        unresolved_hard_count=unresolved_hard_count,
        constraint_mismatches=constraint_mismatches,
        can_publish=can_publish,
        can_publish_anyway=True,
    )


@router.post("/conflicts/resolve-all", response_model=TimetableConflictResolveAllOut)
def resolve_all_timetable_conflicts(
    request: TimetableConflictResolveAllIn,
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> TimetableConflictResolveAllOut:
    if request.payload is not None:
        working_payload = OfficialTimetablePayload.model_validate(request.payload.model_dump(by_alias=True))
        source = "provided"
    else:
        working_payload = _load_official_payload(db)
        source = "official"

    resolved_entries: list[TimetableConflict] = []
    max_rounds = min(400, max(40, len(working_payload.timetable_data) * 3))
    visited_signatures: set[tuple[tuple[str, str, str, str, str, str, str], ...]] = {
        _payload_conflict_resolution_signature(working_payload),
    }
    rounds = 0
    while rounds < max_rounds:
        rounds += 1
        decision_map = _load_conflict_decision_map(db) if source == "official" else {}
        merged_conflicts = _merge_conflicts_with_decisions(
            conflicts=_build_conflicts(working_payload, db),
            decisions=decision_map,
        )
        _auto_resolved, _manual_resolved, _ignored, pending = _categorize_conflicts_for_review(merged_conflicts)
        target_conflicts = pending
        if request.scope == "hard":
            target_conflicts = [item for item in pending if item.severity == "high"]
        if not target_conflicts:
            break

        pre_hard = sum(1 for item in merged_conflicts if item.severity == "high" and not item.resolved)
        pre_total = len(merged_conflicts)
        best_candidate: dict | None = None
        for conflict in sorted(target_conflicts, key=_conflict_resolution_priority):
            candidate_payload = OfficialTimetablePayload.model_validate(working_payload.model_dump(by_alias=True))
            resolved_payload, resolution_message = _apply_best_effort_resolution(
                payload=candidate_payload,
                conflict=conflict,
                db=db,
            )
            if resolved_payload is None:
                continue

            signature = _payload_conflict_resolution_signature(resolved_payload)
            if signature in visited_signatures:
                continue

            post_conflicts = _build_conflicts(resolved_payload, db)
            post_hard = sum(1 for item in post_conflicts if item.severity == "high")
            conflict_resolved = not any(item.id == conflict.id for item in post_conflicts)
            if not conflict_resolved and post_hard >= pre_hard and len(post_conflicts) >= pre_total:
                continue

            candidate_score = (
                post_hard,
                len(post_conflicts),
                0 if conflict_resolved else 1,
                _conflict_resolution_priority(conflict),
            )
            if best_candidate is None or candidate_score < best_candidate["score"]:
                best_candidate = {
                    "score": candidate_score,
                    "payload": resolved_payload,
                    "conflict": conflict,
                    "message": resolution_message,
                    "signature": signature,
                    "post_conflicts": post_conflicts,
                }

        if best_candidate is None:
            break

        working_payload = best_candidate["payload"]
        visited_signatures.add(best_candidate["signature"])

        resolved_conflict = best_candidate["conflict"]
        resolution_message = str(best_candidate["message"] or "Auto resolution applied.")
        post_conflicts = best_candidate["post_conflicts"]
        note = (
            "[Auto-Resolved Bulk] "
            f"scope={request.scope}; before={pre_total}; after={len(post_conflicts)}; "
            f"hard_before={pre_hard}; hard_after={sum(1 for item in post_conflicts if item.severity == 'high')}; "
            f"action={resolution_message}"
        )
        _upsert_auto_resolved_decision(
            db=db,
            conflict=resolved_conflict,
            current_user=current_user,
            resolution_message=resolution_message,
            note=note,
        )
        resolved_record = TimetableConflict.model_validate(resolved_conflict.model_dump(by_alias=True))
        resolved_record.resolved = True
        resolved_record.decision = "yes"
        resolved_record.resolution_mode = "auto"
        resolved_record.decision_note = note
        resolved_record.resolution = resolution_message
        resolved_entries.append(resolved_record)

    merged_final = _merge_conflicts_with_decisions(
        conflicts=_build_conflicts(working_payload, db),
        decisions=_load_conflict_decision_map(db) if source == "official" else {},
    )
    _, _, _, pending_final = _categorize_conflicts_for_review(merged_final)
    remaining_conflicts = pending_final
    if request.scope == "hard":
        remaining_conflicts = [item for item in pending_final if item.severity == "high"]
    constraint_mismatches = _collect_constraint_mismatches(working_payload, db)

    promote_official = request.promote_official if request.promote_official is not None else (source == "provided")
    promoted_version_label: str | None = None
    if promote_official:
        payload_dict = working_payload.model_dump(by_alias=True)
        record = db.get(OfficialTimetable, 1)
        if record is None:
            record = OfficialTimetable(id=1, payload=payload_dict, updated_by_id=current_user.id)
            db.add(record)
        else:
            record.payload = payload_dict
            record.updated_by_id = current_user.id

        summary = _build_analytics(working_payload, _build_conflicts(working_payload, db), db).model_dump(by_alias=True)
        summary["source"] = "conflict-auto-resolve"
        promoted_version_label = _next_version_label(db)
        db.add(
            TimetableVersion(
                label=promoted_version_label,
                payload=payload_dict,
                summary=summary,
                created_by_id=current_user.id,
            )
        )
        log_activity(
            db,
            user=current_user,
            action="timetable.conflicts.resolve_all.promote",
            entity_type="official_timetable",
            entity_id="1",
            details={
                "source": source,
                "scope": request.scope,
                "resolved_count": len(resolved_entries),
                "remaining_conflicts": len(remaining_conflicts),
                "constraint_mismatches": len(constraint_mismatches),
                "note": request.note,
                "promoted_version_label": promoted_version_label,
            },
        )

    db.commit()
    return TimetableConflictResolveAllOut(
        source=source,
        resolved_payload=working_payload,
        resolved_count=len(resolved_entries),
        remaining_conflicts=remaining_conflicts,
        auto_resolved_conflicts=resolved_entries,
        constraint_mismatches=constraint_mismatches,
        promoted_version_label=promoted_version_label,
    )


@router.post("/conflicts/{conflict_id}/decision", response_model=ConflictDecisionOut)
def decide_timetable_conflict(
    conflict_id: str,
    payload: ConflictDecisionIn,
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> ConflictDecisionOut:
    """
    Applies a decision to a specific conflict (e.g., 'ignore', 'resolve').
    
    If 'resolve' is chosen, the system attempts to automatically fix the conflict 
    and creates a new version of the timetable.
    """
    record = db.get(OfficialTimetable, 1)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Official timetable not found")

    official_payload = OfficialTimetablePayload.model_validate(record.payload)
    conflicts = _build_conflicts(official_payload, db)
    current = next((item for item in conflicts if item.id == conflict_id), None)

    decision = db.execute(
        select(TimetableConflictDecision).where(TimetableConflictDecision.conflict_id == conflict_id)
    ).scalar_one_or_none()
    if decision is None:
        decision = TimetableConflictDecision(conflict_id=conflict_id, decision=ConflictDecision.no, resolved=False)
        db.add(decision)

    if current is None:
        if decision.decision == ConflictDecision.yes and decision.resolved:
            return ConflictDecisionOut(
                conflict_id=conflict_id,
                decision=decision.decision.value,
                resolved=True,
                message="Conflict is already resolved.",
                published_version_label=None,
            )
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conflict not found in current timetable")

    decision.note = payload.note
    decision.decided_by_id = current_user.id
    decision.decision = ConflictDecision(payload.decision)
    decision.conflict_snapshot = current.model_dump(by_alias=True)

    if payload.decision == "no":
        decision.resolved = False
        db.commit()
        return ConflictDecisionOut(
            conflict_id=conflict_id,
            decision=payload.decision,
            resolved=False,
            message="Recommendation skipped. Conflict remains active.",
            published_version_label=None,
        )

    working_payload = OfficialTimetablePayload.model_validate(official_payload.model_dump(by_alias=True))
    resolved_payload, resolution_message = _apply_best_effort_resolution(
        payload=working_payload,
        conflict=current,
        db=db,
    )
    if resolved_payload is None:
        decision.resolved = False
        db.commit()
        return ConflictDecisionOut(
            conflict_id=conflict_id,
            decision=payload.decision,
            resolved=False,
            message=resolution_message,
            published_version_label=None,
        )

    post_conflicts = _build_conflicts(resolved_payload, db)
    if any(item.id == conflict_id for item in post_conflicts):
        decision.resolved = False
        db.commit()
        return ConflictDecisionOut(
            conflict_id=conflict_id,
            decision=payload.decision,
            resolved=False,
            message="Automatic change did not fully resolve this conflict; manual action is required.",
            published_version_label=None,
        )

    record.payload = resolved_payload.model_dump(by_alias=True)
    record.updated_by_id = current_user.id

    version_label = _next_version_label(db)
    db.add(
        TimetableVersion(
            label=version_label,
            payload=resolved_payload.model_dump(by_alias=True),
            summary={
                "program_id": resolved_payload.program_id,
                "term_number": resolved_payload.term_number,
                "slots": len(resolved_payload.timetable_data),
                "source": "conflict-resolution",
                "resolved_conflict_id": conflict_id,
            },
            created_by_id=current_user.id,
        )
    )

    decision.resolved = True
    db.commit()
    return ConflictDecisionOut(
        conflict_id=conflict_id,
        decision=payload.decision,
        resolved=True,
        message=resolution_message,
        published_version_label=version_label,
    )


@router.get("/analytics", response_model=TimetableAnalytics)
def get_timetable_analytics(
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> TimetableAnalytics:
    payload = _load_official_payload(db)
    conflicts = _build_conflicts(payload, db)
    return _build_analytics(payload, conflicts, db)


@router.get("/versions", response_model=list[TimetableVersionOut])
def list_timetable_versions(
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> list[TimetableVersionOut]:
    """
    Lists all historical versions of the official timetable.
    
    Versions are created automatically on publish or major modification (like conflict resolution).
    """
    return list(db.execute(select(TimetableVersion).order_by(TimetableVersion.created_at.desc())).scalars())


@router.get("/versions/{version_id}/payload", response_model=OfficialTimetablePayload)
def get_timetable_version_payload(
    version_id: str,
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> OfficialTimetablePayload:
    """
    Returns the full timetable payload for a specific historical version.

    Used by the Versions page to render side-by-side visual comparison.
    """
    del current_user
    version = db.get(TimetableVersion, version_id)
    if version is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Version not found")
    return OfficialTimetablePayload.model_validate(version.payload)


@router.get("/versions/compare", response_model=TimetableVersionCompare)
def compare_timetable_versions(
    from_id: str = Query(..., alias="from"),
    to_id: str = Query(..., alias="to"),
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> TimetableVersionCompare:
    from_version = db.get(TimetableVersion, from_id)
    to_version = db.get(TimetableVersion, to_id)
    if from_version is None or to_version is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Version not found")

    from_payload = OfficialTimetablePayload.model_validate(from_version.payload)
    to_payload = OfficialTimetablePayload.model_validate(to_version.payload)
    from_slots = _slot_fingerprints(from_payload)
    to_slots = _slot_fingerprints(to_payload)

    added = to_slots - from_slots
    removed = from_slots - to_slots
    changed = min(len(added), len(removed))

    return TimetableVersionCompare(
        from_version_id=from_version.id,
        to_version_id=to_version.id,
        added_slots=len(added),
        removed_slots=len(removed),
        changed_slots=changed,
        from_label=from_version.label,
        to_label=to_version.label,
    )


@router.get("/trends", response_model=list[TimetableTrendPoint])
def timetable_trends(
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> list[TimetableTrendPoint]:
    versions = list(db.execute(select(TimetableVersion).order_by(TimetableVersion.created_at.asc())).scalars())
    trend_points: list[TimetableTrendPoint] = []
    for version in versions:
        payload = OfficialTimetablePayload.model_validate(version.payload)
        conflicts = _build_conflicts(payload, db)
        analytics = _build_analytics(payload, conflicts, db)
        trend_points.append(
            TimetableTrendPoint(
                version_id=version.id,
                label=version.label,
                created_at=version.created_at,
                constraint_satisfaction=analytics.optimization_summary.constraint_satisfaction,
                conflicts_detected=analytics.optimization_summary.conflicts_detected,
            )
        )
    return trend_points


@router.put("/official", response_model=OfficialTimetablePayload)
def upsert_official_timetable(
    payload: OfficialTimetablePayload,
    version_label: str | None = Query(default=None, alias="versionLabel", max_length=100),
    force: bool = Query(default=False),
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> OfficialTimetablePayload:
    course_by_id = {course.id: course for course in payload.course_data}
    faculty_by_id = {faculty.id: faculty for faculty in payload.faculty_data}
    room_by_id = {room.id: room for room in payload.room_data}
    shared_groups: list[tuple[str, str, set[str]]] = []
    shared_groups_by_course: dict[str, list[set[str]]] = {}
    if payload.program_id and payload.term_number is not None:
        shared_groups = load_shared_lecture_groups(
            db=db,
            program_id=payload.program_id,
            term_number=payload.term_number,
        )
        shared_groups_by_course = build_shared_group_lookup(shared_groups)

    working_hours = load_working_hours(db)
    schedule_policy = load_schedule_policy(db)
    program_constraint = load_program_constraint(db, payload.program_id) if payload.program_id else None
    program_daily_slots = normalize_program_daily_slots(
        program_constraint.daily_time_slots if program_constraint is not None else None
    )
    period_minutes = schedule_policy.period_minutes
    day_segments: dict[str, list[tuple[int, int]]] = {}
    day_blocked_segments: dict[str, list[tuple[int, int, str]]] = {}
    for day, hours_entry in working_hours.items():
        if not hours_entry.enabled:
            continue
        if program_daily_slots:
            teaching_segments, blocked_segments = build_teaching_segments_from_program_slots(program_daily_slots)
            if teaching_segments:
                day_segments[day] = teaching_segments
            if blocked_segments:
                day_blocked_segments[day] = blocked_segments
            continue

        day_start = parse_time_to_minutes(hours_entry.start_time)
        day_end = parse_time_to_minutes(hours_entry.end_time)
        day_segments[day] = build_teaching_segments(
            day_start=day_start,
            day_end=day_end,
            period_minutes=period_minutes,
            breaks=schedule_policy.breaks,
        )
        day_blocked_segments[day] = [
            (
                parse_time_to_minutes(item.start_time),
                parse_time_to_minutes(item.end_time),
                item.name,
            )
            for item in schedule_policy.breaks
        ]

    for slot in payload.timetable_data:
        hours_entry = working_hours.get(slot.day)
        segments = day_segments.get(slot.day, [])
        if hours_entry is None or not hours_entry.enabled or not segments:
            if not force:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Timeslot {slot.id} occurs on a non-working day ({slot.day})",
                )
        slot_start = parse_time_to_minutes(slot.startTime)
        slot_end = parse_time_to_minutes(slot.endTime)
        allowed_start = min(start for start, _end in segments)
        allowed_end = max(end for _start, end in segments)
        if slot_start < allowed_start or slot_end > allowed_end:
            if not force:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"Timeslot {slot.id} on {slot.day} must be within working hours "
                        f"{hours_entry.start_time}-{hours_entry.end_time}"
                    ),
                )
        slot_duration = slot_end - slot_start
        if slot_duration % period_minutes != 0:
            if not force:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Timeslot {slot.id} must be a multiple of {period_minutes} minutes",
                )
        if not is_slot_aligned_with_segments(slot_start, slot_end, segments):
            if not force:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Timeslot {slot.id} must align to configured teaching slot boundaries",
                )
        blocked_overlap = next(
            (
                (start, end, label)
                for start, end, label in day_blocked_segments.get(slot.day, [])
                if slot_start < end and slot_end > start
            ),
            None,
        )
        if blocked_overlap is not None:
            if not force:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"Timeslot {slot.id} overlaps non-teaching slot '{blocked_overlap[2]}' "
                        f"({_minutes_to_time(blocked_overlap[0])}-{_minutes_to_time(blocked_overlap[1])})"
                    ),
                )

    if payload.term_number is None:
        has_constraints = db.execute(select(SemesterConstraint.id)).first() is not None
        if has_constraints:
            if not force:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="termNumber is required to validate semester constraints",
                )
    else:
        constraint = load_semester_constraint(db, payload.term_number)
        if constraint is not None:
            enforce_semester_constraints(payload, constraint, force=force)

    enforce_resource_conflicts(payload, course_by_id, shared_groups_by_course, force=force)
    enforce_course_scheduling(payload, course_by_id, room_by_id, schedule_policy, force=force)
    student_counts_by_slot = enforce_room_capacity(payload, room_by_id, db, force=force)
    enforce_shared_lecture_constraints(
        payload,
        shared_groups,
        shared_groups_by_course,
        room_by_id,
        student_counts_by_slot,
        force=force,
    )
    enforce_section_credit_aligned_minutes(payload, db, schedule_policy, force=force)
    enforce_program_credit_requirements(payload, course_by_id, db, force=force)
    enforce_elective_overlap_constraints(payload, db, force=force)
    enforce_prerequisite_constraints(payload, db, force=force)
    enforce_faculty_overload_preferences(payload, db, force=force)
    enforce_single_faculty_per_course_sections(payload, course_by_id, faculty_by_id, force=force)

    record = db.get(OfficialTimetable, 1)
    payload_dict = payload.model_dump(by_alias=True)
    old_payload = OfficialTimetablePayload.model_validate(record.payload) if record is not None else None
    if record is None:
        record = OfficialTimetable(id=1, payload=payload_dict, updated_by_id=current_user.id)
        db.add(record)
    else:
        record.payload = payload_dict
        record.updated_by_id = current_user.id

    conflicts = _build_conflicts(payload, db)
    analytics = _build_analytics(payload, conflicts, db)
    summary = analytics.model_dump(by_alias=True)
    version = TimetableVersion(
        label=(version_label.strip() if version_label else _next_version_label(db)),
        payload=payload_dict,
        summary=summary,
        created_by_id=current_user.id,
    )
    db.add(version)

    # Resolve all impacted users relative to current official version (if any)
    impacted_faculty_user_ids, impacted_student_user_ids = _resolve_impacted_schedule_users(
        db=db,
        old_payload=old_payload,
        new_payload=payload,
    )

    if old_payload is not None:
        old_slots = _slot_fingerprints(old_payload)
        new_slots = _slot_fingerprints(payload)
        added = len(new_slots - old_slots)
        removed = len(old_slots - new_slots)
        change_message = (
            f"Official timetable updated ({version.label}). Added {added} slot(s), removed {removed} slot(s)."
        )
    else:
        change_message = f"Official timetable published ({version.label})."
    log_activity(
        db,
        user=current_user,
        action="timetable.publish",
        entity_type="official_timetable",
        entity_id="1",
        details={"version_label": version.label, **summary},
    )

    db.commit()
    try:
        if impacted_faculty_user_ids:
            notify_users(
                db,
                user_ids=impacted_faculty_user_ids,
                title="Teaching Schedule Updated",
                message=(
                    f"The official timetable was updated ({version.label}). "
                    "Your assigned teaching slots have changed."
                ),
                notification_type=NotificationType.timetable,
                exclude_user_id=current_user.id,
                deliver_email=False,
            )
        if impacted_student_user_ids:
            notify_users(
                db,
                user_ids=impacted_student_user_ids,
                title="Class Schedule Updated",
                message=(
                    f"The official timetable was updated ({version.label}). "
                    "Your section schedule has changed."
                ),
                notification_type=NotificationType.timetable,
                exclude_user_id=current_user.id,
                deliver_email=False,
            )

        notify_all_users(
            db,
            title="Timetable Update",
            message=change_message,
            notification_type=NotificationType.timetable,
            exclude_user_id=current_user.id,
            deliver_email=False,
        )
        db.commit()
    except Exception:
        # We DO NOT rollback here because the timetable itself was already committed at line 2903.
        # We only lose the notification records if this fails, but it's better than losing the publish.
        logger.exception(
            "TIMETABLE PUBLISH NOTIFICATION FAILED | user_id=%s | program_id=%s | term=%s | version=%s",
            current_user.id,
            payload.program_id,
            payload.term_number,
            version.label,
        )
    db.refresh(record)
    return OfficialTimetablePayload.model_validate(record.payload)


def _serialize_change_request_rows(
    rows: list[TimetableChangeRequest],
    db: Session,
) -> list[TimetableChangeRequestOut]:
    user_ids: set[str] = set()
    for row in rows:
        if row.requested_by_id:
            user_ids.add(row.requested_by_id)
        if row.approver_user_id:
            user_ids.add(row.approver_user_id)

    user_name_by_id: dict[str, str] = {}
    if user_ids:
        for item in db.execute(select(User.id, User.name).where(User.id.in_(user_ids))).all():
            user_name_by_id[str(item.id)] = str(item.name)

    result: list[TimetableChangeRequestOut] = []
    for row in rows:
        model = TimetableChangeRequestOut.model_validate(row)
        result.append(
            model.model_copy(
                update={
                    "requested_by_name": user_name_by_id.get(row.requested_by_id),
                    "approver_name": user_name_by_id.get(row.approver_user_id) if row.approver_user_id else None,
                }
            )
        )
    return result


@router.get("/change-requests", response_model=list[TimetableChangeRequestOut])
def list_timetable_change_requests(
    status_filter: TimetableChangeRequestStatus | None = Query(default=None, alias="status"),
    mine: bool = Query(default=False),
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler, UserRole.faculty, UserRole.student)),
    db: Session = Depends(get_db),
) -> list[TimetableChangeRequestOut]:
    statement = select(TimetableChangeRequest)
    if status_filter is not None:
        statement = statement.where(TimetableChangeRequest.status == status_filter)

    if current_user.role not in {UserRole.admin, UserRole.scheduler}:
        statement = statement.where(
            (TimetableChangeRequest.requested_by_id == current_user.id)
            | (TimetableChangeRequest.approver_user_id == current_user.id)
        )
    elif mine:
        statement = statement.where(
            (TimetableChangeRequest.requested_by_id == current_user.id)
            | (TimetableChangeRequest.approver_user_id == current_user.id)
        )

    rows = list(
        db.execute(
            statement.order_by(TimetableChangeRequest.created_at.desc())
        ).scalars()
    )
    return _serialize_change_request_rows(rows, db)


@router.post("/change-requests", response_model=TimetableChangeRequestOut, status_code=status.HTTP_201_CREATED)
def propose_timetable_change_request(
    payload: TimetableChangeRequestProposalIn,
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler, UserRole.faculty, UserRole.student)),
    db: Session = Depends(get_db),
) -> TimetableChangeRequestOut:
    official_payload = _load_official_payload(db)
    target_slot = next((item for item in official_payload.timetable_data if item.id == payload.slot_id), None)
    if target_slot is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Selected slot is not present in official timetable")
    request_kind = (payload.request_kind or "slot_move").strip().lower()

    if payload.room_id is not None and not any(item.id == payload.room_id for item in official_payload.room_data):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Selected roomId is not valid for this timetable")

    if payload.faculty_id is not None and not any(item.id == payload.faculty_id for item in official_payload.faculty_data):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Selected facultyId is not valid for this timetable")

    if payload.assistant_faculty_ids:
        invalid_assistants = [
            faculty_id
            for faculty_id in payload.assistant_faculty_ids
            if not any(item.id == faculty_id for item in official_payload.faculty_data)
        ]
        if invalid_assistants:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid assistantFacultyIds provided: {', '.join(invalid_assistants)}",
            )

    approver_user: User | None = None
    approver_role: str | None = None

    if current_user.role == UserRole.student:
        user_section = (current_user.section_name or "").strip().lower()
        if user_section and user_section != target_slot.section.strip().lower():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Students can only propose changes for their own section",
            )
        if payload.faculty_id:
            approver_user = _resolve_faculty_user_by_id(official_payload, payload.faculty_id, db)
        else:
            approver_user = _resolve_faculty_user_for_slot(official_payload, target_slot, db)
        approver_role = UserRole.faculty.value
    elif current_user.role == UserRole.faculty:
        faculty_email = (current_user.email or "").strip().lower()
        allowed_faculty_ids: set[str] = {
            item.id
            for item in official_payload.faculty_data
            if (item.email or "").strip().lower() == faculty_email
        }
        if not allowed_faculty_ids and faculty_email:
            faculty_profile = (
                db.execute(select(Faculty).where(func.lower(Faculty.email) == faculty_email))
                .scalars()
                .first()
            )
            if faculty_profile is not None:
                allowed_faculty_ids.add(faculty_profile.id)

        if not allowed_faculty_ids:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Faculty profile is not mapped to your user account",
            )

        if target_slot.facultyId not in allowed_faculty_ids and not any(
            item in allowed_faculty_ids for item in _slot_assistant_faculty_ids(target_slot)
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Faculty can only propose changes for slots assigned to them",
            )
        target_faculty_id = (payload.faculty_id or "").strip()
        if target_faculty_id and target_faculty_id not in allowed_faculty_ids:
            approver_user = _resolve_faculty_user_by_id(official_payload, target_faculty_id, db)
            approver_role = UserRole.faculty.value
        else:
            approver_user = _resolve_cr_student_for_section(
                db=db,
                program_id=official_payload.program_id,
                term_number=official_payload.term_number,
                section_name=target_slot.section,
            )
            approver_role = UserRole.student.value
    else:
        approver_user = current_user
        approver_role = current_user.role.value

    if approver_user is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No eligible approver could be resolved for this request",
        )

    request_row = TimetableChangeRequest(
        program_id=official_payload.program_id,
        term_number=official_payload.term_number,
        slot_id=payload.slot_id,
        requested_by_id=current_user.id,
        requested_by_role=current_user.role.value,
        approver_user_id=approver_user.id,
        approver_role=approver_role,
        status=TimetableChangeRequestStatus.pending,
        proposal=payload.model_dump(by_alias=True),
        request_note=(payload.note or "").strip() or None,
    )
    db.add(request_row)
    db.flush()

    create_notification(
        db,
        user_id=approver_user.id,
        title="Timetable Change Request",
        message=(
            f"{current_user.name} requested a {request_kind.replace('_', ' ')} for slot {payload.slot_id} "
            f"({payload.day} {payload.start_time}-{payload.end_time})."
        ),
        notification_type=NotificationType.timetable,
        recipient=approver_user,
        deliver_email=False,
    )
    log_activity(
        db,
        user=current_user,
        action="timetable.change_request.propose",
        entity_type="timetable_change_request",
        entity_id=request_row.id,
        details={
            "slot_id": payload.slot_id,
            "request_kind": request_kind,
            "requested_by_role": current_user.role.value,
            "approver_user_id": approver_user.id,
            "approver_role": approver_role,
        },
    )
    db.commit()
    db.refresh(request_row)
    return _serialize_change_request_rows([request_row], db)[0]


@router.post("/change-requests/{request_id}/decision", response_model=TimetableChangeRequestDecisionOut)
def decide_timetable_change_request(
    request_id: str,
    payload: TimetableChangeRequestDecisionIn,
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler, UserRole.faculty, UserRole.student)),
    db: Session = Depends(get_db),
) -> TimetableChangeRequestDecisionOut:
    request_row = db.get(TimetableChangeRequest, request_id)
    if request_row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Change request not found")

    if request_row.status != TimetableChangeRequestStatus.pending:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Change request is already {request_row.status.value}",
        )

    if current_user.role not in {UserRole.admin, UserRole.scheduler} and request_row.approver_user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You are not allowed to decide this request")

    decision_note = (payload.note or "").strip() or None

    if payload.decision == "reject":
        request_row.status = TimetableChangeRequestStatus.rejected
        request_row.decision_note = decision_note
        request_row.decided_at = datetime.now(timezone.utc)
        db.add(request_row)
        db.commit()
        db.refresh(request_row)
        return TimetableChangeRequestDecisionOut(
            request=_serialize_change_request_rows([request_row], db)[0],
            message="Change request rejected.",
        )

    official_payload = _load_official_payload(db)
    updated_payload, updated_slot, affected_slot_ids = _apply_change_proposal_to_payload(
        official_payload,
        request_row.proposal,
    )
    impacted_conflicts = [
        item
        for item in _build_conflicts(updated_payload, db)
        if not item.resolved and any(slot_id in item.affected_slots for slot_id in affected_slot_ids)
    ]

    resolution_note_parts: list[str] = []
    if impacted_conflicts:
        room_only = all(item.conflict_type in ROOM_ONLY_CONFLICT_TYPES for item in impacted_conflicts)
        if not room_only:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Proposed change introduces non-room conflicts. "
                    "Please submit a different proposal."
                ),
            )

        alternative_room_id = _find_alternative_room_id_for_slot(updated_payload, updated_slot)
        if alternative_room_id is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only room conflict was found, but no alternative free room is available.",
            )
        updated_slot.roomId = alternative_room_id
        resolution_note_parts.append(f"Room adjusted automatically to {alternative_room_id}.")

        remaining_after_room_fix = [
            item
            for item in _build_conflicts(updated_payload, db)
            if not item.resolved and any(slot_id in item.affected_slots for slot_id in affected_slot_ids)
        ]
        if remaining_after_room_fix:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not auto-resolve room conflicts for the approved request.",
            )

    version_label = f"Change Request {request_row.id[:8]}"
    upsert_official_timetable(
        payload=updated_payload,
        version_label=version_label,
        force=False,
        current_user=current_user,
        db=db,
    )

    request_row.status = TimetableChangeRequestStatus.applied
    request_row.decision_note = decision_note
    request_row.decided_at = datetime.now(timezone.utc)
    request_row.applied_at = datetime.now(timezone.utc)
    if resolution_note_parts:
        request_row.resolution_note = " ".join(resolution_note_parts)
    db.add(request_row)

    requester = db.get(User, request_row.requested_by_id)
    if requester is not None:
        create_notification(
            db,
            user_id=requester.id,
            title="Timetable Change Request Applied",
            message=(
                f"Your change request for slot {request_row.slot_id} was approved and applied."
            ),
            notification_type=NotificationType.timetable,
            recipient=requester,
            deliver_email=False,
        )

    log_activity(
        db,
        user=current_user,
        action="timetable.change_request.apply",
        entity_type="timetable_change_request",
        entity_id=request_row.id,
        details={
            "slot_id": request_row.slot_id,
            "request_kind": str((request_row.proposal or {}).get("requestKind") or "slot_move"),
            "requester_id": request_row.requested_by_id,
            "approver_id": request_row.approver_user_id,
            "decision_note": decision_note,
            "resolution_note": request_row.resolution_note,
        },
    )
    db.commit()
    db.refresh(request_row)

    return TimetableChangeRequestDecisionOut(
        request=_serialize_change_request_rows([request_row], db)[0],
        message="Change request approved and timetable updated.",
    )
