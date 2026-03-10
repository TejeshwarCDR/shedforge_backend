from __future__ import annotations

from collections import Counter
from datetime import datetime, time, timedelta, timezone
import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db, require_roles
from app.core.config import get_settings
from app.models.activity_log import ActivityLog
from app.models.course import Course
from app.models.faculty import Faculty
from app.models.feedback import FeedbackItem, FeedbackStatus
from app.models.leave_request import LeaveRequest, LeaveStatus
from app.models.notification import Notification, NotificationType
from app.models.program import Program
from app.models.program_structure import ProgramSection, ProgramTerm
from app.models.room import Room
from app.models.timetable import OfficialTimetable
from app.models.timetable_issue import IssueStatus, TimetableIssue
from app.models.user import User, UserRole
from app.schemas.system_analytics import (
    ActivityAnalyticsOut,
    AnalyticsScopeOut,
    CapacitySnapshotOut,
    DailyCountPoint,
    LabeledCount,
    MetricDefinitionOut,
    OperationsSnapshotOut,
    ResourceInventoryOut,
    SystemAnalyticsOut,
    TimetableSnapshotOut,
    UtilizationSnapshotOut,
)
from app.schemas.timetable import OfficialTimetablePayload, parse_time_to_minutes
from app.services.audit import log_activity
from app.services.notifications import notify_all_users

router = APIRouter()
settings = get_settings()


def _enum_label(value: object) -> str:
    if hasattr(value, "value"):
        return str(getattr(value, "value"))
    return str(value)


def _to_labeled_counts(counter: dict[str, int], limit: int | None = None) -> list[LabeledCount]:
    ordered = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    if limit is not None:
        ordered = ordered[:limit]
    return [LabeledCount(label=label, value=value) for label, value in ordered]


def _to_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@router.get("/system/info")
def system_info(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    has_timetable = db.get(OfficialTimetable, 1) is not None
    return {
        "name": settings.project_name,
        "api_prefix": settings.api_prefix,
        "help_sections": [
            "Authentication and roles",
            "Academic setup",
            "Generation and publishing",
            "Conflict resolution",
            "Reports and exports",
        ],
        "features": {
            "official_timetable_published": has_timetable,
            "generator_enabled": True,
            "issues_enabled": True,
            "notifications_enabled": True,
            "backups_enabled": True,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/system/analytics", response_model=SystemAnalyticsOut)
def system_analytics(
    days: int = Query(default=14, ge=1, le=90),
    program_id: str | None = Query(default=None, alias="programId"),
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> SystemAnalyticsOut:
    scoped_program = db.get(Program, program_id) if program_id else None
    if program_id and scoped_program is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Program not found")

    scoped_programs: list[Program]
    if scoped_program is not None:
        scoped_programs = [scoped_program]
    else:
        scoped_programs = list(db.execute(select(Program)).scalars().all())

    scoped_program_ids = {item.id for item in scoped_programs}

    programs = len(scoped_programs)
    program_terms_query = select(func.count(ProgramTerm.id))
    program_sections_query = select(func.count(ProgramSection.id))
    courses_query = select(func.count(Course.id))
    faculty_query = select(func.count(Faculty.id))
    room_total_query = select(func.count(Room.id))
    total_room_capacity_query = select(func.coalesce(func.sum(Room.capacity), 0))

    if scoped_program is not None:
        program_terms_query = program_terms_query.where(ProgramTerm.program_id == scoped_program.id)
        program_sections_query = program_sections_query.where(ProgramSection.program_id == scoped_program.id)
        courses_query = courses_query.where(Course.program_id == scoped_program.id)
        faculty_query = faculty_query.where(Faculty.program_id == scoped_program.id)
        room_total_query = room_total_query.where(Room.program_id == scoped_program.id)
        total_room_capacity_query = total_room_capacity_query.where(Room.program_id == scoped_program.id)

    program_terms = int(db.execute(program_terms_query).scalar_one() or 0)
    program_sections = int(db.execute(program_sections_query).scalar_one() or 0)
    courses = int(db.execute(courses_query).scalar_one() or 0)
    faculty = int(db.execute(faculty_query).scalar_one() or 0)
    rooms_total = int(db.execute(room_total_query).scalar_one() or 0)
    total_room_capacity = int(db.execute(total_room_capacity_query).scalar_one() or 0)
    users_total = int(db.execute(select(func.count(User.id))).scalar_one() or 0)

    room_type_counts = {"lecture": 0, "lab": 0, "seminar": 0}
    room_capacity_by_type = {"lecture": 0, "lab": 0, "seminar": 0}
    room_type_count_query = select(Room.type, func.count(Room.id)).group_by(Room.type)
    room_type_capacity_query = select(Room.type, func.coalesce(func.sum(Room.capacity), 0)).group_by(Room.type)
    if scoped_program is not None:
        room_type_count_query = room_type_count_query.where(Room.program_id == scoped_program.id)
        room_type_capacity_query = room_type_capacity_query.where(Room.program_id == scoped_program.id)
    for room_type, count in db.execute(room_type_count_query).all():
        room_type_counts[_enum_label(room_type)] = int(count)
    for room_type, total_capacity in db.execute(room_type_capacity_query).all():
        room_capacity_by_type[_enum_label(room_type)] = int(total_capacity or 0)

    users_by_role = {role.value: 0 for role in UserRole}
    for role, count in db.execute(select(User.role, func.count(User.id)).group_by(User.role)).all():
        users_by_role[_enum_label(role)] = int(count)

    if scoped_program_ids:
        configured_capacity_query = select(func.coalesce(func.sum(ProgramSection.capacity), 0)).where(
            ProgramSection.program_id.in_(scoped_program_ids)
        )
        configured_section_capacity = int(db.execute(configured_capacity_query).scalar_one() or 0)
    else:
        configured_section_capacity = 0
    if configured_section_capacity <= 0:
        configured_section_capacity = sum(
            int(item.total_students or 0)
            if int(item.total_students or 0) > 0
            else int(item.sections or 0) * int(item.default_section_capacity or 0)
            for item in scoped_programs
        )

    scoped_faculty_rows = db.execute(select(Faculty.id, Faculty.max_hours).where(Faculty.program_id.in_(scoped_program_ids))).all() if scoped_program_ids else []
    faculty_capacity_minutes = int(sum(max(0, int(max_hours or 0)) * 60 for _faculty_id, max_hours in scoped_faculty_rows))
    if faculty_capacity_minutes <= 0 and faculty > 0:
        faculty_capacity_minutes = faculty * 20 * 60

    record = db.get(OfficialTimetable, 1)
    timetable_payload: OfficialTimetablePayload | None = None
    if record is not None:
        try:
            timetable_payload = OfficialTimetablePayload.model_validate(record.payload)
        except Exception:
            timetable_payload = None

    timetable_scoped = bool(
        timetable_payload is not None and (
            scoped_program is None or timetable_payload.program_id == scoped_program.id
        )
    )
    slots = timetable_payload.timetable_data if timetable_scoped and timetable_payload is not None else []

    course_payload_by_id = {
        course.id: course for course in (timetable_payload.course_data if timetable_payload is not None else [])
    }
    scheduled_student_seats = 0
    room_slot_minutes = 0
    faculty_slot_minutes = 0
    slot_days_counter = Counter()
    unique_grid_windows: set[tuple[str, str, str]] = set()
    timetable_sections_set: set[str] = set()
    timetable_faculty_ids: set[str] = set()
    timetable_rooms_set: set[str] = set()
    timetable_courses_set: set[str] = set()

    for slot in slots:
        start_min = parse_time_to_minutes(slot.startTime)
        end_min = parse_time_to_minutes(slot.endTime)
        duration = max(0, end_min - start_min)
        if duration <= 0:
            continue

        slot_days_counter[slot.day] += 1
        unique_grid_windows.add((slot.day, slot.startTime, slot.endTime))
        section_name = (slot.section or "").strip().upper()
        if section_name:
            timetable_sections_set.add(section_name)
        if slot.courseId:
            timetable_courses_set.add(slot.courseId)
        if slot.roomId:
            timetable_rooms_set.add(slot.roomId)
        if slot.facultyId:
            timetable_faculty_ids.add(slot.facultyId)
        for assistant_id in slot.assistant_faculty_ids:
            if assistant_id:
                timetable_faculty_ids.add(assistant_id)

        scheduled_student_seats += int(slot.studentCount or 0)

        slot_course = course_payload_by_id.get(slot.courseId)
        requires_classroom = bool(getattr(slot_course, "assign_classroom", True)) if slot_course is not None else True
        requires_faculty = bool(getattr(slot_course, "assign_faculty", True)) if slot_course is not None else True

        if requires_classroom and slot.roomId and not str(slot.roomId).startswith("nr-r-"):
            room_slot_minutes += duration

        if requires_faculty:
            primary_faculty_id = str(slot.facultyId or "").strip()
            if primary_faculty_id and not primary_faculty_id.startswith("nr-f-"):
                faculty_slot_minutes += duration
            for assistant_id in slot.assistant_faculty_ids:
                cleaned = str(assistant_id or "").strip()
                if cleaned and not cleaned.startswith("nr-f-"):
                    faculty_slot_minutes += duration

    timetable_sections = len(timetable_sections_set)
    timetable_faculty = len({item for item in timetable_faculty_ids if not str(item).startswith("nr-f-")})
    timetable_rooms = len({item for item in timetable_rooms_set if not str(item).startswith("nr-r-")})
    timetable_courses = len(timetable_courses_set)

    active_terms: set[int] = set()
    for slot in slots:
        slot_course = course_payload_by_id.get(slot.courseId)
        semester_number = int(getattr(slot_course, "semester_number", 0) or 0) if slot_course is not None else 0
        if semester_number <= 0 and timetable_payload is not None and timetable_payload.term_number is not None:
            semester_number = int(timetable_payload.term_number)
        if semester_number > 0:
            active_terms.add(semester_number)

    total_grid_minutes = int(
        sum(max(0, parse_time_to_minutes(end) - parse_time_to_minutes(start)) for _day, start, end in unique_grid_windows)
    )
    available_room_minutes = rooms_total * total_grid_minutes

    room_utilization_percent = round(
        min(100.0, (room_slot_minutes * 100.0 / available_room_minutes)) if available_room_minutes > 0 else 0.0,
        1,
    )
    faculty_utilization_percent = round(
        min(100.0, (faculty_slot_minutes * 100.0 / faculty_capacity_minutes)) if faculty_capacity_minutes > 0 else 0.0,
        1,
    )

    configured_sections_denominator = 0
    if scoped_program_ids and active_terms:
        configured_sections_denominator = int(
            db.execute(
                select(func.count(ProgramSection.id)).where(
                    ProgramSection.program_id.in_(scoped_program_ids),
                    ProgramSection.term_number.in_(active_terms),
                )
            ).scalar_one()
            or 0
        )
    if configured_sections_denominator <= 0:
        configured_sections_denominator = program_sections
    if configured_sections_denominator <= 0:
        configured_sections_denominator = sum(max(0, int(item.sections or 0)) for item in scoped_programs)
    section_coverage_percent = round(
        min(100.0, (timetable_sections * 100.0 / configured_sections_denominator)) if configured_sections_denominator > 0 else 0.0,
        1,
    )

    total_logs = int(db.execute(select(func.count(ActivityLog.id))).scalar_one() or 0)
    window_days = int(days)
    now = datetime.now(timezone.utc)
    start_date = (now - timedelta(days=window_days - 1)).date()
    window_start = datetime.combine(start_date, time.min, tzinfo=timezone.utc)

    window_logs = list(
        db.execute(select(ActivityLog).where(ActivityLog.created_at >= window_start)).scalars()
    )
    recent_logs = list(
        db.execute(select(ActivityLog).order_by(ActivityLog.created_at.desc()).limit(20)).scalars()
    )

    actions_by_day_counter: dict[str, int] = {}
    for offset in range(window_days):
        day = start_date + timedelta(days=offset)
        actions_by_day_counter[day.isoformat()] = 0

    top_actions_counter: Counter[str] = Counter()
    top_entities_counter: Counter[str] = Counter()
    active_user_ids: set[str] = set()

    for item in window_logs:
        created = _to_utc(item.created_at)
        if created is not None:
            key = created.date().isoformat()
            if key in actions_by_day_counter:
                actions_by_day_counter[key] += 1
        if item.action:
            top_actions_counter[item.action] += 1
        if item.entity_type:
            top_entities_counter[item.entity_type] += 1
        if item.user_id:
            active_user_ids.add(item.user_id)

    notification_type_counts = {item.value: 0 for item in NotificationType}
    for notif_type, count in db.execute(
        select(Notification.notification_type, func.count(Notification.id)).group_by(Notification.notification_type)
    ).all():
        notification_type_counts[_enum_label(notif_type)] = int(count)
    unread_notifications = int(
        db.execute(select(func.count(Notification.id)).where(Notification.is_read.is_(False))).scalar_one() or 0
    )

    leave_status_counts = {item.value: 0 for item in LeaveStatus}
    for leave_status, count in db.execute(
        select(LeaveRequest.status, func.count(LeaveRequest.id)).group_by(LeaveRequest.status)
    ).all():
        leave_status_counts[_enum_label(leave_status)] = int(count)

    issue_status_counts = {item.value: 0 for item in IssueStatus}
    for issue_status, count in db.execute(
        select(TimetableIssue.status, func.count(TimetableIssue.id)).group_by(TimetableIssue.status)
    ).all():
        issue_status_counts[_enum_label(issue_status)] = int(count)

    feedback_status_counts = {item.value: 0 for item in FeedbackStatus}
    for feedback_status, count in db.execute(
        select(FeedbackItem.status, func.count(FeedbackItem.id)).group_by(FeedbackItem.status)
    ).all():
        feedback_status_counts[_enum_label(feedback_status)] = int(count)

    if timetable_payload is None:
        timetable_scope_note = "No published timetable snapshot is available yet."
    elif timetable_scoped:
        timetable_scope_note = "Metrics are computed from the published timetable within the selected scope."
    else:
        timetable_scope_note = "Published timetable belongs to a different program. Timetable-scoped metrics are zeroed for this scope."

    metric_definitions = [
        MetricDefinitionOut(
            key="room_utilization",
            label="Room Utilization",
            definition="Percentage of room-minute capacity used by scheduled teaching slots.",
            formula="scheduled_room_minutes / (rooms_total * grid_minutes) * 100",
            numerator=float(room_slot_minutes),
            denominator=float(available_room_minutes),
            unit="percent",
            computedValue=room_utilization_percent,
        ),
        MetricDefinitionOut(
            key="faculty_utilization",
            label="Faculty Utilization",
            definition="Percentage of faculty workload capacity consumed by primary + assistant assignments.",
            formula="scheduled_faculty_minutes / sum(faculty_max_hours * 60) * 100",
            numerator=float(faculty_slot_minutes),
            denominator=float(faculty_capacity_minutes),
            unit="percent",
            computedValue=faculty_utilization_percent,
        ),
        MetricDefinitionOut(
            key="section_coverage",
            label="Section Coverage",
            definition="Configured sections that appear in the published timetable at least once.",
            formula="scheduled_sections / configured_sections * 100",
            numerator=float(timetable_sections),
            denominator=float(max(0, configured_sections_denominator)),
            unit="percent",
            computedValue=section_coverage_percent,
        ),
    ]

    return SystemAnalyticsOut(
        generatedAt=now.isoformat(),
        scope=AnalyticsScopeOut(
            mode="program" if scoped_program is not None else "all_programs",
            programId=scoped_program.id if scoped_program is not None else None,
            programCode=scoped_program.code if scoped_program is not None else None,
            programName=scoped_program.name if scoped_program is not None else None,
            timetableScoped=timetable_scoped,
            timetableScopeNote=timetable_scope_note,
        ),
        inventory=ResourceInventoryOut(
            programs=programs,
            programTerms=program_terms,
            programSections=program_sections,
            courses=courses,
            faculty=faculty,
            roomsTotal=rooms_total,
            lectureRooms=room_type_counts["lecture"],
            labRooms=room_type_counts["lab"],
            seminarRooms=room_type_counts["seminar"],
            usersTotal=users_total,
            usersByRole=users_by_role,
        ),
        timetable=TimetableSnapshotOut(
            isPublished=timetable_scoped,
            updatedAt=_to_utc(record.updated_at).isoformat() if record is not None and record.updated_at else None,
            totalSlots=len(slots),
            sections=timetable_sections,
            faculty=timetable_faculty,
            rooms=timetable_rooms,
            courses=timetable_courses,
            slotsByDay=dict(sorted(slot_days_counter.items())),
        ),
        utilization=UtilizationSnapshotOut(
            roomUtilizationPercent=room_utilization_percent,
            facultyUtilizationPercent=faculty_utilization_percent,
            sectionCoveragePercent=section_coverage_percent,
        ),
        capacity=CapacitySnapshotOut(
            totalRoomCapacity=total_room_capacity,
            lectureRoomCapacity=room_capacity_by_type["lecture"],
            labRoomCapacity=room_capacity_by_type["lab"],
            seminarRoomCapacity=room_capacity_by_type["seminar"],
            configuredSectionCapacity=configured_section_capacity,
            scheduledStudentSeats=scheduled_student_seats,
        ),
        activity=ActivityAnalyticsOut(
            windowDays=window_days,
            totalLogs=total_logs,
            actionsLastWindow=len(window_logs),
            activeUsers=len(active_user_ids),
            actionsByDay=[
                DailyCountPoint(date=day, value=count)
                for day, count in sorted(actions_by_day_counter.items())
            ],
            topActions=_to_labeled_counts(dict(top_actions_counter), limit=8),
            topEntities=_to_labeled_counts(dict(top_entities_counter), limit=8),
            recentLogs=recent_logs,
        ),
        operations=OperationsSnapshotOut(
            unreadNotifications=unread_notifications,
            notificationsByType=_to_labeled_counts(notification_type_counts),
            leavesByStatus=_to_labeled_counts(leave_status_counts),
            issuesByStatus=_to_labeled_counts(issue_status_counts),
            feedbackByStatus=_to_labeled_counts(feedback_status_counts),
        ),
        metricDefinitions=metric_definitions,
    )


@router.post("/system/backup")
def trigger_backup(
    current_user: User = Depends(require_roles(UserRole.admin)),
    db: Session = Depends(get_db),
) -> dict:
    backup_dir = Path("database/backups")
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = backup_dir / f"shedforge-backup-{timestamp}.json"
    official = db.get(OfficialTimetable, 1)

    data = {
        "timestamp": timestamp,
        "programs": [item.id for item in db.execute(select(Program.id)).scalars()],
        "courses": [item.id for item in db.execute(select(Course.id)).scalars()],
        "rooms": [item.id for item in db.execute(select(Room.id)).scalars()],
        "faculty": [item.id for item in db.execute(select(Faculty.id)).scalars()],
        "official_timetable": official.payload if official else None,
    }
    backup_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    notify_all_users(
        db,
        title="System Backup Completed",
        message=f"Backup {backup_path.name} was created by {current_user.name}.",
        notification_type=NotificationType.system,
        deliver_email=True,
    )
    log_activity(
        db,
        user=current_user,
        action="system.backup",
        entity_type="backup",
        entity_id=backup_path.name,
        details={"path": str(backup_path)},
    )
    db.commit()
    return {"success": True, "backup_file": str(backup_path)}
