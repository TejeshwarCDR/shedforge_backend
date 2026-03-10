from __future__ import annotations

import math
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db, require_roles
from app.models.course import Course
from app.models.faculty import Faculty
from app.models.institution_settings import InstitutionSettings
from app.models.program import Program
from app.models.program_constraint import ProgramConstraint
from app.models.program_structure import ProgramCourse, ProgramTerm
from app.models.semester_constraint import SemesterConstraint
from app.models.user import User, UserRole
from app.schemas.constraints import (
    ConstraintViolation,
    ProgramConstraintOut,
    ProgramConstraintReport,
    ProgramConstraintUpsert,
    ProgramDailyTimeSlot,
    SemesterConstraintOut,
    SemesterConstraintUpsert,
)
from app.schemas.settings import (
    DEFAULT_SCHEDULE_POLICY,
    DEFAULT_WORKING_HOURS,
    SchedulePolicyUpdate,
    WorkingHoursEntry,
    parse_time_to_minutes,
)
from app.services.notifications import notify_admin_update

router = APIRouter()

LUNCH_START_MINUTES = parse_time_to_minutes("13:15")
LUNCH_END_MINUTES = parse_time_to_minutes("14:05")
REMOVED_DAILY_SLOT_RANGES: set[tuple[int, int]] = {
    (parse_time_to_minutes("10:45"), parse_time_to_minutes("11:20")),
    (parse_time_to_minutes("11:20"), parse_time_to_minutes("12:10")),
    (parse_time_to_minutes("12:10"), parse_time_to_minutes("13:00")),
    (parse_time_to_minutes("14:40"), parse_time_to_minutes("15:30")),
    (parse_time_to_minutes("15:30"), parse_time_to_minutes("16:20")),
    (parse_time_to_minutes("16:20"), parse_time_to_minutes("16:35")),
}


def _is_removed_daily_slot_range(start: int, end: int) -> bool:
    return (start, end) in REMOVED_DAILY_SLOT_RANGES


def _is_canonical_lunch_slot(start: int, end: int) -> bool:
    return start == LUNCH_START_MINUTES and end == LUNCH_END_MINUTES


def _overlaps_lunch_window(start: int, end: int) -> bool:
    return start < LUNCH_END_MINUTES and end > LUNCH_START_MINUTES


def _sanitize_program_daily_time_slots(raw_slots: list[dict] | list[ProgramDailyTimeSlot] | None) -> list[dict]:
    sanitized_by_key: dict[tuple[int, int], ProgramDailyTimeSlot] = {}
    has_canonical_lunch = False
    has_teaching_slots = False

    def read_value(entry: dict | ProgramDailyTimeSlot, key: str, default: str = "") -> str:
        if isinstance(entry, dict):
            return str(entry.get(key, default))
        value = getattr(entry, key, default)
        return str(value if value is not None else default)

    for item in raw_slots or []:
        start_time = read_value(item, "start_time").strip()
        end_time = read_value(item, "end_time").strip()
        try:
            start = parse_time_to_minutes(start_time)
            end = parse_time_to_minutes(end_time)
        except Exception:
            continue
        if end <= start:
            continue
        if _is_removed_daily_slot_range(start, end):
            continue
        if _overlaps_lunch_window(start, end) and not _is_canonical_lunch_slot(start, end):
            continue

        raw_tag = read_value(item, "tag", "teaching").strip().lower()
        if raw_tag not in {"teaching", "block", "break", "lunch"}:
            raw_tag = "teaching"
        label = read_value(item, "label").strip() or None

        if _is_canonical_lunch_slot(start, end):
            raw_tag = "lunch"
            label = "Lunch Break"
            has_canonical_lunch = True
        elif raw_tag == "lunch":
            # Keep lunch as a single canonical block only.
            continue
        elif raw_tag == "teaching":
            has_teaching_slots = True

        sanitized_by_key[(start, end)] = ProgramDailyTimeSlot(
            start_time=f"{start // 60:02d}:{start % 60:02d}",
            end_time=f"{end // 60:02d}:{end % 60:02d}",
            tag=raw_tag,  # type: ignore[arg-type]
            label=label,
        )

    if has_teaching_slots and not has_canonical_lunch:
        sanitized_by_key[(LUNCH_START_MINUTES, LUNCH_END_MINUTES)] = ProgramDailyTimeSlot(
            start_time="13:15",
            end_time="14:05",
            tag="lunch",
            label="Lunch Break",
        )

    ordered = sorted(sanitized_by_key.values(), key=lambda slot: parse_time_to_minutes(slot.start_time))
    return [slot.model_dump() for slot in ordered]


def _computed_course_credits(course: Course) -> float:
    lecture = max(0, int(course.theory_hours or 0))
    tutorial = max(0, int(course.tutorial_hours or 0))
    practical = max(0, int(course.lab_hours or 0))
    return float(lecture + tutorial + (practical / 2.0))


def _weekly_contact_units(course: Course) -> int:
    lecture = max(0, int(course.theory_hours or 0))
    tutorial = max(0, int(course.tutorial_hours or 0))
    practical = max(0, int(course.lab_hours or 0))
    split_sum = lecture + tutorial + practical
    if split_sum > 0:
        return split_sum
    return max(0, int(course.hours_per_week or 0))


def _load_defaults_from_settings(db: Session) -> tuple[list[WorkingHoursEntry], SchedulePolicyUpdate]:
    record = db.get(InstitutionSettings, 1)
    if record is None:
        return DEFAULT_WORKING_HOURS, DEFAULT_SCHEDULE_POLICY

    working_hours = [WorkingHoursEntry.model_validate(item) for item in (record.working_hours or [])]
    if not working_hours:
        working_hours = DEFAULT_WORKING_HOURS
    schedule_policy = SchedulePolicyUpdate(
        period_minutes=record.period_minutes or DEFAULT_SCHEDULE_POLICY.period_minutes,
        lab_contiguous_slots=record.lab_contiguous_slots or DEFAULT_SCHEDULE_POLICY.lab_contiguous_slots,
        breaks=record.break_windows or [item.model_dump() for item in DEFAULT_SCHEDULE_POLICY.breaks],
    )
    return working_hours, schedule_policy


def _build_default_daily_time_slots(db: Session) -> list[dict]:
    working_hours, schedule_policy = _load_defaults_from_settings(db)
    first_enabled = next((entry for entry in working_hours if entry.enabled), None)
    if first_enabled is None:
        return []

    day_start = parse_time_to_minutes(first_enabled.start_time)
    day_end = parse_time_to_minutes(first_enabled.end_time)
    period = max(1, int(schedule_policy.period_minutes))

    breaks: list[tuple[str, int, int]] = []
    for item in schedule_policy.breaks:
        start = parse_time_to_minutes(item.start_time)
        end = parse_time_to_minutes(item.end_time)
        if end <= day_start or start >= day_end:
            continue
        breaks.append((item.name, max(day_start, start), min(day_end, end)))
    breaks.sort(key=lambda item: item[1])

    boundaries: set[int] = {day_start, day_end}
    cursor = day_start
    while cursor < day_end:
        boundaries.add(cursor)
        cursor += period
    for _name, start, end in breaks:
        boundaries.add(start)
        boundaries.add(end)

    sorted_points = sorted(boundaries)
    slots: list[dict] = []
    for index in range(len(sorted_points) - 1):
        start = sorted_points[index]
        end = sorted_points[index + 1]
        if end <= start:
            continue
        overlap_break = next(
            (
                (name, break_start, break_end)
                for name, break_start, break_end in breaks
                if start < break_end and end > break_start
            ),
            None,
        )
        if overlap_break is None:
            slot = ProgramDailyTimeSlot(
                start_time=f"{start // 60:02d}:{start % 60:02d}",
                end_time=f"{end // 60:02d}:{end % 60:02d}",
                tag="teaching",
            )
        else:
            break_name = overlap_break[0].strip() or "Break"
            tag = "lunch" if "lunch" in break_name.lower() else "break"
            slot = ProgramDailyTimeSlot(
                start_time=f"{start // 60:02d}:{start % 60:02d}",
                end_time=f"{end // 60:02d}:{end % 60:02d}",
                tag=tag,
                label=break_name,
            )
        slots.append(slot.model_dump())
    return _sanitize_program_daily_time_slots(slots)


def _to_program_constraint_out(record: ProgramConstraint | None, *, db: Session, program_id: str) -> ProgramConstraintOut:
    if record is not None:
        out = ProgramConstraintOut.model_validate(record, from_attributes=True)
        out.daily_time_slots = _sanitize_program_daily_time_slots(out.daily_time_slots)
        return out
    return ProgramConstraintOut(
        id=f"default-{program_id}",
        program_id=program_id,
        daily_time_slots=_sanitize_program_daily_time_slots(_build_default_daily_time_slots(db)),
        faculty_min_hours_per_week=14,
        faculty_max_hours_per_week=20,
        temporal_window_semesters=3,
        auto_assign_research_slots=True,
        enforce_student_credit_load=True,
        enforce_ltp_split=True,
        enforce_lab_contiguous_blocks=True,
        updated_at=None,
    )


@router.get("/constraints/semesters", response_model=list[SemesterConstraintOut])
def list_semester_constraints(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[SemesterConstraintOut]:
    constraints = db.execute(select(SemesterConstraint).order_by(SemesterConstraint.term_number)).scalars().all()
    return list(constraints)


@router.get("/constraints/semesters/{term_number}", response_model=SemesterConstraintOut)
def get_semester_constraint(
    term_number: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SemesterConstraintOut:
    constraint = (
        db.execute(select(SemesterConstraint).where(SemesterConstraint.term_number == term_number))
        .scalars()
        .first()
    )
    if constraint is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Semester constraint not found")
    return constraint


@router.put("/constraints/semesters/{term_number}", response_model=SemesterConstraintOut)
def upsert_semester_constraint(
    term_number: int,
    payload: SemesterConstraintUpsert,
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> SemesterConstraintOut:
    if payload.term_number != term_number:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Term number mismatch")

    constraint = (
        db.execute(select(SemesterConstraint).where(SemesterConstraint.term_number == term_number))
        .scalars()
        .first()
    )
    data = payload.model_dump()
    if constraint is None:
        constraint = SemesterConstraint(**data)
        db.add(constraint)
        notify_admin_update(
            db,
            title="Semester Constraint Added",
            message=f"{current_user.name} added semester {term_number} scheduling constraints.",
            actor_user_id=current_user.id,
        )
    else:
        for key, value in data.items():
            setattr(constraint, key, value)
        notify_admin_update(
            db,
            title="Semester Constraint Updated",
            message=f"{current_user.name} updated semester {term_number} scheduling constraints.",
            actor_user_id=current_user.id,
        )

    db.commit()
    db.refresh(constraint)
    return constraint


@router.delete("/constraints/semesters/{term_number}", status_code=status.HTTP_204_NO_CONTENT)
def delete_semester_constraint(
    term_number: int,
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> None:
    constraint = (
        db.execute(select(SemesterConstraint).where(SemesterConstraint.term_number == term_number))
        .scalars()
        .first()
    )
    if constraint is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Semester constraint not found")
    notify_admin_update(
        db,
        title="Semester Constraint Removed",
        message=f"{current_user.name} removed semester {term_number} scheduling constraints.",
        actor_user_id=current_user.id,
    )
    db.delete(constraint)
    db.commit()


@router.get("/constraints/programs", response_model=list[ProgramConstraintOut])
def list_program_constraints(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[ProgramConstraintOut]:
    constraints = db.execute(select(ProgramConstraint).order_by(ProgramConstraint.program_id.asc())).scalars().all()
    return [_to_program_constraint_out(item, db=db, program_id=item.program_id) for item in constraints]


@router.get("/constraints/programs/{program_id}", response_model=ProgramConstraintOut)
def get_program_constraint(
    program_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ProgramConstraintOut:
    program = db.get(Program, program_id)
    if program is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Program not found")
    record = (
        db.execute(select(ProgramConstraint).where(ProgramConstraint.program_id == program_id))
        .scalars()
        .first()
    )
    return _to_program_constraint_out(record, db=db, program_id=program_id)


@router.put("/constraints/programs/{program_id}", response_model=ProgramConstraintOut)
def upsert_program_constraint(
    program_id: str,
    payload: ProgramConstraintUpsert,
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> ProgramConstraintOut:
    if payload.program_id != program_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Program ID mismatch")
    program = db.get(Program, program_id)
    if program is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Program not found")

    record = (
        db.execute(select(ProgramConstraint).where(ProgramConstraint.program_id == program_id))
        .scalars()
        .first()
    )
    data = payload.model_dump()
    data["daily_time_slots"] = _sanitize_program_daily_time_slots(data.get("daily_time_slots"))
    if record is None:
        record = ProgramConstraint(**data)
        db.add(record)
        notify_admin_update(
            db,
            title="Program Constraints Added",
            message=f"{current_user.name} configured scheduling constraints for program {program.code}.",
            actor_user_id=current_user.id,
        )
    else:
        for key, value in data.items():
            setattr(record, key, value)
        notify_admin_update(
            db,
            title="Program Constraints Updated",
            message=f"{current_user.name} updated scheduling constraints for program {program.code}.",
            actor_user_id=current_user.id,
        )

    db.commit()
    db.refresh(record)
    return ProgramConstraintOut.model_validate(record, from_attributes=True)


@router.get("/constraints/programs/{program_id}/report", response_model=ProgramConstraintReport)
def get_program_constraint_report(
    program_id: str,
    term_number: int | None = Query(default=None, alias="termNumber", ge=1, le=20),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ProgramConstraintReport:
    program = db.get(Program, program_id)
    if program is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Program not found")

    raw_constraint = (
        db.execute(select(ProgramConstraint).where(ProgramConstraint.program_id == program_id))
        .scalars()
        .first()
    )
    constraint = _to_program_constraint_out(raw_constraint, db=db, program_id=program_id)

    violations: list[ConstraintViolation] = []
    if not constraint.daily_time_slots:
        violations.append(
            ConstraintViolation(
                code="daily_time_slots_missing",
                severity="warn",
                message="No program daily time slots are configured. Scheduler will fallback to institution settings.",
            )
        )

    term_stmt = select(ProgramTerm).where(ProgramTerm.program_id == program_id)
    if term_number is not None:
        term_stmt = term_stmt.where(ProgramTerm.term_number == term_number)
    terms = db.execute(term_stmt).scalars().all()
    terms_by_number = {item.term_number: item for item in terms}
    term_numbers = sorted(terms_by_number.keys())

    if not term_numbers and term_number is not None:
        violations.append(
            ConstraintViolation(
                code="term_not_configured",
                severity="hard",
                term_number=term_number,
                message=f"Term {term_number} is not configured for this program.",
            )
        )

    if term_numbers:
        program_courses = (
            db.execute(
                select(ProgramCourse).where(
                    ProgramCourse.program_id == program_id,
                    ProgramCourse.term_number.in_(term_numbers),
                )
            )
            .scalars()
            .all()
        )
    else:
        program_courses = []
    program_courses_by_term: dict[int, list[ProgramCourse]] = {}
    for item in program_courses:
        program_courses_by_term.setdefault(item.term_number, []).append(item)

    all_course_ids = sorted({item.course_id for item in program_courses})
    if all_course_ids:
        loaded_courses = db.execute(select(Course).where(Course.id.in_(all_course_ids))).scalars().all()
    else:
        loaded_courses = []
    course_by_id = {item.id: item for item in loaded_courses}

    for term_no in term_numbers:
        rows = program_courses_by_term.get(term_no, [])
        if not rows:
            violations.append(
                ConstraintViolation(
                    code="term_courses_missing",
                    severity="hard",
                    term_number=term_no,
                    message=f"Term {term_no} has no mapped courses.",
                )
            )
            continue

        required_rows = [row for row in rows if row.is_required]
        active_rows = required_rows if required_rows else rows

        computed_term_credits = 0.0
        total_contact_units = 0
        for mapping in active_rows:
            course = course_by_id.get(mapping.course_id)
            if course is None:
                violations.append(
                    ConstraintViolation(
                        code="course_mapping_missing",
                        severity="hard",
                        term_number=term_no,
                        course_id=mapping.course_id,
                        message=f"Program course mapping references missing course {mapping.course_id}.",
                    )
                )
                continue

            total_contact_units += _weekly_contact_units(course)
            computed_term_credits += _computed_course_credits(course)

            if constraint.enforce_ltp_split:
                split_total = max(0, int(course.theory_hours or 0)) + max(0, int(course.tutorial_hours or 0)) + max(0, int(course.lab_hours or 0))
                if split_total != max(0, int(course.hours_per_week or 0)):
                    violations.append(
                        ConstraintViolation(
                            code="ltp_split_mismatch",
                            severity="hard",
                            term_number=term_no,
                            course_id=course.id,
                            message=(
                                f"Course {course.code} has LTP split {split_total}h but hours_per_week "
                                f"is {course.hours_per_week}h."
                            ),
                        )
                    )

            if constraint.enforce_lab_contiguous_blocks:
                lab_hours = max(0, int(course.lab_hours or 0))
                together = max(1, int(course.practical_contiguous_slots or 1))
                if lab_hours > 0 and together > lab_hours:
                    violations.append(
                        ConstraintViolation(
                            code="lab_together_exceeds_practical",
                            severity="hard",
                            term_number=term_no,
                            course_id=course.id,
                            message=(
                                f"Course {course.code} has together-count {together} but practical "
                                f"hours are only {lab_hours}."
                            ),
                        )
                    )
                if lab_hours <= 0 and together != 1:
                    violations.append(
                        ConstraintViolation(
                            code="lab_together_invalid_for_theory",
                            severity="warn",
                            term_number=term_no,
                            course_id=course.id,
                            message=(
                                f"Course {course.code} has no practical hours but together-count is {together}. "
                                "Set it to 1 for non-lab courses."
                            ),
                        )
                    )

        if constraint.enforce_student_credit_load:
            term = terms_by_number.get(term_no)
            required_credits = float(term.credits_required if term is not None else 0.0)
            if required_credits > 0.0 and not math.isclose(computed_term_credits, required_credits, abs_tol=0.01):
                violations.append(
                    ConstraintViolation(
                        code="student_credit_load_mismatch",
                        severity="hard",
                        term_number=term_no,
                        message=(
                            f"Term {term_no} configured credits ({required_credits:g}) do not match "
                            f"computed credits from LTP ({computed_term_credits:.2f}). "
                            f"Computed weekly contact hours: {total_contact_units}."
                        ),
                    )
                )

    faculty_rows = db.execute(select(Faculty).where(Faculty.program_id == program_id)).scalars().all()
    for faculty in faculty_rows:
        if faculty.max_hours > constraint.faculty_max_hours_per_week:
            violations.append(
                ConstraintViolation(
                    code="faculty_max_exceeds_program_limit",
                    severity="warn",
                    faculty_id=faculty.id,
                    message=(
                        f"Faculty {faculty.name} max hours ({faculty.max_hours}) exceed program limit "
                        f"({constraint.faculty_max_hours_per_week})."
                    ),
                )
            )

        configured_target = max(0, int(faculty.workload_hours or 0))
        if configured_target > 0 and configured_target < constraint.faculty_min_hours_per_week:
            violations.append(
                ConstraintViolation(
                    code="faculty_min_target_not_met",
                    severity="warn",
                    faculty_id=faculty.id,
                    message=(
                        f"Faculty {faculty.name} configured workload target ({configured_target}) is below "
                        f"program minimum ({constraint.faculty_min_hours_per_week})."
                    ),
                )
            )

    return ProgramConstraintReport(
        program_id=program_id,
        generated_at=datetime.now(timezone.utc),
        violation_count=len(violations),
        violations=violations,
    )
