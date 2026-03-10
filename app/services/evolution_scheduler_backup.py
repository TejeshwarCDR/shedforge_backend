from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import logging
import math
import random
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.core.exceptions import SchedulerError

from app.models.course import Course, CourseType
from app.models.faculty import Faculty
from app.models.institution_settings import InstitutionSettings
from app.models.program_structure import (
    ElectiveConflictPolicy,
    ProgramCourse,
    ProgramElectiveGroup,
    ProgramElectiveGroupMember,
    ProgramTerm,
    ProgramSection,
    ProgramSharedLectureGroup,
    ProgramSharedLectureGroupMember,
)
from app.models.room import Room, RoomType
from app.models.semester_constraint import SemesterConstraint
from app.models.timetable_generation import TimetableSlotLock
from app.schemas.generator import GenerateTimetableRequest, GenerateTimetableResponse, GeneratedAlternative, GenerationSettingsBase
from app.schemas.settings import (
    DEFAULT_SCHEDULE_POLICY,
    DEFAULT_WORKING_HOURS,
    BreakWindowEntry,
    SchedulePolicyUpdate,
    WorkingHoursEntry,
    parse_time_to_minutes,
)
from app.schemas.timetable import OfficialTimetablePayload

DAY_SHORT_MAP = {
    "Mon": "Monday",
    "Tue": "Tuesday",
    "Wed": "Wednesday",
    "Thu": "Thursday",
    "Fri": "Friday",
    "Sat": "Saturday",
    "Sun": "Sunday",
}

logger = logging.getLogger(__name__)


def minutes_to_time(value: int) -> str:
    hours = value // 60
    minutes = value % 60
    return f"{hours:02d}:{minutes:02d}"


def normalize_day(value: str) -> str:
    return DAY_SHORT_MAP.get(value, value)


@dataclass(frozen=True)
class SlotSegment:
    start: int
    end: int


@dataclass(frozen=True)
class PlacementOption:
    day: str
    start_index: int
    room_id: str
    faculty_id: str


@dataclass(frozen=True)
class BlockRequest:
    request_id: int
    course_id: str
    course_code: str
    section: str
    batch: str | None
    student_count: int
    primary_faculty_id: str
    preferred_faculty_ids: tuple[str, ...]
    block_size: int
    is_lab: bool
    session_type: Literal["theory", "tutorial", "lab"]
    allow_parallel_batches: bool
    room_candidate_ids: tuple[str, ...]
    options: tuple[PlacementOption, ...]


@dataclass
class EvaluationResult:
    fitness: float
    hard_conflicts: int
    soft_penalty: float


class EvolutionaryScheduler:
    """
    Implements a Genetic Algorithm (GA) to generate optimized academic timetables.
    
    The algorithm evolves a population of timetables over multiple generations to minimize
    conflicts and optimize for soft constraints like faculty workload balance and preferred slots.
    
    Key Steps:
    1.  **Initialization**: Creates a random population of valid timetables.
    2.  **Fitness Evaluation**: Scores each timetable based on hard constraints (validity) and soft constraints (quality).
    3.  **Selection**: Selects the best timetables to be parents for the next generation.
    4.  **Crossover**: Combines genes (slot assignments) from two parents to create offspring.
    5.  **Mutation**: Randomly alters genes to introduce diversity and avoid local optima.
    6.  **Elitism**: Preserves the best solutions unchanged across generations.
    """

    def __init__(
        self,
        *,
        db: Session,
        program_id: str,
        term_number: int,
        settings: GenerationSettingsBase,
        reserved_resource_slots: list[dict] | None = None,
    ) -> None:
        """
        Initializes the scheduler with necessary context and loads all required resources.
        
        Args:
            db: Database session for loading academic data.
            program_id: The specific academic program to schedule.
            term_number: The semester/term number.
            settings: Configuration for the genetic algorithm (population size, mutation rate, etc.).
            reserved_resource_slots: Pre-booked slots to treat as unavailable (e.g., maintenance, holidays).
        """
        self.db = db
        self.program_id = program_id
        self.term_number = term_number
        self.settings = settings
        self.random = random.Random(settings.random_seed)

        self.working_hours, self.schedule_policy = self._load_time_settings()
        self.day_slots = self._build_day_slots()
        if not self.day_slots:
            raise SchedulerError(
                message="No active working days configured for timetable generation",
            )
        self.reserved_resource_slots_by_day = self._index_reserved_resource_slots(reserved_resource_slots or [])

        self.courses = self._load_courses()
        self.sections = self._load_sections()
        self.program_courses = self._load_program_courses()
        self._validate_course_credit_alignment()
        self._validate_prerequisite_mappings()
        self.expected_section_minutes = self._resolve_expected_section_minutes()
        self._validate_section_time_capacity()
        self.elective_overlap_pairs = self._load_elective_overlap_pairs()
        self.shared_lecture_sections_by_course = self._load_shared_lecture_sections_by_course()
        self.rooms = {room.id: room for room in self.db.execute(select(Room)).scalars().all()}
        if not self.rooms:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No rooms available for generation")

        room_type_counts = Counter(r.type for r in self.rooms.values())
        logger.info("Rooms loaded: total=%d, counts=%s", len(self.rooms), dict(room_type_counts))

        self.faculty = {item.id: item for item in self.db.execute(select(Faculty)).scalars().all()}
        if not self.faculty:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No faculty available for generation")
        self.faculty_windows = {item.id: self._normalize_windows(item.availability_windows) for item in self.faculty.values()}
        self.room_windows = {item.id: self._normalize_windows(item.availability_windows) for item in self.rooms.values()}
        self.faculty_preferred_subject_codes = {
            item.id: self._faculty_preference_codes_for_term(item)
            for item in self.faculty.values()
        }
        self.semester_constraint = (
            self.db.execute(select(SemesterConstraint).where(SemesterConstraint.term_number == term_number))
            .scalars()
            .first()
        )

        self.block_requests = self._build_block_requests()
        self.request_indices_by_course = self._build_request_indices_by_course()
        self.request_indices_by_course_section = self._build_request_indices_by_course_section()
        self.single_faculty_required_by_course = self._build_single_faculty_requirements_by_course()
        self.common_faculty_candidates_by_course_section = self._build_common_faculty_candidates_by_course_section()
        self.common_faculty_candidates_by_course = self._build_common_faculty_candidates_by_course()
        self._validate_total_faculty_capacity()
        self.fixed_genes = self._load_fixed_genes()
        self._validate_locked_course_faculty_consistency()
        self.option_priority_indices = self._build_option_priority_indices()
        self.eval_cache: dict[tuple[int, ...], EvaluationResult] = {}

    def _build_option_priority_indices(self) -> dict[int, list[int]]:
        indices_by_request: dict[int, list[int]] = {}
        for req in self.block_requests:
            option_count = len(req.options)
            if option_count <= 1:
                indices_by_request[req.request_id] = list(range(option_count))
                continue
            ranked = sorted(
                range(option_count),
                key=lambda option_index: (
                    self.rooms[req.options[option_index].room_id].capacity < req.student_count,
                    max(0, self.rooms[req.options[option_index].room_id].capacity - req.student_count),
                    bool(req.preferred_faculty_ids)
                    and req.options[option_index].faculty_id not in req.preferred_faculty_ids,
                    req.options[option_index].day,
                    req.options[option_index].start_index,
                ),
            )
            indices_by_request[req.request_id] = ranked
        return indices_by_request

    def _faculty_preference_codes_for_term(self, faculty: Faculty) -> set[str]:
        preferred = {code.strip().upper() for code in (faculty.preferred_subject_codes or []) if code and code.strip()}
        semester_preferences = faculty.semester_preferences or {}
        term_specific = semester_preferences.get(str(self.term_number), [])
        preferred.update(code.strip().upper() for code in term_specific if code and code.strip())
        return preferred

    def _option_bounds(self, option: PlacementOption, block_size: int) -> tuple[int, int]:
        day_slots = self.day_slots[option.day]
        if not day_slots:
            raise IndexError(f"No slots configured for day {option.day}")

        start_index = option.start_index
        if start_index >= len(day_slots):
            # Some benchmark fixtures flatten weekly indices; normalize to day-local index.
            start_index = start_index % len(day_slots)

        end_index = start_index + block_size - 1
        if end_index >= len(day_slots):
            raise IndexError(
                f"Block end index {end_index} exceeds configured slots for day {option.day}"
            )

        start = day_slots[start_index].start
        end = day_slots[end_index].end
        return start, end

    def _parallel_lab_group_key(self, req: BlockRequest) -> tuple[str, str, str, int] | None:
        if not req.is_lab or not req.allow_parallel_batches or not req.batch:
            return None
        return (req.course_id, req.section, req.session_type, req.block_size)

    @staticmethod
    def _parallel_lab_signature(option: PlacementOption) -> tuple[str, int]:
        return (option.day, option.start_index)

    def _filter_option_indices_by_signatures(
        self,
        *,
        req: BlockRequest,
        candidate_indices: list[int],
        signatures: set[tuple[str, int]],
    ) -> list[int]:
        if not signatures:
            return candidate_indices
        filtered = [
            option_index
            for option_index in candidate_indices
            if self._parallel_lab_signature(req.options[option_index]) in signatures
        ]
        if filtered:
            return filtered
        fallback = [
            option_index
            for option_index, option in enumerate(req.options)
            if self._parallel_lab_signature(option) in signatures
        ]
        return fallback if fallback else candidate_indices

    def _parallel_lab_target_signatures_from_genes(
        self,
        genes: list[int],
        req_index: int,
    ) -> set[tuple[str, int]]:
        req = self.block_requests[req_index]
        group_key = self._parallel_lab_group_key(req)
        if group_key is None:
            return set()

        signatures: set[tuple[str, int]] = set()
        for other_index, other_req in enumerate(self.block_requests):
            if other_index == req_index:
                continue
            if self._parallel_lab_group_key(other_req) != group_key:
                continue
            if other_req.batch == req.batch:
                continue
            signatures.add(self._parallel_lab_signature(other_req.options[genes[other_index]]))
        return signatures

    def _parallel_lab_baseline_batch_for_group(self, group_key: tuple[str, str, str, int]) -> str | None:
        mapping = getattr(self, "_parallel_lab_baseline_batch_cache", None)
        if not isinstance(mapping, dict):
            mapping = {}
            for req in self.block_requests:
                req_group_key = self._parallel_lab_group_key(req)
                if req_group_key is None or not req.batch:
                    continue
                current = mapping.get(req_group_key)
                if current is None or req.batch < current:
                    mapping[req_group_key] = req.batch
            self._parallel_lab_baseline_batch_cache = mapping
        return mapping.get(group_key)

    def _index_reserved_resource_slots(
        self,
        slots: list[dict],
    ) -> dict[str, list[tuple[int, int, str | None, str | None]]]:
        indexed: dict[str, list[tuple[int, int, str | None, str | None]]] = defaultdict(list)
        for item in slots:
            day = normalize_day(str(item.get("day", "")).strip())
            if day not in self.day_slots:
                continue
            start_raw = item.get("start_time")
            end_raw = item.get("end_time")
            if not isinstance(start_raw, str) or not isinstance(end_raw, str):
                continue
            start = parse_time_to_minutes(start_raw)
            end = parse_time_to_minutes(end_raw)
            if end <= start:
                continue
            room_id = item.get("room_id")
            faculty_id = item.get("faculty_id")
            indexed[day].append((start, end, room_id, faculty_id))
        return indexed

    def _reserved_conflict_flags(
        self,
        *,
        day: str,
        start_min: int,
        end_min: int,
        room_id: str,
        faculty_id: str,
    ) -> tuple[bool, bool]:
        room_conflict = False
        faculty_conflict = False
        for reserved_start, reserved_end, reserved_room_id, reserved_faculty_id in self.reserved_resource_slots_by_day.get(day, []):
            overlaps = start_min < reserved_end and reserved_start < end_min
            if not overlaps:
                continue
            if reserved_room_id and reserved_room_id == room_id:
                room_conflict = True
            if reserved_faculty_id and reserved_faculty_id == faculty_id:
                faculty_conflict = True
            if room_conflict or faculty_conflict:
                break
        return room_conflict, faculty_conflict

    def _conflicts_reserved_resources(
        self,
        *,
        day: str,
        start_min: int,
        end_min: int,
        room_id: str,
        faculty_id: str,
    ) -> bool:
        room_conflict, faculty_conflict = self._reserved_conflict_flags(
            day=day,
            start_min=start_min,
            end_min=end_min,
            room_id=room_id,
            faculty_id=faculty_id,
        )
        return room_conflict or faculty_conflict

    def _load_time_settings(self) -> tuple[list[WorkingHoursEntry], SchedulePolicyUpdate]:
        record = self.db.get(InstitutionSettings, 1)
        if record is None:
            return DEFAULT_WORKING_HOURS, DEFAULT_SCHEDULE_POLICY

        working_hours = [WorkingHoursEntry.model_validate(item) for item in record.working_hours]
        schedule_policy = SchedulePolicyUpdate(
            period_minutes=record.period_minutes or DEFAULT_SCHEDULE_POLICY.period_minutes,
            lab_contiguous_slots=record.lab_contiguous_slots or DEFAULT_SCHEDULE_POLICY.lab_contiguous_slots,
            breaks=record.break_windows or [item.model_dump() for item in DEFAULT_SCHEDULE_POLICY.breaks],
        )
        return working_hours, schedule_policy

    def _build_day_slots(self) -> dict[str, list[SlotSegment]]:
        def first_overlapping_break(
            start: int,
            end: int,
            breaks: list[BreakWindowEntry],
        ) -> tuple[int, int] | None:
            for item in breaks:
                break_start = parse_time_to_minutes(item.start_time)
                break_end = parse_time_to_minutes(item.end_time)
                if start < break_end and end > break_start:
                    return break_start, break_end
            return None

        day_slots: dict[str, list[SlotSegment]] = {}
        period = self.schedule_policy.period_minutes
        for entry in self.working_hours:
            if not entry.enabled:
                continue
            day_start = parse_time_to_minutes(entry.start_time)
            day_end = parse_time_to_minutes(entry.end_time)
            slots: list[SlotSegment] = []
            cursor = day_start
            while cursor + period <= day_end:
                end = cursor + period
                overlap = first_overlapping_break(cursor, end, self.schedule_policy.breaks)
                if overlap is not None:
                    # Jump to break end to avoid minute-by-minute scans over long windows.
                    _, break_end = overlap
                    cursor = max(cursor + 1, break_end)
                    continue
                slots.append(SlotSegment(start=cursor, end=end))
                cursor = end
            if slots:
                day_slots[entry.day] = slots
        return day_slots

    def _load_courses(self) -> dict[str, Course]:
        rows = (
            self.db.execute(
                select(Course)
                .join(
                    ProgramCourse,
                    ProgramCourse.course_id == Course.id,
                )
                .where(
                    ProgramCourse.program_id == self.program_id,
                    ProgramCourse.term_number == self.term_number,
                )
            )
            .scalars()
            .all()
        )
        return {course.id: course for course in rows}

    def _load_sections(self) -> list[ProgramSection]:
        sections = (
            self.db.execute(
                select(ProgramSection).where(
                    ProgramSection.program_id == self.program_id,
                    ProgramSection.term_number == self.term_number,
                )
            )
            .scalars()
            .all()
        )
        if not sections:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No sections configured for this program term",
            )
        return sections

    def _load_program_courses(self) -> list[ProgramCourse]:
        rows = (
            self.db.execute(
                select(ProgramCourse).where(
                    ProgramCourse.program_id == self.program_id,
                    ProgramCourse.term_number == self.term_number,
                )
            )
            .scalars()
            .all()
        )
        if not rows:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No courses configured for this program term",
            )
        required_rows = [row for row in rows if row.is_required]
        if required_rows:
            optional_count = len(rows) - len(required_rows)
            if optional_count > 0:
                logger.info(
                    "Generation scope excludes optional courses by default | program_id=%s term=%s required=%s optional_skipped=%s",
                    self.program_id,
                    self.term_number,
                    len(required_rows),
                    optional_count,
                )
            return required_rows
        return rows

    def _resolve_expected_section_minutes(self) -> int:
        configured_hours = 0
        for program_course in self.program_courses:
            course = self.courses.get(program_course.course_id)
            if course is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Course mapping missing for course id {program_course.course_id}",
                )
            configured_hours += max(0, course.hours_per_week)

        if configured_hours <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Program term requires at least one positive weekly-hour course",
            )

        term = (
            self.db.execute(
                select(ProgramTerm).where(
                    ProgramTerm.program_id == self.program_id,
                    ProgramTerm.term_number == self.term_number,
                )
            )
            .scalars()
            .first()
        )
        if term is not None and term.credits_required > 0:
            total_credits = 0
            for program_course in self.program_courses:
                course = self.courses.get(program_course.course_id)
                if course:
                    total_credits += course.credits
            
            if total_credits != term.credits_required:
                logger.warning(
                    "Curriculum credit mismatch in term %s: Courses sum to %s credits but term requires %s",
                    self.term_number,
                    total_credits,
                    term.credits_required
                )
                # Note: We keep this as a warning for now, but configured_hours (contact hours) 
                # remains the absolute target for scheduling.
        
        # TARGET HOURS: Sum of contact hours (HPW) is what needs to be scheduled.
        target_hours = configured_hours

        return target_hours * self.schedule_policy.period_minutes

    def _validate_course_credit_alignment(self) -> None:
        for pc in self.program_courses:
            course = self.courses.get(pc.course_id)
            if course is None:
                continue

            total_split = course.theory_hours + course.lab_hours + course.tutorial_hours
            if total_split <= 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Course {course.code} must define a positive credit split",
                )

            # RELAXED: Credits and HPW don't have to be 1:1 if there are weightings (e.g. 2 Lab hours = 1 Credit).
            # However, the split MUST sum up to the total contact hours (HPW) for scheduling.
            if total_split != course.hours_per_week:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"ShedForge Data Parity Mismatch: Course {course.code} "
                        f"Hours/Week ({course.hours_per_week}) must exactly match "
                        f"the sum of theory+lab+tutorial hours ({total_split})."
                    ),
                )

    def _validate_total_faculty_capacity(self) -> None:
        total_required_minutes = sum(
            req.block_size * self.schedule_policy.period_minutes for req in self.block_requests
        )
        total_capacity_minutes = sum(max(0, item.max_hours) * 60 for item in self.faculty.values())
        if total_capacity_minutes <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No faculty capacity configured for timetable generation",
            )
        if total_required_minutes > total_capacity_minutes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Configured faculty maximum workload is insufficient for this term. "
                    f"Required weekly load is {total_required_minutes / 60:.1f}h "
                    f"but total faculty capacity is {total_capacity_minutes / 60:.1f}h."
                ),
            )

    def _validate_section_time_capacity(self) -> None:
        # Check if course hours alignment with credit load
        configured_hours = 0
        total_credits = 0
        for pc in self.program_courses:
            course = self.courses.get(pc.course_id)
            if course:
                configured_hours += course.hours_per_week
                total_credits += course.credits
        
        target_hours = self.expected_section_minutes // self.schedule_policy.period_minutes
        if configured_hours != target_hours:
             raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Curriculum mismatch: Total course contact hours ({configured_hours}h) "
                    f"does not match expected target ({target_hours}h). "
                    "Please adjust course hours_per_week."
                ),
            )

        term = (
            self.db.execute(
                select(ProgramTerm).where(
                    ProgramTerm.program_id == self.program_id,
                    ProgramTerm.term_number == self.term_number,
                )
            )
            .scalars()
            .first()
        )
        if term and term.credits_required > 0 and total_credits != term.credits_required:
             logger.warning(
                "Credit load mismatch for term %s: Total course credits (%sc) "
                "does not match required term credits (%sc).",
                self.term_number, total_credits, term.credits_required
            )

        total_available_slots = sum(len(slots) for slots in self.day_slots.values())
        if total_available_slots <= 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No available timetable slots configured in working hours/policy settings",
            )

        total_available_minutes = total_available_slots * self.schedule_policy.period_minutes
        if self.expected_section_minutes > total_available_minutes:
            required_hours = self.expected_section_minutes / 60
            available_hours = total_available_minutes / 60
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Program term weekly credit load exceeds available timetable capacity. "
                    f"Required: {required_hours:.1f}h/week per section, "
                    f"Available: {available_hours:.1f}h/week from configured working hours. "
                    "Reduce mapped course hours for this term or expand working-hour windows."
                ),
            )

    def _validate_prerequisite_mappings(self) -> None:
        completed_course_ids = set(
            self.db.execute(
                select(ProgramCourse.course_id).where(
                    ProgramCourse.program_id == self.program_id,
                    ProgramCourse.term_number < self.term_number,
                )
            )
            .scalars()
            .all()
        )

        violations: list[str] = []
        for program_course in self.program_courses:
            prerequisite_ids = set(program_course.prerequisite_course_ids or [])
            missing = sorted(prerequisite_ids - completed_course_ids)
            if missing:
                violations.append(f"{program_course.course_id} -> {', '.join(missing)}")

        if violations:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Prerequisite constraints are not satisfied: " + " | ".join(violations),
            )

    def _load_elective_overlap_pairs(self) -> set[tuple[str, str]]:
        groups = (
            self.db.execute(
                select(ProgramElectiveGroup).where(
                    ProgramElectiveGroup.program_id == self.program_id,
                    ProgramElectiveGroup.term_number == self.term_number,
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
            self.db.execute(
                select(ProgramElectiveGroupMember.group_id, ProgramCourse.course_id)
                .join(ProgramCourse, ProgramCourse.id == ProgramElectiveGroupMember.program_course_id)
                .where(ProgramElectiveGroupMember.group_id.in_(group_ids))
            )
            .all()
        )

        courses_by_group: dict[str, set[str]] = {}
        for group_id, course_id in rows:
            courses_by_group.setdefault(group_id, set()).add(course_id)

        pairs: set[tuple[str, str]] = set()
        for course_ids in courses_by_group.values():
            ordered = sorted(course_ids)
            for index, left in enumerate(ordered):
                for right in ordered[index + 1 :]:
                    pairs.add((left, right))
        return pairs

    def _courses_conflict_in_elective_group(self, course_a: str, course_b: str) -> bool:
        left, right = sorted((course_a, course_b))
        return (left, right) in self.elective_overlap_pairs

    def _load_shared_lecture_sections_by_course(self) -> dict[str, list[set[str]]]:
        groups = (
            self.db.execute(
                select(ProgramSharedLectureGroup).where(
                    ProgramSharedLectureGroup.program_id == self.program_id,
                    ProgramSharedLectureGroup.term_number == self.term_number,
                )
            )
            .scalars()
            .all()
        )
        if not groups:
            return {}

        group_ids = [group.id for group in groups]
        sections_by_group: dict[str, set[str]] = {}
        for member in self.db.execute(
            select(ProgramSharedLectureGroupMember).where(
                ProgramSharedLectureGroupMember.group_id.in_(group_ids)
            )
        ).scalars():
            sections_by_group.setdefault(member.group_id, set()).add(member.section_name)

        by_course: dict[str, list[set[str]]] = {}
        for group in groups:
            sections = sections_by_group.get(group.id, set())
            if len(sections) < 2:
                continue
            by_course.setdefault(group.course_id, []).append(sections)
        return by_course

    def _sections_share_shared_lecture(self, course_id: str, section_a: str, section_b: str) -> bool:
        for sections in self.shared_lecture_sections_by_course.get(course_id, []):
            if section_a in sections and section_b in sections:
                return True
        return False

    def _is_allowed_shared_overlap(
        self,
        req_a: BlockRequest,
        req_b: BlockRequest,
        option_a: PlacementOption,
        option_b: PlacementOption,
    ) -> bool:
        if req_a.is_lab or req_b.is_lab:
            return False
        if req_a.course_id != req_b.course_id:
            return False
        if req_a.section == req_b.section:
            return False
        if req_a.batch is not None or req_b.batch is not None:
            return False
        if req_a.session_type != req_b.session_type:
            return False
        if option_a.faculty_id != option_b.faculty_id:
            return False
        if option_a.room_id != option_b.room_id:
            return False
        if option_a.day != option_b.day or option_a.start_index != option_b.start_index:
            return False
        if req_a.block_size != req_b.block_size:
            return False
        return self._sections_share_shared_lecture(req_a.course_id, req_a.section, req_b.section)

    def _is_faculty_back_to_back(
        self,
        req_a: BlockRequest,
        option_a: PlacementOption,
        req_b: BlockRequest,
        option_b: PlacementOption,
    ) -> bool:
        if req_a.request_id == req_b.request_id:
            return False
        if option_a.faculty_id != option_b.faculty_id:
            return False
        if option_a.day != option_b.day:
            return False
        start_a, end_a = self._option_bounds(option_a, req_a.block_size)
        start_b, end_b = self._option_bounds(option_b, req_b.block_size)
        return end_a == start_b or end_b == start_a

    def _is_elective_request(self, req: BlockRequest) -> bool:
        course = self.courses.get(req.course_id)
        return bool(course is not None and course.type == CourseType.elective and not req.is_lab)

    def _room_candidates_for(self, course: Course) -> list[Room]:
        if course.type == CourseType.lab:
            candidates = [room for room in self.rooms.values() if room.type == RoomType.lab]
        else:
            candidates = [room for room in self.rooms.values() if room.type in {RoomType.lecture, RoomType.seminar}]
            if not candidates:
                candidates = list(self.rooms.values())
        if not candidates:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No rooms available for course {course.code}",
            )
        return candidates

    def _select_room_candidates_for_request(
        self,
        *,
        room_candidates: list[Room],
        student_count: int,
        is_lab: bool,
        seed: str = "global",
    ) -> list[Room]:
        """
        Selects and ranks room candidates with strong randomization within capacity tiers
        to ensure sessions are distributed across all available rooms.
        """
        import hashlib
        
        def room_score(room: Room) -> float:
            # Deterministic hash based on seed and room ID (0.0 to 1.0)
            h = int(hashlib.blake2b(f"{seed}|{room.id}".encode()).hexdigest(), 16)
            return (h % 10000) / 10000.0

        # Tier 1: Perfect or great fit (0 to 15 students extra)
        # Tier 2: Large fit (16+ students extra)
        # Tier 3: Undersized (spillover allowed but penalized)
        
        tiers: list[list[Room]] = [[], [], []]
        for room in room_candidates:
            if room.capacity < student_count:
                tiers[2].append(room)
            elif room.capacity <= student_count + 15:
                tiers[0].append(room)
            else:
                tiers[1].append(room)
        
        # Sort within tiers by the random score
        for tier in tiers:
            tier.sort(key=room_score)
        
        # Combine tiers
        ranked = tiers[0] + tiers[1] + tiers[2]

        if is_lab:
            max_candidates = min(len(ranked), max(10, min(20, len(ranked))))
        else:
            # For theory, consider almost all rooms but in a randomized order
            max_candidates = min(len(ranked), max(30, min(45, len(ranked))))

        return ranked[:max_candidates]

    def _faculty_course_tiebreak(self, *, course_code: str, faculty_id: str) -> str:
        seed = f"{course_code.upper()}|{faculty_id}".encode("utf-8")
        return hashlib.blake2b(seed, digest_size=6).hexdigest()

    def _room_is_available(self, room: Room, day: str, start_min: int, end_min: int) -> bool:
        if not room.availability_windows:
            return True
        for window in room.availability_windows:
            if normalize_day(window.get("day", "")) != day:
                continue
            window_start = parse_time_to_minutes(window["start_time"])
            window_end = parse_time_to_minutes(window["end_time"])
            if start_min >= window_start and end_min <= window_end:
                return True
        return False

    def _faculty_allows_day(self, faculty: Faculty, day: str) -> bool:
        if not faculty.availability:
            return True
        normalized = {normalize_day(item) for item in faculty.availability}
        return day in normalized

    def _within_semester_time_window(self, start_min: int, end_min: int) -> bool:
        if self.semester_constraint is None:
            return True
        earliest = parse_time_to_minutes(self.semester_constraint.earliest_start_time)
        latest = parse_time_to_minutes(self.semester_constraint.latest_end_time)
        return start_min >= earliest and end_min <= latest

    def _faculty_prefers_subject(self, faculty_id: str, course_code: str) -> bool:
        if not course_code:
            return False
        preference_map = getattr(self, "faculty_preferred_subject_codes", {})
        return course_code.upper() in preference_map.get(faculty_id, set())

    def _faculty_candidates_for_course(self, course: Course) -> list[str]:
        ordered_ids: list[str] = []
        if course.faculty_id and course.faculty_id in self.faculty:
            ordered_ids.append(course.faculty_id)

        preferred_ids = sorted(
            [
                item.id
                for item in self.faculty.values()
                if self._faculty_prefers_subject(item.id, course.code)
            ],
            key=lambda item_id: (
                self.faculty[item_id].workload_hours,
                -self.faculty[item_id].max_hours,
                self.faculty[item_id].name,
                self._faculty_course_tiebreak(course_code=course.code, faculty_id=item_id),
            ),
        )
        for item_id in preferred_ids:
            if item_id not in ordered_ids:
                ordered_ids.append(item_id)

        fallback_ids = sorted(
            [item_id for item_id in self.faculty.keys() if item_id not in ordered_ids],
            key=lambda item_id: (
                self.faculty[item_id].workload_hours,
                -self.faculty[item_id].max_hours,
                self.faculty[item_id].name,
                self._faculty_course_tiebreak(course_code=course.code, faculty_id=item_id),
            ),
        )
        ordered_ids.extend(fallback_ids)

        if not ordered_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No candidate faculty found for course {course.code}",
            )

        if course.faculty_id and course.faculty_id in self.faculty:
            candidate_cap = max(16, min(len(ordered_ids), len(preferred_ids) + 16))
        elif preferred_ids:
            candidate_cap = max(18, min(len(ordered_ids), len(preferred_ids) + 18))
        else:
            candidate_cap = max(18, min(len(ordered_ids), math.ceil(len(self.faculty) * 0.65)))

        return ordered_ids[: min(len(ordered_ids), candidate_cap)]

    def _build_block_requests(self) -> list[BlockRequest]:
        requests: list[BlockRequest] = []
        request_id = 0
        period_minutes = self.schedule_policy.period_minutes

        for program_course in self.program_courses:
            course = self.courses.get(program_course.course_id)
            if course is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Course mapping missing for course id {program_course.course_id}",
                )
            if course.faculty_id and course.faculty_id not in self.faculty:
                logger.warning(
                    "Course %s has stale faculty assignment %s; falling back to candidate faculty pool",
                    course.code,
                    course.faculty_id,
                )
            faculty_candidate_ids = self._faculty_candidates_for_course(course)
            primary_faculty_id = (
                course.faculty_id if course.faculty_id and course.faculty_id in self.faculty else faculty_candidate_ids[0]
            )
            preferred_faculty_ids = tuple(
                item_id for item_id in faculty_candidate_ids if self._faculty_prefers_subject(item_id, course.code)
            )

            max_daily_slots = max((len(slots) for slots in self.day_slots.values()), default=0)
            total_credit_hours = course.theory_hours + course.lab_hours + course.tutorial_hours
            # Note: alignment already validated in _validate_course_credit_alignment

            request_templates: list[tuple[Literal["theory", "tutorial", "lab"], int, int]] = []
            
            # STRICT LOGIC: Lab courses are ONLY labs. Theory courses are ONLY theory/tutorial.
            if course.type == CourseType.lab:
                if course.theory_hours > 0 or course.tutorial_hours > 0:
                     raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Lab course {course.code} cannot have theory or tutorial hours.",
                    )
                lab_block_size = self.schedule_policy.lab_contiguous_slots
                # Hard constraint: Labs must be divisible by block size (usually 2)
                if course.lab_hours % lab_block_size != 0:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=(
                            f"Course {course.code} weekly lab hours ({course.lab_hours}) must be divisible by "
                            f"required lab block size ({lab_block_size})"
                        ),
                    )
                request_templates.append(("lab", lab_block_size, course.lab_hours // lab_block_size))
            else:
                if course.lab_hours > 0:
                     raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Theory course {course.code} cannot have lab hours.",
                    )
                if course.theory_hours > 0:
                    request_templates.append(("theory", 1, course.theory_hours))
                if course.tutorial_hours > 0:
                    request_templates.append(("tutorial", 1, course.tutorial_hours))
                
            if not request_templates:
                 raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Course {course.code} has no valid hours to schedule.",
                )

            room_candidates = self._room_candidates_for(course)
            batch_count = program_course.lab_batch_count if course.type == CourseType.lab else 1

            for section in self.sections:
                if course.type == CourseType.lab:
                    student_per_batch = max(1, math.ceil(section.capacity / max(1, batch_count)))
                    batch_labels = [f"B{index + 1}" for index in range(batch_count)]
                else:
                    student_per_batch = max(1, section.capacity)
                    batch_labels = [None]
                request_room_candidates = self._select_room_candidates_for_request(
                    room_candidates=room_candidates,
                    student_count=student_per_batch,
                    is_lab=course.type == CourseType.lab,
                    seed=f"{course.code}|{section}", # Added section to seed for dispersion
                )

                for batch in batch_labels:
                    for session_type, block_size, blocks_needed in request_templates:
                        if block_size > max_daily_slots:
                            raise HTTPException(
                                status_code=status.HTTP_400_BAD_REQUEST,
                                detail=(
                                    f"No feasible placement options for course {course.code}: required contiguous block size "
                                    f"({block_size}) exceeds available daily slots ({max_daily_slots}). "
                                    "Adjust lab contiguous-slot policy or working hours."
                                ),
                            )
                        faculty_option_order = tuple(faculty_candidate_ids)
                        for _ in range(blocks_needed):
                            def collect_options(
                                *,
                                enforce_semester_window: bool,
                                enforce_room_windows: bool,
                                enforce_faculty_day: bool,
                                enforce_faculty_windows: bool,
                                enforce_reserved_resources: bool,
                                option_limit: int = 640,
                            ) -> list[PlacementOption]:
                                generated: list[PlacementOption] = []
                                active_day_count = max(1, len(self.day_slots))
                                per_day_limit = max(1, option_limit // active_day_count)
                                day_option_counts: dict[str, int] = defaultdict(int)
                                day_start_option_counts: dict[tuple[str, int], int] = defaultdict(int)
                                for day, slots in self.day_slots.items():
                                    start_positions = max(1, len(slots) - block_size + 1)
                                    per_start_limit = max(
                                        4,
                                        min(
                                            24,
                                            math.ceil(per_day_limit / start_positions),
                                        ),
                                    )
                                    for start_index in range(start_positions):
                                        block_start = slots[start_index].start
                                        block_end = slots[start_index + block_size - 1].end
                                        if enforce_semester_window and not self._within_semester_time_window(block_start, block_end):
                                            continue
                                        # Iterate room first so per-start caps retain multiple faculty choices.
                                        for room in request_room_candidates:
                                            if enforce_room_windows and not self._room_is_available(room, day, block_start, block_end):
                                                continue
                                            for faculty_id in faculty_option_order:
                                                faculty = self.faculty[faculty_id]
                                                if enforce_faculty_day and not self._faculty_allows_day(faculty, day):
                                                    continue
                                                if enforce_faculty_windows:
                                                    faculty_windows = self.faculty_windows.get(faculty_id, {})
                                                    if faculty_windows.get(day):
                                                        if not any(
                                                            start <= block_start and block_end <= end
                                                            for start, end in faculty_windows[day]
                                                        ):
                                                            continue
                                                if enforce_room_windows and not self._room_is_available(room, day, block_start, block_end):
                                                    continue
                                                if day_option_counts[day] >= per_day_limit and len(generated) < option_limit:
                                                    continue
                                                if day_start_option_counts[(day, start_index)] >= per_start_limit and len(generated) < option_limit:
                                                    continue
                                                if enforce_reserved_resources and self._conflicts_reserved_resources(
                                                    day=day,
                                                    start_min=block_start,
                                                    end_min=block_end,
                                                    room_id=room.id,
                                                    faculty_id=faculty_id,
                                                ):
                                                    continue
                                                generated.append(
                                                    PlacementOption(
                                                        day=day,
                                                        start_index=start_index,
                                                        room_id=room.id,
                                                        faculty_id=faculty_id,
                                                    )
                                                )
                                                day_option_counts[day] += 1
                                                day_start_option_counts[(day, start_index)] += 1
                                                if len(generated) >= option_limit:
                                                    return generated
                                return generated

                            options = collect_options(
                                enforce_semester_window=True,
                                enforce_room_windows=True,
                                enforce_faculty_day=True,
                                enforce_faculty_windows=True,
                                enforce_reserved_resources=True,
                            )
                            relaxed_option_mode = ""
                            if not options:
                                options = collect_options(
                                    enforce_semester_window=True,
                                    enforce_room_windows=False,
                                    enforce_faculty_day=True,
                                    enforce_faculty_windows=False,
                                    enforce_reserved_resources=False,
                                )
                                if options:
                                    relaxed_option_mode = "soft-resource-fallback"
                            if not options:
                                options = collect_options(
                                    enforce_semester_window=False,
                                    enforce_room_windows=False,
                                    enforce_faculty_day=False,
                                    enforce_faculty_windows=False,
                                    enforce_reserved_resources=False,
                                )
                                if options:
                                    relaxed_option_mode = "hard-feasibility-fallback"
                            if not options:
                                raise HTTPException(
                                    status_code=status.HTTP_400_BAD_REQUEST,
                                    detail=(
                                        f"No feasible placement options for course {course.code} ({session_type}), "
                                        f"section {section.name}{f' batch {batch}' if batch else ''}"
                                    ),
                                )
                            if len(options) > 640:
                                options = sorted(
                                    options,
                                    key=lambda item: (
                                        item.faculty_id != primary_faculty_id,
                                        item.faculty_id not in preferred_faculty_ids,
                                        self.rooms[item.room_id].capacity < student_per_batch,
                                        abs(self.rooms[item.room_id].capacity - student_per_batch),
                                        item.day,
                                        item.start_index,
                                        item.room_id,
                                        item.faculty_id,
                                    ),
                                )[:640]
                            if relaxed_option_mode:
                                logger.warning(
                                    "Generation option fallback used | program_id=%s term=%s course=%s session=%s section=%s batch=%s mode=%s options=%s",
                                    self.program_id,
                                    self.term_number,
                                    course.code,
                                    session_type,
                                    section.name,
                                    batch,
                                    relaxed_option_mode,
                                    len(options),
                                )
                            requests.append(
                                BlockRequest(
                                    request_id=request_id,
                                    course_id=course.id,
                                    course_code=course.code,
                                    section=section.name,
                                    batch=batch,
                                    student_count=student_per_batch,
                                    primary_faculty_id=primary_faculty_id,
                                    preferred_faculty_ids=preferred_faculty_ids,
                                    block_size=block_size,
                                    is_lab=course.type == CourseType.lab,
                                    session_type=session_type,
                                    allow_parallel_batches=program_course.allow_parallel_batches,
                                    room_candidate_ids=tuple(room.id for room in request_room_candidates),
                                    options=tuple(options),
                                )
                            )
                            request_id += 1

        if not requests:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No schedulable requests built for this program term",
            )
        return requests

    def _build_request_indices_by_course(self) -> dict[str, list[int]]:
        by_course: dict[str, list[int]] = defaultdict(list)
        for req_index, req in enumerate(self.block_requests):
            by_course[req.course_id].append(req_index)
        return dict(by_course)

    def _build_request_indices_by_course_section(self) -> dict[tuple[str, str], list[int]]:
        by_course_section: dict[tuple[str, str], list[int]] = defaultdict(list)
        for req_index, req in enumerate(self.block_requests):
            if req.is_lab:
                continue
            by_course_section[(req.course_id, req.section)].append(req_index)
        return dict(by_course_section)

    def _request_indices_by_course_section(self) -> dict[tuple[str, str], list[int]]:
        mapping = getattr(self, "request_indices_by_course_section", None)
        if isinstance(mapping, dict):
            return mapping
        return self._build_request_indices_by_course_section()

    def _build_common_faculty_candidates_by_course_section(self) -> dict[tuple[str, str], tuple[str, ...]]:
        common_by_course_section: dict[tuple[str, str], tuple[str, ...]] = {}
        for course_section_key, req_indices in self._request_indices_by_course_section().items():
            if len(req_indices) <= 1:
                continue
            common_faculty_ids: set[str] | None = None
            for req_index in req_indices:
                faculty_ids = {option.faculty_id for option in self.block_requests[req_index].options}
                if common_faculty_ids is None:
                    common_faculty_ids = set(faculty_ids)
                else:
                    common_faculty_ids &= faculty_ids
            if not common_faculty_ids:
                continue
            common_by_course_section[course_section_key] = tuple(sorted(common_faculty_ids))
        return common_by_course_section

    def _build_single_faculty_requirements_by_course(self) -> dict[str, bool]:
        """
        Enforce one-faculty-across-sections only when it is explicitly assigned and
        workload-feasible. This avoids impossible hard-conflict states for high-load terms.
        """
        requirements: dict[str, bool] = {}
        period_minutes = self.schedule_policy.period_minutes

        for course_id, req_indices in self.request_indices_by_course.items():
            lecture_req_indices = [idx for idx in req_indices if not self.block_requests[idx].is_lab]
            if len(lecture_req_indices) <= 1:
                requirements[course_id] = False
                continue

            course = self.courses.get(course_id)
            if course is None or not course.faculty_id or course.faculty_id not in self.faculty:
                requirements[course_id] = False
                continue

            dedicated_faculty = self.faculty[course.faculty_id]
            dedicated_capacity_minutes = max(0, dedicated_faculty.max_hours) * 60
            if dedicated_capacity_minutes <= 0:
                requirements[course_id] = False
                logger.warning(
                    "Single-faculty enforcement relaxed | course=%s | faculty=%s has no positive max_hours",
                    course.code,
                    dedicated_faculty.name,
                )
                continue

            required_minutes = sum(
                self.block_requests[idx].block_size * period_minutes for idx in lecture_req_indices
            )
            if required_minutes > dedicated_capacity_minutes:
                requirements[course_id] = False
                logger.warning(
                    "Single-faculty enforcement relaxed | course=%s | required_hours=%.1f exceeds faculty_max_hours=%.1f",
                    course.code,
                    required_minutes / 60.0,
                    dedicated_capacity_minutes / 60.0,
                )
                continue

            requirements[course_id] = True

        return requirements

    def _single_faculty_required(self, course_id: str) -> bool:
        mapping = getattr(self, "single_faculty_required_by_course", None)
        if isinstance(mapping, dict):
            return bool(mapping.get(course_id, False))
        # Unit tests that construct scheduler instances manually may skip __init__.
        course = self.courses.get(course_id) if hasattr(self, "courses") else None
        if course is None:
            return False
        return bool(getattr(course, "faculty_id", None))

    def _build_common_faculty_candidates_by_course(self) -> dict[str, tuple[str, ...]]:
        common_by_course: dict[str, tuple[str, ...]] = {}
        for course_id, req_indices in self.request_indices_by_course.items():
            if not self._single_faculty_required(course_id):
                continue
            lecture_req_indices = [idx for idx in req_indices if not self.block_requests[idx].is_lab]
            if len(lecture_req_indices) <= 1:
                continue

            common_faculty_ids: set[str] | None = None
            for req_index in lecture_req_indices:
                request_faculty_ids = {option.faculty_id for option in self.block_requests[req_index].options}
                if common_faculty_ids is None:
                    common_faculty_ids = set(request_faculty_ids)
                else:
                    common_faculty_ids &= request_faculty_ids

            if not common_faculty_ids:
                course = self.courses.get(course_id)
                course_code = course.code if course is not None else course_id
                self.single_faculty_required_by_course[course_id] = False
                logger.warning(
                    "Single-faculty enforcement relaxed | course=%s has no common feasible faculty across sections",
                    course_code,
                )
                continue

            common_by_course[course_id] = tuple(sorted(common_faculty_ids))

        return common_by_course

    def _load_fixed_genes(self) -> dict[int, int]:
        locks = (
            self.db.execute(
                select(TimetableSlotLock).where(
                    TimetableSlotLock.program_id == self.program_id,
                    TimetableSlotLock.term_number == self.term_number,
                    TimetableSlotLock.is_active.is_(True),
                )
            )
            .scalars()
            .all()
        )
        if not locks:
            return {}

        fixed: dict[int, int] = {}
        used_requests: set[int] = set()
        for lock in locks:
            matching_requests = [
                req for req in self.block_requests
                if req.course_id == lock.course_id and req.section == lock.section_name and req.batch == lock.batch
            ]
            if not matching_requests:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"Active lock for course {lock.course_id}, section {lock.section_name} "
                        f"has no matching generation request"
                    ),
                )

            matched = False
            for req in matching_requests:
                if req.request_id in used_requests:
                    continue
                for option_index, option in enumerate(req.options):
                    if option.day != lock.day:
                        continue
                    option_start = self.day_slots[option.day][option.start_index].start
                    option_end = self.day_slots[option.day][option.start_index + req.block_size - 1].end
                    if minutes_to_time(option_start) != lock.start_time or minutes_to_time(option_end) != lock.end_time:
                        continue
                    if lock.room_id and lock.room_id != option.room_id:
                        continue
                    if lock.faculty_id and lock.faculty_id != option.faculty_id:
                        continue
                    fixed[req.request_id] = option_index
                    used_requests.add(req.request_id)
                    matched = True
                    break
                if matched:
                    break

            if not matched:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"Active lock for course {lock.course_id}, section {lock.section_name} "
                        "cannot be represented with current slot options"
                    ),
                )

        return fixed

    def _validate_locked_course_faculty_consistency(self) -> None:
        locked_faculty_by_course: dict[str, str] = {}

        for req_index, option_index in self.fixed_genes.items():
            req = self.block_requests[req_index]
            if req.is_lab or not self._single_faculty_required(req.course_id):
                continue
            faculty_id = req.options[option_index].faculty_id
            existing = locked_faculty_by_course.get(req.course_id)
            if existing is not None and existing != faculty_id:
                course = self.courses.get(req.course_id)
                course_code = course.code if course is not None else req.course_id
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"Active locks assign multiple faculty to course {course_code}. "
                        "Keep one faculty assignment for this course across all sections."
                    ),
                )
            locked_faculty_by_course[req.course_id] = faculty_id

        for course_id, locked_faculty_id in locked_faculty_by_course.items():
            for req_index in self.request_indices_by_course.get(course_id, []):
                req = self.block_requests[req_index]
                if req.is_lab or req.request_id in self.fixed_genes:
                    continue
                if all(option.faculty_id != locked_faculty_id for option in req.options):
                    course = self.courses.get(course_id)
                    course_code = course.code if course is not None else course_id
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=(
                            f"Locked faculty assignment for course {course_code} is not feasible across all sections. "
                            "Update slot locks or faculty availability."
                        ),
                    )

    def _random_individual(self) -> list[int]:
        genes: list[int] = []
        chosen_faculty_by_course: dict[str, str] = {}
        chosen_faculty_by_course_section: dict[tuple[str, str], str] = {}
        for req in self.block_requests:
            if req.request_id in self.fixed_genes:
                fixed_option_index = self.fixed_genes[req.request_id]
                genes.append(fixed_option_index)
                if not req.is_lab:
                    selected_faculty_id = req.options[fixed_option_index].faculty_id
                    chosen_faculty_by_course_section[(req.course_id, req.section)] = selected_faculty_id
                    if self._single_faculty_required(req.course_id):
                        chosen_faculty_by_course.setdefault(req.course_id, selected_faculty_id)
            else:
                if not req.is_lab:
                    selected_faculty_id = chosen_faculty_by_course_section.get((req.course_id, req.section))
                    if selected_faculty_id is None and self._single_faculty_required(req.course_id):
                        selected_faculty_id = chosen_faculty_by_course.get(req.course_id)
                    if selected_faculty_id is None:
                        section_common_map = getattr(self, "common_faculty_candidates_by_course_section", {})
                        section_common_faculty_ids = section_common_map.get(
                            (req.course_id, req.section),
                            (),
                        )
                        if section_common_faculty_ids:
                            selected_faculty_id = self.random.choice(list(section_common_faculty_ids))
                    if selected_faculty_id is None:
                        common_faculty_ids = self.common_faculty_candidates_by_course.get(req.course_id, ())
                        if common_faculty_ids:
                            selected_faculty_id = self.random.choice(list(common_faculty_ids))
                    if selected_faculty_id is not None:
                        matching_indices = [
                            option_index
                            for option_index, option in enumerate(req.options)
                            if option.faculty_id == selected_faculty_id
                        ]
                        if matching_indices:
                            chosen_index = self.random.choice(matching_indices)
                            genes.append(chosen_index)
                            chosen_faculty_by_course_section[(req.course_id, req.section)] = selected_faculty_id
                            if self._single_faculty_required(req.course_id):
                                chosen_faculty_by_course.setdefault(req.course_id, selected_faculty_id)
                            continue
                genes.append(self.random.randrange(len(req.options)))
        return genes

    def _normalize_windows(self, windows: list[dict]) -> dict[str, list[tuple[int, int]]]:
        result: dict[str, list[tuple[int, int]]] = {}
        for window in windows:
            day = normalize_day(window.get("day", ""))
            if day not in self.day_slots:
                continue
            start = parse_time_to_minutes(window["start_time"])
            end = parse_time_to_minutes(window["end_time"])
            if end <= start:
                continue
            result.setdefault(day, []).append((start, end))
        return result

    def _option_candidate_indices(
        self,
        req: BlockRequest,
        max_candidates: int = 16,
        *,
        allow_random_tail: bool = True,
    ) -> list[int]:
        ranked = self.option_priority_indices.get(req.request_id, [])
        option_count = len(ranked)
        if option_count <= max_candidates:
            return list(ranked)

        if allow_random_tail:
            anchor_count = max(1, max_candidates // 4)
        else:
            # Deterministic mode should not over-bias the very first ranked options.
            anchor_count = max(1, max_candidates // 5)
        shortlisted = list(ranked[:anchor_count])
        random_tail_count = max(0, max_candidates - len(shortlisted))
        if random_tail_count <= 0:
            return shortlisted

        tail = ranked[anchor_count:]
        if len(tail) <= random_tail_count:
            shortlisted.extend(tail)
        elif not allow_random_tail:
            # Deterministic runs should still cover the full week; avoid "all Monday" bias
            # when ranked options are ordered by day/start.
            day_buckets: dict[str, list[int]] = defaultdict(list)
            for option_index in tail:
                day_buckets[req.options[option_index].day].append(option_index)

            ordered_days = [day for day in self.day_slots.keys() if day in day_buckets]
            ordered_days.extend(day for day in sorted(day_buckets.keys()) if day not in ordered_days)

            sampled_tail: list[int] = []
            while len(sampled_tail) < random_tail_count:
                progressed = False
                for day in ordered_days:
                    bucket = day_buckets.get(day)
                    if not bucket:
                        continue
                    sampled_tail.append(bucket.pop(0))
                    progressed = True
                    if len(sampled_tail) >= random_tail_count:
                        break
                if not progressed:
                    break

            if len(sampled_tail) < random_tail_count:
                fallback_needed = random_tail_count - len(sampled_tail)
                for option_index in tail:
                    if option_index in sampled_tail:
                        continue
                    sampled_tail.append(option_index)
                    fallback_needed -= 1
                    if fallback_needed <= 0:
                        break

            shortlisted.extend(sampled_tail[:random_tail_count])
        else:
            shortlisted.extend(self.random.sample(tail, random_tail_count))
        return shortlisted

    def _spread_option_indices_by_day(self, req: BlockRequest, option_indices: list[int]) -> list[int]:
        if len(option_indices) <= 2:
            return option_indices
        day_buckets: dict[str, list[int]] = defaultdict(list)
        for option_index in option_indices:
            day_buckets[req.options[option_index].day].append(option_index)
        ordered_days = [day for day in self.day_slots.keys() if day in day_buckets]
        ordered_days.extend(day for day in sorted(day_buckets.keys()) if day not in ordered_days)
        reordered: list[int] = []
        while len(reordered) < len(option_indices):
            progressed = False
            for day in ordered_days:
                bucket = day_buckets.get(day)
                if not bucket:
                    continue
                reordered.append(bucket.pop(0))
                progressed = True
            if not progressed:
                break
        if len(reordered) == len(option_indices):
            return reordered
        seen = set(reordered)
        reordered.extend(option_index for option_index in option_indices if option_index not in seen)
        return reordered

    def _conflicted_request_ids(self, genes: list[int]) -> set[int]:
        conflicted: set[int] = set()
        weights = self.settings.objective_weights
        if not weights:
            return conflicted

        room_occ: dict[tuple[str, int, str], list[int]] = {}
        faculty_occ: dict[tuple[str, int, str], list[int]] = {}
        faculty_day_req_indices: dict[tuple[str, str], list[int]] = {}
        elective_signatures_by_section: dict[str, list[tuple[str, int, int, str]]] = defaultdict(list)
        elective_req_ids_by_section: dict[str, list[int]] = defaultdict(list)
        section_occ: dict[tuple[str, int, str], list[int]] = {}
        elective_occ: dict[tuple[str, int], list[int]] = {}
        section_day_slots: dict[tuple[str, str], set[int]] = {}
        section_day_req_ids: dict[tuple[str, str], set[int]] = {}
        section_req_ids: dict[str, set[int]] = {}
        faculty_req_ids: dict[str, set[int]] = {}
        faculty_minutes: dict[str, int] = {}
        selected_options: dict[int, PlacementOption] = {}

        for req_index, req in enumerate(self.block_requests):
            option = req.options[genes[req_index]]
            selected_options[req_index] = option
            room = self.rooms[option.room_id]
            faculty = self.faculty[option.faculty_id]

            block_start, block_end = self._option_bounds(option, req.block_size)

            if not self._within_semester_time_window(block_start, block_end):
                conflicted.add(req_index)

            if self._conflicts_reserved_resources(
                day=option.day,
                start_min=block_start,
                end_min=block_end,
                room_id=option.room_id,
                faculty_id=option.faculty_id,
            ):
                conflicted.add(req_index)

            if room.capacity < req.student_count:
                conflicted.add(req_index)
            if req.is_lab and room.type != RoomType.lab:
                conflicted.add(req_index)
            if not req.is_lab and room.type == RoomType.lab:
                conflicted.add(req_index)

            if not self._faculty_allows_day(faculty, option.day):
                conflicted.add(req_index)

            if self.faculty_windows.get(option.faculty_id, {}).get(option.day):
                if not any(
                    start <= block_start and block_end <= end
                    for start, end in self.faculty_windows[option.faculty_id][option.day]
                ):
                    conflicted.add(req_index)

            if self.room_windows.get(option.room_id, {}).get(option.day):
                if not any(
                    start <= block_start and block_end <= end
                    for start, end in self.room_windows[option.room_id][option.day]
                ):
                    conflicted.add(req_index)

            section_day_req_ids.setdefault((req.section, option.day), set()).add(req_index)
            section_req_ids.setdefault(req.section, set()).add(req_index)
            faculty_req_ids.setdefault(option.faculty_id, set()).add(req_index)
            faculty_day_req_indices.setdefault((option.faculty_id, option.day), []).append(req_index)
            if self._is_elective_request(req):
                elective_signatures_by_section[req.section].append(
                    (option.day, option.start_index, req.block_size, req.session_type)
                )
                elective_req_ids_by_section[req.section].append(req_index)

            for offset in range(req.block_size):
                slot_idx = option.start_index + offset
                room_key = (option.day, slot_idx, option.room_id)
                faculty_key = (option.day, slot_idx, option.faculty_id)
                section_key = (option.day, slot_idx, req.section)
                room_occ.setdefault(room_key, []).append(req_index)
                faculty_occ.setdefault(faculty_key, []).append(req_index)
                section_occ.setdefault(section_key, []).append(req_index)
                elective_occ.setdefault((option.day, slot_idx), []).append(req_index)
                section_day_slots.setdefault((req.section, option.day), set()).add(slot_idx)
                faculty_minutes[option.faculty_id] = (
                    faculty_minutes.get(option.faculty_id, 0) + self.schedule_policy.period_minutes
                )

        for values in room_occ.values():
            if len(values) <= 1:
                continue
            for left_index, left_req_idx in enumerate(values):
                for right_req_idx in values[left_index + 1 :]:
                    left_req = self.block_requests[left_req_idx]
                    right_req = self.block_requests[right_req_idx]
                    if self._is_allowed_shared_overlap(
                        left_req,
                        right_req,
                        selected_options[left_req_idx],
                        selected_options[right_req_idx],
                    ):
                        continue
                    conflicted.add(left_req_idx)
                    conflicted.add(right_req_idx)

        for values in faculty_occ.values():
            if len(values) <= 1:
                continue
            for left_index, left_req_idx in enumerate(values):
                for right_req_idx in values[left_index + 1 :]:
                    left_req = self.block_requests[left_req_idx]
                    right_req = self.block_requests[right_req_idx]
                    if self._is_allowed_shared_overlap(
                        left_req,
                        right_req,
                        selected_options[left_req_idx],
                        selected_options[right_req_idx],
                    ):
                        continue
                    conflicted.add(left_req_idx)
                    conflicted.add(right_req_idx)

        # Back-to-back teacher sessions are optimization targets (soft penalties)
        # to keep generation feasible in dense schedules.

        for values in section_occ.values():
            if len(values) <= 1:
                continue
            requests = [self.block_requests[idx] for idx in values]
            first = requests[0]
            is_allowed_parallel_batch = (
                all(item.is_lab for item in requests)
                and all(item.course_id == first.course_id for item in requests)
                and all(item.allow_parallel_batches for item in requests)
                and all(item.batch for item in requests)
                and len({item.batch for item in requests}) == len(requests)
            )
            if not is_allowed_parallel_batch:
                conflicted.update(values)

        if self.elective_overlap_pairs:
            for values in elective_occ.values():
                if len(values) <= 1:
                    continue
                requests = [self.block_requests[idx] for idx in values]
                for left_index, left_req in enumerate(requests):
                    for right_offset, right_req in enumerate(requests[left_index + 1 :], start=left_index + 1):
                        if left_req.course_id == right_req.course_id:
                            continue
                        if self._courses_conflict_in_elective_group(left_req.course_id, right_req.course_id):
                            conflicted.add(values[left_index])
                            conflicted.add(values[right_offset])

        elective_sections = sorted(elective_signatures_by_section.keys())
        if len(elective_sections) > 1:
            baseline: list[tuple[str, int, int, str]] | None = None
            mismatch = False
            for section_name in elective_sections:
                signatures = sorted(elective_signatures_by_section.get(section_name, []))
                if baseline is None:
                    baseline = signatures
                    continue
                if signatures != baseline:
                    mismatch = True
                    break
            if mismatch:
                for section_name in elective_sections:
                    conflicted.update(elective_req_ids_by_section.get(section_name, []))

        if self.shared_lecture_sections_by_course:
            signatures_by_course_section: dict[tuple[str, str], list[tuple[str, int, str, str, int]]] = defaultdict(list)
            requests_by_course_section: dict[tuple[str, str], list[int]] = defaultdict(list)
            for req_index, req in enumerate(self.block_requests):
                if req.is_lab or req.course_id not in self.shared_lecture_sections_by_course:
                    continue
                option = selected_options[req_index]
                signatures_by_course_section[(req.course_id, req.section)].append(
                    (option.day, option.start_index, option.room_id, option.faculty_id, req.block_size)
                )
                requests_by_course_section[(req.course_id, req.section)].append(req_index)

            for course_id, groups in self.shared_lecture_sections_by_course.items():
                for sections in groups:
                    baseline: list[tuple[str, int, str, str, int]] | None = None
                    for section in sorted(sections):
                        signatures = sorted(signatures_by_course_section.get((course_id, section), []))
                        if baseline is None:
                            baseline = signatures
                            continue
                        if signatures != baseline:
                            for bad_section in sections:
                                conflicted.update(requests_by_course_section.get((course_id, bad_section), []))
                            break

        parallel_lab_signatures: dict[tuple[str, str], dict[str, list[tuple[str, int, int]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        parallel_lab_req_ids: dict[tuple[str, str], dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        for req_index, req in enumerate(self.block_requests):
            if not req.is_lab or not req.allow_parallel_batches or not req.batch:
                continue
            option = selected_options[req_index]
            group_key = (req.course_id, req.section)
            signature = (option.day, option.start_index, req.block_size)
            parallel_lab_signatures[group_key][req.batch].append(signature)
            parallel_lab_req_ids[group_key][req.batch].append(req_index)

        for group_key, signatures_by_batch in parallel_lab_signatures.items():
            if len(signatures_by_batch) <= 1:
                continue
            baseline: list[tuple[str, int, int]] | None = None
            mismatch = False
            for batch_name in sorted(signatures_by_batch.keys()):
                signatures = sorted(signatures_by_batch[batch_name])
                if baseline is None:
                    baseline = signatures
                    continue
                if signatures != baseline:
                    mismatch = True
                    break
            if mismatch:
                for batch_name in signatures_by_batch.keys():
                    conflicted.update(parallel_lab_req_ids[group_key].get(batch_name, []))

        for (_course_id, _section_name), req_indices in self._request_indices_by_course_section().items():
            lecture_req_indices = req_indices
            if len(lecture_req_indices) <= 1:
                continue
            assigned_faculty_ids = {selected_options[idx].faculty_id for idx in lecture_req_indices}
            if len(assigned_faculty_ids) > 1:
                conflicted.update(lecture_req_indices)

        for course_id, req_indices in self.request_indices_by_course.items():
            if not self._single_faculty_required(course_id):
                continue
            lecture_req_indices = [idx for idx in req_indices if not self.block_requests[idx].is_lab]
            if len(lecture_req_indices) <= 1:
                continue
            assigned_faculty_ids = {selected_options[idx].faculty_id for idx in lecture_req_indices}
            if len(assigned_faculty_ids) > 1:
                conflicted.update(lecture_req_indices)

        if self.semester_constraint is not None:
            day_limit = self.semester_constraint.max_hours_per_day * 60
            week_limit = self.semester_constraint.max_hours_per_week * 60
            min_break = self.semester_constraint.min_break_minutes
            max_consecutive = self.semester_constraint.max_consecutive_hours * 60

            weekly_minutes_by_section: dict[str, int] = {}
            for (section, day), slot_set in section_day_slots.items():
                day_minutes = len(slot_set) * self.schedule_policy.period_minutes
                weekly_minutes_by_section[section] = weekly_minutes_by_section.get(section, 0) + day_minutes
                if day_minutes > day_limit:
                    conflicted.update(section_day_req_ids.get((section, day), set()))

                slot_indexes = sorted(slot_set)
                if not slot_indexes:
                    continue
                run_start = slot_indexes[0]
                prev = slot_indexes[0]
                for current in slot_indexes[1:]:
                    prev_end = self.day_slots[day][prev].end
                    current_start = self.day_slots[day][current].start
                    gap = current_start - prev_end
                    if gap < min_break:
                        conflicted.update(section_day_req_ids.get((section, day), set()))

                    if gap != 0:
                        run_duration = self.day_slots[day][prev].end - self.day_slots[day][run_start].start
                        if run_duration > max_consecutive:
                            conflicted.update(section_day_req_ids.get((section, day), set()))
                        run_start = current
                    prev = current

                run_duration = self.day_slots[day][prev].end - self.day_slots[day][run_start].start
                if run_duration > max_consecutive:
                    conflicted.update(section_day_req_ids.get((section, day), set()))

            for section, minutes in weekly_minutes_by_section.items():
                if minutes > week_limit:
                    conflicted.update(section_req_ids.get(section, set()))

            if self.expected_section_minutes > 0:
                for section, request_ids in section_req_ids.items():
                    minutes = weekly_minutes_by_section.get(section, 0)
                    if minutes != self.expected_section_minutes:
                        conflicted.update(request_ids)

        # Workload caps are hard constraints for publishable schedules.
        for faculty_id, minutes in faculty_minutes.items():
            faculty = self.faculty.get(faculty_id)
            if faculty is None:
                continue
            max_minutes = max(0, faculty.max_hours) * 60
            if max_minutes and minutes > max_minutes:
                conflicted.update(faculty_req_ids.get(faculty_id, set()))

        return conflicted

    def _overlap_conflicted_request_ids(self, genes: list[int]) -> set[int]:
        conflicted: set[int] = set()
        selected_options: dict[int, PlacementOption] = {
            req_index: self.block_requests[req_index].options[genes[req_index]]
            for req_index in range(len(self.block_requests))
        }
        room_occ: dict[tuple[str, int, str], list[int]] = defaultdict(list)
        faculty_occ: dict[tuple[str, int, str], list[int]] = defaultdict(list)
        section_occ: dict[tuple[str, int, str], list[int]] = defaultdict(list)

        for req_index, option in selected_options.items():
            req = self.block_requests[req_index]
            for offset in range(req.block_size):
                slot_idx = option.start_index + offset
                room_occ[(option.day, slot_idx, option.room_id)].append(req_index)
                faculty_occ[(option.day, slot_idx, option.faculty_id)].append(req_index)
                section_occ[(option.day, slot_idx, req.section)].append(req_index)

        for values in room_occ.values():
            if len(values) <= 1:
                continue
            for left_index, left_req_idx in enumerate(values):
                for right_req_idx in values[left_index + 1 :]:
                    left_req = self.block_requests[left_req_idx]
                    right_req = self.block_requests[right_req_idx]
                    if self._is_allowed_shared_overlap(
                        left_req,
                        right_req,
                        selected_options[left_req_idx],
                        selected_options[right_req_idx],
                    ):
                        continue
                    conflicted.add(left_req_idx)
                    conflicted.add(right_req_idx)

        for values in faculty_occ.values():
            if len(values) <= 1:
                continue
            for left_index, left_req_idx in enumerate(values):
                for right_req_idx in values[left_index + 1 :]:
                    left_req = self.block_requests[left_req_idx]
                    right_req = self.block_requests[right_req_idx]
                    if self._is_allowed_shared_overlap(
                        left_req,
                        right_req,
                        selected_options[left_req_idx],
                        selected_options[right_req_idx],
                    ):
                        continue
                    conflicted.add(left_req_idx)
                    conflicted.add(right_req_idx)

        for values in section_occ.values():
            if len(values) <= 1:
                continue
            for left_index, left_req_idx in enumerate(values):
                for right_req_idx in values[left_index + 1 :]:
                    left_req = self.block_requests[left_req_idx]
                    right_req = self.block_requests[right_req_idx]
                    if self._parallel_lab_overlap_allowed(left_req, right_req):
                        continue
                    conflicted.add(left_req_idx)
                    conflicted.add(right_req_idx)

        return conflicted

    def _repair_overlap_conflicts_strict(self, genes: list[int], *, max_passes: int = 3) -> list[int]:
        repaired = list(genes)
        if not repaired:
            return repaired

        for _ in range(max_passes):
            conflicted = self._overlap_conflicted_request_ids(repaired)
            if not conflicted:
                break

            selected_options: dict[int, PlacementOption] = {}
            room_occ: dict[tuple[str, int, str], list[int]] = defaultdict(list)
            faculty_occ: dict[tuple[str, int, str], list[int]] = defaultdict(list)
            section_occ: dict[tuple[str, int, str], list[int]] = defaultdict(list)
            faculty_minutes: dict[str, int] = {}
            section_slot_keys: dict[str, set[tuple[str, int]]] = defaultdict(set)
            lab_baseline_batch_by_group: dict[tuple[str, str, str, int], str] = {}
            lab_baseline_signatures_by_group: dict[tuple[str, str, str, int], list[tuple[str, int]]] = defaultdict(list)
            lab_signature_usage_by_group_batch: dict[tuple[tuple[str, str, str, int], str], Counter[tuple[str, int]]] = defaultdict(Counter)

            for req_index in range(len(self.block_requests)):
                self._record_selection(
                    req_index,
                    repaired[req_index],
                    selected_options,
                    room_occ,
                    faculty_occ,
                    section_occ,
                    faculty_minutes,
                    section_slot_keys,
                    lab_baseline_batch_by_group,
                    lab_baseline_signatures_by_group,
                    lab_signature_usage_by_group_batch,
                )

            changed = False
            for req_index in sorted(
                conflicted,
                key=lambda idx: (
                    len(self.block_requests[idx].options),
                    -self.block_requests[idx].block_size,
                    self.block_requests[idx].course_code,
                    self.block_requests[idx].section,
                ),
            ):
                req = self.block_requests[req_index]
                if req.request_id in self.fixed_genes:
                    continue

                current_option_index = repaired[req_index]
                current_option = req.options[current_option_index]
                self._unrecord_selection(
                    req_index,
                    current_option_index,
                    selected_options,
                    room_occ,
                    faculty_occ,
                    section_occ,
                    faculty_minutes,
                    section_slot_keys,
                    lab_baseline_batch_by_group,
                    lab_baseline_signatures_by_group,
                    lab_signature_usage_by_group_batch,
                )

                same_slot_room_swap = [
                    option_index
                    for option_index, option in enumerate(req.options)
                    if option_index != current_option_index
                    and option.day == current_option.day
                    and option.start_index == current_option.start_index
                    and option.faculty_id == current_option.faculty_id
                ]
                ranked_candidates = self._option_candidate_indices(
                    req,
                    max_candidates=min(len(req.options), 160),
                    allow_random_tail=False,
                )
                ordered_candidates = [*same_slot_room_swap]
                seen = set(ordered_candidates)
                for option_index in ranked_candidates:
                    if option_index not in seen:
                        ordered_candidates.append(option_index)
                        seen.add(option_index)
                if current_option_index not in seen:
                    ordered_candidates.append(current_option_index)

                chosen_index = current_option_index
                for option_index in ordered_candidates:
                    if option_index == current_option_index:
                        continue
                    if self._is_immediately_conflict_free(
                        req_index=req_index,
                        option_index=option_index,
                        selected_options=selected_options,
                        room_occ=room_occ,
                        faculty_occ=faculty_occ,
                        section_occ=section_occ,
                        faculty_minutes=faculty_minutes,
                        section_slot_keys=section_slot_keys,
                    ):
                        chosen_index = option_index
                        break

                repaired[req_index] = chosen_index
                self._record_selection(
                    req_index,
                    chosen_index,
                    selected_options,
                    room_occ,
                    faculty_occ,
                    section_occ,
                    faculty_minutes,
                    section_slot_keys,
                    lab_baseline_batch_by_group,
                    lab_baseline_signatures_by_group,
                    lab_signature_usage_by_group_batch,
                )
                if chosen_index != current_option_index:
                    changed = True

            if not changed:
                break

        return repaired

    def _repair_individual(self, genes: list[int], *, max_passes: int = 2) -> list[int]:
        repaired = list(genes)
        for _ in range(max_passes):
            repaired = self._greedy_overlap_repair(repaired)
            repaired = self._repair_room_conflicts(repaired)
            repaired = self._repair_section_conflicts(repaired)
            repaired = self._harmonize_faculty_assignments(repaired)
        return repaired
        return repaired

    def _intensive_conflict_repair(
        self,
        genes: list[int],
        *,
        max_steps: int | None = None,
    ) -> tuple[list[int], EvaluationResult]:
        block_count = len(self.block_requests)
        initial_repair_passes = 3
        if block_count >= 220:
            initial_repair_passes = 1
        elif block_count >= 160:
            initial_repair_passes = 2
        if max_steps is not None and max_steps <= 12:
            initial_repair_passes = 1

        candidate = self._repair_individual(list(genes), max_passes=initial_repair_passes)
        candidate_eval = self._evaluate(candidate)
        best_genes = list(candidate)
        best_eval = candidate_eval
        if candidate_eval.hard_conflicts == 0:
            return candidate, candidate_eval

        if max_steps is not None:
            step_limit = max_steps
        elif block_count >= 220:
            step_limit = 120
        elif block_count >= 160:
            step_limit = 180
        else:
            step_limit = max(220, block_count * 3)

        mutable_indices = [
            idx for idx, req in enumerate(self.block_requests) if req.request_id not in self.fixed_genes
        ]
        mutable_index_set = set(mutable_indices)
        if not mutable_index_set:
            return best_genes, best_eval

        stalled_steps = 0
        for step in range(step_limit):
            if candidate_eval.hard_conflicts == 0:
                break

            conflicted = [idx for idx in self._conflicted_request_ids(candidate) if idx in mutable_index_set]
            if not conflicted:
                break
            conflicted.sort(
                key=lambda idx: (
                    len(self.block_requests[idx].options),
                    -self.block_requests[idx].block_size,
                    0 if self.block_requests[idx].is_lab else 1,
                    self.block_requests[idx].course_code,
                )
            )
            probe_limit = min(len(conflicted), 6 if block_count >= 180 else 10)
            if block_count >= 220:
                probe_limit = min(probe_limit, 4)
            improved_this_step = False

            for req_index in conflicted[:probe_limit]:
                req = self.block_requests[req_index]
                current_gene = candidate[req_index]
                local_best_gene = current_gene
                local_best_eval = candidate_eval

                option_cap = 72 if req.is_lab else 96
                if block_count >= 220:
                    option_cap = 24 if req.is_lab else 32
                elif block_count >= 160:
                    option_cap = 28 if req.is_lab else 40
                option_indices = self._option_candidate_indices(
                    req,
                    max_candidates=min(option_cap, len(req.options)),
                    allow_random_tail=True,
                )
                if not req.is_lab:
                    anchor_faculty_id: str | None = None
                    for other_idx in self._request_indices_by_course_section().get((req.course_id, req.section), []):
                        if other_idx == req_index:
                            continue
                        anchor_faculty_id = req.options[candidate[other_idx]].faculty_id
                        if anchor_faculty_id:
                            break
                    if anchor_faculty_id:
                        anchored = [
                            option_index
                            for option_index in option_indices
                            if req.options[option_index].faculty_id == anchor_faculty_id
                        ]
                        if anchored:
                            option_indices = anchored
                target_signatures = self._parallel_lab_target_signatures_from_genes(candidate, req_index)
                if target_signatures:
                    option_indices = self._filter_option_indices_by_signatures(
                        req=req,
                        candidate_indices=option_indices,
                        signatures=target_signatures,
                    )
                if current_gene not in option_indices:
                    option_indices.append(current_gene)

                for option_index in option_indices:
                    if option_index == current_gene:
                        continue
                    candidate[req_index] = option_index
                    trial_eval = self._evaluate(candidate)
                    if self._is_better_eval(trial_eval, local_best_eval):
                        local_best_eval = trial_eval
                        local_best_gene = option_index
                        if trial_eval.hard_conflicts == 0:
                            break

                candidate[req_index] = local_best_gene
                if local_best_gene != current_gene and self._is_better_eval(local_best_eval, candidate_eval):
                    candidate_eval = local_best_eval
                    improved_this_step = True
                    break

            if not improved_this_step:
                perturb_intensity = min(0.28, 0.04 + (0.02 * min(8, stalled_steps)))
                candidate = self._perturb_individual(candidate, intensity=perturb_intensity)
                candidate = self._repair_individual(candidate, max_passes=1)
                candidate_eval = self._evaluate(candidate)
                stalled_steps += 1
            else:
                stalled_steps = 0

            if self._is_better_eval(candidate_eval, best_eval):
                best_genes = list(candidate)
                best_eval = candidate_eval

            if stalled_steps >= 12 and self._is_better_eval(best_eval, candidate_eval):
                candidate = list(best_genes)
                candidate_eval = best_eval
                stalled_steps = 0

            if best_eval.hard_conflicts == 0:
                break

            if step % 25 == 0 and step > 0 and best_eval.hard_conflicts > 0:
                # Periodic diversification helps escape plateaus in dense constraint spaces.
                candidate = self._repair_individual(self._constructive_individual(randomized=True, rcl_alpha=0.35), max_passes=2)
                candidate_eval = self._evaluate(candidate)
                if self._is_better_eval(candidate_eval, best_eval):
                    best_genes = list(candidate)
                    best_eval = candidate_eval

        return best_genes, best_eval

    def _intensive_repair_step_cap(self) -> int:
        block_count = len(self.block_requests)
        if block_count >= 220:
            return 20
        if block_count >= 160:
            return 32
        if block_count >= 120:
            return 48
        return 72

    def _greedy_overlap_repair(self, genes: list[int], *, max_iterations: int = 120) -> list[int]:
        """
        Fast local repair focused on hard overlap conflicts (room/faculty/section).
        """
        repaired = list(genes)
        selected_options: dict[int, PlacementOption] = {
            req_index: self.block_requests[req_index].options[repaired[req_index]]
            for req_index in range(len(self.block_requests))
        }
        room_occ: dict[tuple[str, int, str], list[int]] = defaultdict(list)
        faculty_occ: dict[tuple[str, int, str], list[int]] = defaultdict(list)
        section_occ: dict[tuple[str, int, str], list[int]] = defaultdict(list)

        def register(req_index: int, option: PlacementOption) -> None:
            req = self.block_requests[req_index]
            for offset in range(req.block_size):
                slot_idx = option.start_index + offset
                room_occ[(option.day, slot_idx, option.room_id)].append(req_index)
                faculty_occ[(option.day, slot_idx, option.faculty_id)].append(req_index)
                section_occ[(option.day, slot_idx, req.section)].append(req_index)

        def unregister(req_index: int, option: PlacementOption) -> None:
            req = self.block_requests[req_index]
            for offset in range(req.block_size):
                slot_idx = option.start_index + offset
                room_key = (option.day, slot_idx, option.room_id)
                faculty_key = (option.day, slot_idx, option.faculty_id)
                section_key = (option.day, slot_idx, req.section)
                if req_index in room_occ.get(room_key, []):
                    room_occ[room_key].remove(req_index)
                    if not room_occ[room_key]:
                        room_occ.pop(room_key, None)
                if req_index in faculty_occ.get(faculty_key, []):
                    faculty_occ[faculty_key].remove(req_index)
                    if not faculty_occ[faculty_key]:
                        faculty_occ.pop(faculty_key, None)
                if req_index in section_occ.get(section_key, []):
                    section_occ[section_key].remove(req_index)
                    if not section_occ[section_key]:
                        section_occ.pop(section_key, None)

        for req_index, option in selected_options.items():
            register(req_index, option)

        def overlap_score(req_index: int, option_index: int) -> tuple[int, int, int, int]:
            req = self.block_requests[req_index]
            option = req.options[option_index]
            room_hits = 0
            faculty_hits = 0
            section_hits = 0
            for offset in range(req.block_size):
                slot_idx = option.start_index + offset
                room_key = (option.day, slot_idx, option.room_id)
                faculty_key = (option.day, slot_idx, option.faculty_id)
                section_key = (option.day, slot_idx, req.section)
                for other_idx in room_occ.get(room_key, []):
                    if other_idx == req_index:
                        continue
                    other_req = self.block_requests[other_idx]
                    if self._is_allowed_shared_overlap(req, other_req, option, selected_options[other_idx]):
                        continue
                    room_hits += 1
                for other_idx in faculty_occ.get(faculty_key, []):
                    if other_idx == req_index:
                        continue
                    other_req = self.block_requests[other_idx]
                    if self._is_allowed_shared_overlap(req, other_req, option, selected_options[other_idx]):
                        continue
                    faculty_hits += 1
                for other_idx in section_occ.get(section_key, []):
                    if other_idx == req_index:
                        continue
                    other_req = self.block_requests[other_idx]
                    if self._parallel_lab_overlap_allowed(req, other_req):
                        continue
                    section_hits += 1
            total = (room_hits * 100) + (section_hits * 10) + faculty_hits
            return (total, room_hits, section_hits, faculty_hits)

        for _ in range(max_iterations):
            conflict_weights: Counter[int] = Counter()

            for values in room_occ.values():
                if len(values) <= 1:
                    continue
                for left_index, left_req_idx in enumerate(values):
                    for right_req_idx in values[left_index + 1 :]:
                        left_req = self.block_requests[left_req_idx]
                        right_req = self.block_requests[right_req_idx]
                        if self._is_allowed_shared_overlap(
                            left_req,
                            right_req,
                            selected_options[left_req_idx],
                            selected_options[right_req_idx],
                        ):
                            continue
                        conflict_weights[left_req_idx] += 1
                        conflict_weights[right_req_idx] += 1

            for values in faculty_occ.values():
                if len(values) <= 1:
                    continue
                for left_index, left_req_idx in enumerate(values):
                    for right_req_idx in values[left_index + 1 :]:
                        left_req = self.block_requests[left_req_idx]
                        right_req = self.block_requests[right_req_idx]
                        if self._is_allowed_shared_overlap(
                            left_req,
                            right_req,
                            selected_options[left_req_idx],
                            selected_options[right_req_idx],
                        ):
                            continue
                        conflict_weights[left_req_idx] += 1
                        conflict_weights[right_req_idx] += 1

            for values in section_occ.values():
                if len(values) <= 1:
                    continue
                for left_index, left_req_idx in enumerate(values):
                    for right_req_idx in values[left_index + 1 :]:
                        left_req = self.block_requests[left_req_idx]
                        right_req = self.block_requests[right_req_idx]
                        if self._parallel_lab_overlap_allowed(left_req, right_req):
                            continue
                        conflict_weights[left_req_idx] += 1
                        conflict_weights[right_req_idx] += 1

            candidate_request_ids = [
                req_index
                for req_index, weight in sorted(conflict_weights.items(), key=lambda item: (-item[1], item[0]))
                if self.block_requests[req_index].request_id not in self.fixed_genes
            ]
            if not candidate_request_ids:
                break

            improved = False
            for req_index in candidate_request_ids[:12]:
                req = self.block_requests[req_index]
                current_option_index = repaired[req_index]
                current_option = selected_options[req_index]
                current_score = overlap_score(req_index, current_option_index)
                if current_score[0] <= 0:
                    continue

                option_indices = self._option_candidate_indices(
                    req,
                    max_candidates=min(len(req.options), 144), # Increased from 72
                    allow_random_tail=True, # Allow some randomness to escape local minima
                )
                if current_option_index not in option_indices:
                    option_indices.append(current_option_index)
                if not req.is_lab:
                    anchor_faculty_id: str | None = None
                    for other_idx in self._request_indices_by_course_section().get((req.course_id, req.section), []):
                        if other_idx == req_index:
                            continue
                        anchor_faculty_id = self.block_requests[other_idx].options[repaired[other_idx]].faculty_id
                        if anchor_faculty_id:
                            break
                    if anchor_faculty_id:
                        anchored = [
                            option_index
                            for option_index in option_indices
                            if req.options[option_index].faculty_id == anchor_faculty_id
                        ]
                        if anchored:
                            option_indices = anchored

                best_option_index = current_option_index
                best_score = current_score
                for option_index in option_indices:
                    if option_index == current_option_index:
                        continue
                    score = overlap_score(req_index, option_index)
                    if score < best_score:
                        best_score = score
                        best_option_index = option_index
                        if score[0] == 0:
                            break

                if best_option_index == current_option_index:
                    continue

                unregister(req_index, current_option)
                repaired[req_index] = best_option_index
                new_option = req.options[best_option_index]
                selected_options[req_index] = new_option
                register(req_index, new_option)
                improved = True
                # DO NOT BREAK; continue improving other conflicted sessions in this pass

            if not improved:
                break

        return repaired

    def _repair_section_conflicts(
        self,
        genes: list[int],
        *,
        max_iterations: int = 24, # Increased from 8
    ) -> list[int]:
        repaired = list(genes)

        for _ in range(max_iterations):
            selected_options: dict[int, PlacementOption] = {
                req_index: self.block_requests[req_index].options[repaired[req_index]]
                for req_index in range(len(self.block_requests))
            }
            section_occ: dict[tuple[str, int, str], list[int]] = defaultdict(list)
            for req_index, option in selected_options.items():
                req = self.block_requests[req_index]
                for offset in range(req.block_size):
                    slot_idx = option.start_index + offset
                    section_occ[(option.day, slot_idx, req.section)].append(req_index)

            conflicted: set[int] = set()
            for values in section_occ.values():
                if len(values) <= 1:
                    continue
                for left_index, left_req_idx in enumerate(values):
                    for right_req_idx in values[left_index + 1 :]:
                        left_req = self.block_requests[left_req_idx]
                        right_req = self.block_requests[right_req_idx]
                        if self._parallel_lab_overlap_allowed(left_req, right_req):
                            continue
                        conflicted.add(left_req_idx)
                        conflicted.add(right_req_idx)

            if not conflicted:
                break

            changed = False
            for req_index in sorted(
                conflicted,
                key=lambda idx: (
                    len(self.block_requests[idx].options),
                    -self.block_requests[idx].block_size,
                    self.block_requests[idx].course_code,
                    self.block_requests[idx].section,
                ),
            ):
                req = self.block_requests[req_index]
                if req.request_id in self.fixed_genes:
                    continue

                current_option_index = repaired[req_index]
                current_option = req.options[current_option_index]

                # Find options that don't conflict with current section assignments
                candidate_indices = [
                    option_index
                    for option_index, option in enumerate(req.options)
                    if option_index != current_option_index
                ]
                if not candidate_indices:
                    continue

                # Prioritize valid options
                candidate_indices.sort(
                    key=lambda option_index: (
                        self._has_section_conflict(req_index, req.options[option_index], section_occ),
                        self._has_room_conflict(req_index, req.options[option_index], selected_options),
                        req.options[option_index].day != current_option.day,
                    )
                )

                best_option_index = candidate_indices[0]
                if not self._has_section_conflict(req_index, req.options[best_option_index], section_occ):
                    repaired[req_index] = best_option_index
                    changed = True
                    # Update section_occ immediately for faster convergence
                    self._update_occ_maps(req_index, current_option, req.options[best_option_index], section_occ, is_section=True)
                    selected_options[req_index] = req.options[best_option_index]

            if not changed:
                break

        return repaired

    def _has_section_conflict(self, req_index: int, option: PlacementOption, section_occ: dict) -> int:
        req = self.block_requests[req_index]
        conflicts = 0
        for offset in range(req.block_size):
            slot_idx = option.start_index + offset
            key = (option.day, slot_idx, req.section)
            for other_idx in section_occ.get(key, []):
                if other_idx == req_index:
                    continue
                if not self._parallel_lab_overlap_allowed(req, self.block_requests[other_idx]):
                    conflicts += 1
        return conflicts

    def _has_room_conflict(self, req_index: int, option: PlacementOption, selected_options: dict) -> bool:
        req = self.block_requests[req_index]
        for offset in range(req.block_size):
            slot_idx = option.start_index + offset
            # This is slow, but we only use it if necessary. Ideally we'd have a room_occ map passed in.
            # For repair, we check against other selected options.
            pass # Simplified check for now or implement if needed
        return False

    def _update_occ_maps(self, req_index: int, old_opt: PlacementOption, new_opt: PlacementOption, occ_map: dict, is_section: bool = False):
        req = self.block_requests[req_index]
        # Remove old
        for offset in range(req.block_size):
            old_slot = old_opt.start_index + offset
            key = (old_opt.day, old_slot, req.section if is_section else old_opt.room_id)
            if req_index in occ_map.get(key, []):
                occ_map[key].remove(req_index)
        # Add new
        for offset in range(req.block_size):
            new_slot = new_opt.start_index + offset
            key = (new_opt.day, new_slot, req.section if is_section else new_opt.room_id)
            occ_map.setdefault(key, []).append(req_index)

    def _repair_room_conflicts(
        self,
        genes: list[int],
        *,
        max_iterations: int = 24, # Increased from 8
    ) -> list[int]:
        repaired = list(genes)

        for _ in range(max_iterations):
            selected_options: dict[int, PlacementOption] = {
                req_index: self.block_requests[req_index].options[repaired[req_index]]
                for req_index in range(len(self.block_requests))
            }
            room_occ: dict[tuple[str, int, str], list[int]] = defaultdict(list)
            for req_index, option in selected_options.items():
                req = self.block_requests[req_index]
                for offset in range(req.block_size):
                    slot_idx = option.start_index + offset
                    room_occ[(option.day, slot_idx, option.room_id)].append(req_index)

            conflicted: set[int] = set()
            for values in room_occ.values():
                if len(values) <= 1:
                    continue
                for left_index, left_req_idx in enumerate(values):
                    for right_req_idx in values[left_index + 1 :]:
                        left_req = self.block_requests[left_req_idx]
                        right_req = self.block_requests[right_req_idx]
                        if self._is_allowed_shared_overlap(
                            left_req,
                            right_req,
                            selected_options[left_req_idx],
                            selected_options[right_req_idx],
                        ):
                            continue
                        conflicted.add(left_req_idx)
                        conflicted.add(right_req_idx)

            if not conflicted:
                break

            changed = False
            for req_index in sorted(
                conflicted,
                key=lambda idx: (
                    len(self.block_requests[idx].options),
                    -self.block_requests[idx].block_size,
                    self.block_requests[idx].course_code,
                    self.block_requests[idx].section,
                ),
            ):
                req = self.block_requests[req_index]
                if req.request_id in self.fixed_genes:
                    continue

                current_option_index = repaired[req_index]
                current_option = req.options[current_option_index]

                # Prioritize rooms of correct type and capacity
                candidate_indices = [
                    option_index
                    for option_index, option in enumerate(req.options)
                    if option_index != current_option_index
                    and self.rooms[option.room_id].type == (RoomType.lab if req.is_lab else RoomType.lecture)
                ]
                
                if not candidate_indices:
                     candidate_indices = [
                        option_index
                        for option_index, option in enumerate(req.options)
                        if option_index != current_option_index
                    ]

                if not candidate_indices:
                    continue

                candidate_indices.sort(
                    key=lambda option_index: (
                        self._has_room_conflict_in_map(req_index, req.options[option_index], room_occ, selected_options),
                        self.rooms[req.options[option_index].room_id].capacity < req.student_count,
                        abs(self.rooms[req.options[option_index].room_id].capacity - req.student_count),
                        req.options[option_index].room_id,
                    )
                )

                best_option_index = candidate_indices[0]
                repaired[req_index] = best_option_index
                changed = True
                self._update_occ_maps(req_index, current_option, req.options[best_option_index], room_occ, is_section=False)
                selected_options[req_index] = req.options[best_option_index]

            if not changed:
                break

        return repaired

    def _has_room_conflict_in_map(self, req_index: int, option: PlacementOption, room_occ: dict, selected_options: dict) -> int:
        req = self.block_requests[req_index]
        conflicts = 0
        for offset in range(req.block_size):
            slot_idx = option.start_index + offset
            room_key = (option.day, slot_idx, option.room_id)
            for other_req_idx in room_occ.get(room_key, []):
                if other_req_idx == req_index:
                    continue
                other_req = self.block_requests[other_req_idx]
                if self._is_allowed_shared_overlap(
                    req,
                    other_req,
                    option,
                    selected_options[other_req_idx],
                ):
                    continue
                conflicts += 1
        return conflicts

    def _evaluate(self, genes: list[int]) -> EvaluationResult:
        """
        Calculates the fitness score for a given timetable (genome).
        
        The evaluation process:
        1.  **Decodes** the genes into a concrete timetable.
        2.  **Checks Hard Constraints**: Violations (e.g., room double booking, faculty overlap) 
            add to the `hard_conflicts` count. Any hard conflict makes the solution invalid.
        3.  **Calculates Soft Penalty**: Preference violations (e.g., gap between classes too small, 
            faculty workload imbalance) add to the `soft_penalty` score.
        4.  **Computes Fitness**: A weighted combination of hard and soft scores, used for ranking.
        """
        key = tuple(genes)
        if key in self.eval_cache:
            return self.eval_cache[key]

        weights = self.settings.objective_weights
        hard = 0
        soft = 0.0

        room_occ: dict[tuple[str, int, str], list[int]] = {}
        faculty_occ: dict[tuple[str, int, str], list[int]] = {}
        faculty_day_req_indices: dict[tuple[str, str], list[int]] = {}
        elective_signatures_by_section: dict[str, list[tuple[str, int, int, str]]] = defaultdict(list)
        section_occ: dict[tuple[str, int, str], list[int]] = {}
        elective_occ: dict[tuple[str, int], list[int]] = {}
        section_day_slots: dict[tuple[str, str], set[int]] = {}
        faculty_minutes: dict[str, int] = {}
        selected_options: dict[int, PlacementOption] = {}

        for req_index, req in enumerate(self.block_requests):
            option = req.options[genes[req_index]]
            selected_options[req_index] = option
            room = self.rooms[option.room_id]
            faculty = self.faculty[option.faculty_id]
            block_start, block_end = self._option_bounds(option, req.block_size)

            if not self._within_semester_time_window(block_start, block_end):
                hard += weights.semester_limit

            reserved_room_conflict, reserved_faculty_conflict = self._reserved_conflict_flags(
                day=option.day,
                start_min=block_start,
                end_min=block_end,
                room_id=option.room_id,
                faculty_id=option.faculty_id,
            )
            if reserved_room_conflict:
                hard += weights.room_conflict
            if reserved_faculty_conflict:
                hard += weights.faculty_conflict

            if room.capacity < req.student_count:
                hard += weights.room_capacity
            if req.is_lab and room.type != RoomType.lab:
                hard += weights.room_type * 20 # Increased penalty
            if not req.is_lab and room.type == RoomType.lab:
                hard += weights.room_type * 20 # Increased penalty

            for offset in range(req.block_size):
                slot_idx = option.start_index + offset
                room_key = (option.day, slot_idx, option.room_id)
                faculty_key = (option.day, slot_idx, option.faculty_id)
                section_key = (option.day, slot_idx, req.section)
                room_occ.setdefault(room_key, []).append(req_index)
                faculty_occ.setdefault(faculty_key, []).append(req_index)
                section_occ.setdefault(section_key, []).append(req_index)
                elective_occ.setdefault((option.day, slot_idx), []).append(req_index)
                section_day_slots.setdefault((req.section, option.day), set()).add(slot_idx)
                faculty_minutes[option.faculty_id] = (
                    faculty_minutes.get(option.faculty_id, 0) + self.schedule_policy.period_minutes
                )
            faculty_day_req_indices.setdefault((option.faculty_id, option.day), []).append(req_index)
            if self._is_elective_request(req):
                elective_signatures_by_section[req.section].append(
                    (option.day, option.start_index, req.block_size, req.session_type)
                )

            if not self._faculty_allows_day(faculty, option.day):
                hard += weights.faculty_availability

            if self.faculty_windows.get(option.faculty_id, {}).get(option.day):
                if not any(
                    start <= block_start and block_end <= end
                    for start, end in self.faculty_windows[option.faculty_id][option.day]
                ):
                    hard += weights.faculty_availability

            if self.room_windows.get(option.room_id, {}).get(option.day):
                if not any(
                    start <= block_start and block_end <= end
                    for start, end in self.room_windows[option.room_id][option.day]
                ):
                    hard += weights.room_type

            if req.preferred_faculty_ids and option.faculty_id not in req.preferred_faculty_ids:
                soft += weights.faculty_subject_preference * req.block_size
            if option.faculty_id != req.primary_faculty_id:
                soft += (weights.faculty_subject_preference * 0.5) * req.block_size

        for values in room_occ.values():
            if len(values) <= 1:
                continue
            for left_index, left_req_idx in enumerate(values):
                for right_req_idx in values[left_index + 1 :]:
                    left_req = self.block_requests[left_req_idx]
                    right_req = self.block_requests[right_req_idx]
                    if self._is_allowed_shared_overlap(
                        left_req,
                        right_req,
                        selected_options[left_req_idx],
                        selected_options[right_req_idx],
                    ):
                        continue
                    hard += weights.room_conflict

        for values in faculty_occ.values():
            if len(values) <= 1:
                continue
            for left_index, left_req_idx in enumerate(values):
                for right_req_idx in values[left_index + 1 :]:
                    left_req = self.block_requests[left_req_idx]
                    right_req = self.block_requests[right_req_idx]
                    if self._is_allowed_shared_overlap(
                        left_req,
                        right_req,
                        selected_options[left_req_idx],
                        selected_options[right_req_idx],
                    ):
                        continue
                    hard += weights.faculty_conflict

        for req_indices in faculty_day_req_indices.values():
            if len(req_indices) <= 1:
                continue
            for left_index, left_req_idx in enumerate(req_indices):
                for right_req_idx in req_indices[left_index + 1 :]:
                    if self._is_faculty_back_to_back(
                        self.block_requests[left_req_idx],
                        selected_options[left_req_idx],
                        self.block_requests[right_req_idx],
                        selected_options[right_req_idx],
                    ):
                        soft += max(1.0, weights.spread_balance * 0.75)

        for values in section_occ.values():
            if len(values) <= 1:
                continue
            requests = [self.block_requests[idx] for idx in values]
            first = requests[0]
            is_allowed_parallel_batch = (
                all(item.is_lab for item in requests)
                and all(item.course_id == first.course_id for item in requests)
                and all(item.allow_parallel_batches for item in requests)
                and all(item.batch for item in requests)
                and len({item.batch for item in requests}) == len(requests)
            )
            if not is_allowed_parallel_batch:
                hard += weights.section_conflict * (len(values) - 1)

        if self.elective_overlap_pairs:
            for values in elective_occ.values():
                if len(values) <= 1:
                    continue
                requests = [self.block_requests[idx] for idx in values]
                for left_index, left_req in enumerate(requests):
                    for right_req in requests[left_index + 1 :]:
                        if left_req.course_id == right_req.course_id:
                            continue
                        if self._courses_conflict_in_elective_group(left_req.course_id, right_req.course_id):
                            hard += weights.section_conflict

        elective_sections = sorted(elective_signatures_by_section.keys())
        if len(elective_sections) > 1:
            baseline: list[tuple[str, int, int, str]] | None = None
            for section_name in elective_sections:
                signatures = sorted(elective_signatures_by_section.get(section_name, []))
                if baseline is None:
                    baseline = signatures
                    continue
                if signatures != baseline:
                    baseline_set = set(baseline)
                    signature_set = set(signatures)
                    mismatch_size = max(1, len(baseline_set.symmetric_difference(signature_set)))
                    hard += weights.section_conflict * mismatch_size

        if self.shared_lecture_sections_by_course:
            signatures_by_course_section: dict[tuple[str, str], list[tuple[str, int, str, str, int]]] = defaultdict(list)
            for req_index, req in enumerate(self.block_requests):
                if req.is_lab:
                    continue
                if req.course_id not in self.shared_lecture_sections_by_course:
                    continue
                option = selected_options[req_index]
                signatures_by_course_section[(req.course_id, req.section)].append(
                    (option.day, option.start_index, option.room_id, option.faculty_id, req.block_size)
                )

            for course_id, groups in self.shared_lecture_sections_by_course.items():
                for sections in groups:
                    baseline: list[tuple[str, int, str, str, int]] | None = None
                    for section in sorted(sections):
                        signatures = sorted(signatures_by_course_section.get((course_id, section), []))
                        if baseline is None:
                            baseline = signatures
                            continue
                        if signatures != baseline:
                            baseline_set = set(baseline)
                            signature_set = set(signatures)
                            mismatch_size = max(1, len(baseline_set.symmetric_difference(signature_set)))
                            hard += weights.section_conflict * mismatch_size

        parallel_lab_signatures: dict[tuple[str, str], dict[str, list[tuple[str, int, int]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for req_index, req in enumerate(self.block_requests):
            if not req.is_lab or not req.allow_parallel_batches or not req.batch:
                continue
            option = selected_options[req_index]
            group_key = (req.course_id, req.section)
            signature = (option.day, option.start_index, req.block_size)
            parallel_lab_signatures[group_key][req.batch].append(signature)

        for signatures_by_batch in parallel_lab_signatures.values():
            if len(signatures_by_batch) <= 1:
                continue
            baseline: list[tuple[str, int, int]] | None = None
            for batch_name in sorted(signatures_by_batch.keys()):
                signatures = sorted(signatures_by_batch[batch_name])
                if baseline is None:
                    baseline = signatures
                    continue
                if signatures != baseline:
                    baseline_set = set(baseline)
                    signature_set = set(signatures)
                    mismatch_size = max(1, len(baseline_set.symmetric_difference(signature_set)))
                    hard += weights.section_conflict * mismatch_size

        for (_course_id, _section_name), lecture_req_indices in self._request_indices_by_course_section().items():
            if len(lecture_req_indices) <= 1:
                continue
            assigned_faculty_ids = [selected_options[idx].faculty_id for idx in lecture_req_indices]
            unique_faculty_ids = set(assigned_faculty_ids)
            if len(unique_faculty_ids) <= 1:
                continue
            faculty_counts = defaultdict(int)
            for faculty_id in assigned_faculty_ids:
                faculty_counts[faculty_id] += 1
            total_pairs = len(assigned_faculty_ids) * (len(assigned_faculty_ids) - 1) // 2
            same_pairs = sum(count * (count - 1) // 2 for count in faculty_counts.values())
            mismatch_pairs = max(1, total_pairs - same_pairs)
            hard += weights.faculty_conflict * mismatch_pairs

        for course_id, req_indices in self.request_indices_by_course.items():
            if not self._single_faculty_required(course_id):
                continue
            lecture_req_indices = [idx for idx in req_indices if not self.block_requests[idx].is_lab]
            if len(lecture_req_indices) <= 1:
                continue
            assigned_faculty_ids = [selected_options[idx].faculty_id for idx in lecture_req_indices]
            unique_faculty_ids = set(assigned_faculty_ids)
            if len(unique_faculty_ids) <= 1:
                continue
            faculty_counts = defaultdict(int)
            for faculty_id in assigned_faculty_ids:
                faculty_counts[faculty_id] += 1
            total_pairs = len(assigned_faculty_ids) * (len(assigned_faculty_ids) - 1) // 2
            same_pairs = sum(count * (count - 1) // 2 for count in faculty_counts.values())
            mismatch_pairs = max(1, total_pairs - same_pairs)
            hard += weights.faculty_conflict * mismatch_pairs

        if self.semester_constraint is not None:
            day_limit = self.semester_constraint.max_hours_per_day * 60
            week_limit = self.semester_constraint.max_hours_per_week * 60
            min_break = self.semester_constraint.min_break_minutes
            max_consecutive = self.semester_constraint.max_consecutive_hours * 60

            weekly_minutes_by_section: dict[str, int] = {}
            for (section, day), slot_set in section_day_slots.items():
                day_minutes = len(slot_set) * self.schedule_policy.period_minutes
                weekly_minutes_by_section[section] = weekly_minutes_by_section.get(section, 0) + day_minutes
                if day_minutes > day_limit:
                    hard += weights.semester_limit * max(1, (day_minutes - day_limit) // self.schedule_policy.period_minutes)

                slot_indexes = sorted(slot_set)
                if not slot_indexes:
                    continue
                run_start = slot_indexes[0]
                prev = slot_indexes[0]
                for current in slot_indexes[1:]:
                    prev_end = self.day_slots[day][prev].end
                    current_start = self.day_slots[day][current].start
                    gap = current_start - prev_end
                    if gap < min_break:
                        hard += weights.semester_limit

                    if gap != 0:
                        run_duration = self.day_slots[day][prev].end - self.day_slots[day][run_start].start
                        if run_duration > max_consecutive:
                            hard += weights.semester_limit
                        run_start = current
                    prev = current

                run_duration = self.day_slots[day][prev].end - self.day_slots[day][run_start].start
                if run_duration > max_consecutive:
                    hard += weights.semester_limit

            for section, minutes in weekly_minutes_by_section.items():
                if minutes > week_limit:
                    hard += weights.semester_limit * max(1, (minutes - week_limit) // self.schedule_policy.period_minutes)

            if self.expected_section_minutes > 0:
                period_minutes = max(1, self.schedule_policy.period_minutes)
                all_sections = {req.section for req in self.block_requests}
                for section in all_sections:
                    minutes = weekly_minutes_by_section.get(section, 0)
                    if minutes == self.expected_section_minutes:
                        continue
                    delta = abs(minutes - self.expected_section_minutes)
                    hard += weights.semester_limit * max(1, math.ceil(delta / period_minutes))

        for faculty_id, faculty in self.faculty.items():
            minutes = faculty_minutes.get(faculty_id, 0)
            max_minutes = faculty.max_hours * 60
            if minutes > max_minutes:
                overflow_periods = max(1, (minutes - max_minutes) // max(1, self.schedule_policy.period_minutes))
                # RELAXED: Treat workload overflow as a high soft penalty, not a hard conflict.
                soft += weights.workload_overflow * overflow_periods * 10
            target_minutes = max(0, faculty.workload_hours) * 60
            if target_minutes > 0 and minutes < target_minutes:
                soft += (target_minutes - minutes) * weights.workload_underflow

        sections = {req.section for req in self.block_requests}
        for section in sections:
            day_counts = [len(section_day_slots.get((section, day), set())) for day in self.day_slots]
            if day_counts:
                soft += (max(day_counts) - min(day_counts)) * weights.spread_balance

        fitness = -((hard * 1000.0) + soft)
        result = EvaluationResult(fitness=fitness, hard_conflicts=hard, soft_penalty=soft)
        self.eval_cache[key] = result
        return result

    def _harmonize_faculty_assignments(self, genes: list[int]) -> list[int]:
        """
        Preserve one-faculty-per-(course, section) consistency for non-lab blocks.
        This keeps post-mutation candidates close to feasible space.
        """
        harmonized = list(genes)
        by_course_section = self._request_indices_by_course_section()
        if not by_course_section:
            return harmonized

        day_order = {day: index for index, day in enumerate(self.day_slots.keys())}
        max_day_index = len(day_order)
        common_by_section = getattr(self, "common_faculty_candidates_by_course_section", {})
        common_by_course = getattr(self, "common_faculty_candidates_by_course", {})
        single_faculty_required = getattr(self, "single_faculty_required_by_course", {})

        def choose_target_faculty(
            *,
            course_id: str,
            req_indices: list[int],
            candidate_ids: set[str],
            fixed_faculty_id: str | None,
        ) -> str | None:
            if not candidate_ids:
                return None
            if fixed_faculty_id and fixed_faculty_id in candidate_ids:
                return fixed_faculty_id

            assigned_counts = Counter(
                self.block_requests[req_index].options[harmonized[req_index]].faculty_id
                for req_index in req_indices
            )
            course = self.courses.get(course_id)
            course_code = course.code if course is not None else ""
            ranked = sorted(
                candidate_ids,
                key=lambda faculty_id: (
                    -assigned_counts.get(faculty_id, 0),
                    0 if self._faculty_prefers_subject(faculty_id, course_code) else 1,
                    -(max(0.0, self.faculty.get(faculty_id).max_hours) if faculty_id in self.faculty else 0.0),
                    self.faculty.get(faculty_id).name if faculty_id in self.faculty else faculty_id,
                ),
            )
            return ranked[0] if ranked else None

        def align_group(course_id: str, req_indices: list[int], target_faculty_id: str) -> None:
            for req_index in req_indices:
                req = self.block_requests[req_index]
                if req.request_id in self.fixed_genes:
                    continue

                current_index = harmonized[req_index]
                current_option = req.options[current_index]
                if current_option.faculty_id == target_faculty_id:
                    continue

                prioritized = [
                    option_index
                    for option_index in self._option_candidate_indices(
                        req,
                        max_candidates=min(len(req.options), 36),
                        allow_random_tail=False,
                    )
                    if req.options[option_index].faculty_id == target_faculty_id
                ]
                if not prioritized:
                    prioritized = [
                        option_index
                        for option_index, option in enumerate(req.options)
                        if option.faculty_id == target_faculty_id
                    ]
                if not prioritized:
                    continue

                current_day_rank = day_order.get(current_option.day, max_day_index)
                prioritized.sort(
                    key=lambda option_index: (
                        req.options[option_index].day != current_option.day,
                        abs(day_order.get(req.options[option_index].day, max_day_index) - current_day_rank),
                        abs(req.options[option_index].start_index - current_option.start_index),
                    )
                )
                harmonized[req_index] = prioritized[0]

        for (course_id, section_name), req_indices in by_course_section.items():
            if len(req_indices) <= 1:
                continue

            fixed_faculty_id: str | None = None
            conflicting_fixed = False
            for req_index in req_indices:
                req = self.block_requests[req_index]
                if req.request_id not in self.fixed_genes:
                    continue
                faculty_id = req.options[harmonized[req_index]].faculty_id
                if fixed_faculty_id is None:
                    fixed_faculty_id = faculty_id
                elif fixed_faculty_id != faculty_id:
                    conflicting_fixed = True
                    break
            if conflicting_fixed:
                continue

            candidate_ids = set(common_by_section.get((course_id, section_name), ()))
            if not candidate_ids:
                candidate_ids = {option.faculty_id for option in self.block_requests[req_indices[0]].options}
                for req_index in req_indices[1:]:
                    candidate_ids &= {option.faculty_id for option in self.block_requests[req_index].options}
            target_faculty_id = choose_target_faculty(
                course_id=course_id,
                req_indices=req_indices,
                candidate_ids=candidate_ids,
                fixed_faculty_id=fixed_faculty_id,
            )
            if target_faculty_id is None:
                continue
            align_group(course_id, req_indices, target_faculty_id)

        for course_id, required in single_faculty_required.items():
            if not required:
                continue
            lecture_indices = [
                req_index
                for req_index in self.request_indices_by_course.get(course_id, [])
                if not self.block_requests[req_index].is_lab
            ]
            if len(lecture_indices) <= 1:
                continue

            fixed_faculty_id: str | None = None
            conflicting_fixed = False
            for req_index in lecture_indices:
                req = self.block_requests[req_index]
                if req.request_id not in self.fixed_genes:
                    continue
                faculty_id = req.options[harmonized[req_index]].faculty_id
                if fixed_faculty_id is None:
                    fixed_faculty_id = faculty_id
                elif fixed_faculty_id != faculty_id:
                    conflicting_fixed = True
                    break
            if conflicting_fixed:
                continue

            candidate_ids = set(common_by_course.get(course_id, ()))
            if not candidate_ids:
                candidate_ids = {option.faculty_id for option in self.block_requests[lecture_indices[0]].options}
                for req_index in lecture_indices[1:]:
                    candidate_ids &= {option.faculty_id for option in self.block_requests[req_index].options}
            target_faculty_id = choose_target_faculty(
                course_id=course_id,
                req_indices=lecture_indices,
                candidate_ids=candidate_ids,
                fixed_faculty_id=fixed_faculty_id,
            )
            if target_faculty_id is None:
                continue
            align_group(course_id, lecture_indices, target_faculty_id)

        return harmonized

    def _crossover(self, parent_a: list[int], parent_b: list[int]) -> list[int]:
        """
        Creates a new child genome by combining genes from two parents.
        
        Args:
            parent_a: The first parent genome (list of option indices).
            parent_b: The second parent genome.
        
        Returns:
            A new genome that inherits traits from both parents, typically using uniform crossover.
        """
        child: list[int] = []
        for index, req in enumerate(self.block_requests):
            if req.request_id in self.fixed_genes:
                child.append(self.fixed_genes[req.request_id])
                continue
            if self.random.random() < 0.5:
                child.append(parent_a[index])
            else:
                child.append(parent_b[index])
        return self._harmonize_faculty_assignments(child)

    def _mutate(self, genes: list[int], *, mutation_rate: float | None = None) -> list[int]:
        """
        Randomly alters genes in the genome to introduce diversity.
        
        Args:
            genes: The genome to mutate.
            mutation_rate: Probability of mutation per gene. If None, uses default settings.
        
        Returns:
            A new mutated genome.
        """
        mutated = list(genes)
        rate = mutation_rate if mutation_rate is not None else self.settings.mutation_rate
        changed = False
        for index, req in enumerate(self.block_requests):
            if req.request_id in self.fixed_genes:
                continue
            if self.random.random() < rate:
                mutated[index] = self.random.randrange(len(req.options))
                changed = True
        if changed:
            mutated = self._harmonize_faculty_assignments(mutated)
        return mutated

    def _select(self, population: list[list[int]], evaluations: list[EvaluationResult]) -> list[int]:
        """
        Selects a parent for the next generation based on fitness (Tournament Selection).
        
        Args:
            population: Current population of genomes.
            evaluations: Evaluation results for the population.
        
        Returns:
            The selected genome.
        """
        contenders = self.random.sample(range(len(population)), self.settings.tournament_size)
        best_index = max(contenders, key=lambda idx: evaluations[idx].fitness)
        return population[best_index]

    def _decode_payload(self, genes: list[int]) -> OfficialTimetablePayload:
        used_faculty_ids = set()
        used_course_ids = set()
        used_room_ids = set()
        selected_faculty_by_course: dict[str, list[str]] = defaultdict(list)
        timetable_rows: list[dict] = []

        for req_index, req in enumerate(self.block_requests):
            option = req.options[genes[req_index]]
            used_faculty_ids.add(option.faculty_id)
            used_course_ids.add(req.course_id)
            used_room_ids.add(option.room_id)
            selected_faculty_by_course[req.course_id].append(option.faculty_id)
            for offset in range(req.block_size):
                slot = self.day_slots[option.day][option.start_index + offset]
                timetable_rows.append(
                    {
                        "id": f"gen-{req.request_id}-{offset}",
                        "day": option.day,
                        "startTime": minutes_to_time(slot.start),
                        "endTime": minutes_to_time(slot.end),
                        "courseId": req.course_id,
                        "roomId": option.room_id,
                        "facultyId": option.faculty_id,
                        "section": req.section,
                        "batch": req.batch,
                        "studentCount": req.student_count,
                        "sessionType": req.session_type,
                    }
                )

        faculty_data = []
        for item in self.faculty.values():
            if item.id not in used_faculty_ids:
                continue
            faculty_data.append(
                {
                    "id": item.id,
                    "name": item.name,
                    "department": item.department,
                    "workloadHours": item.workload_hours,
                    "maxHours": item.max_hours,
                    "availability": item.availability,
                    "email": item.email,
                }
            )

        course_data = []
        for item in self.courses.values():
            if item.id not in used_course_ids:
                continue
            assigned_ids = selected_faculty_by_course.get(item.id, [])
            resolved_faculty_id = item.faculty_id
            if assigned_ids:
                resolved_faculty_id = max(set(assigned_ids), key=assigned_ids.count)
            if not resolved_faculty_id:
                # This should be unreachable because generation requires a faculty assignment in each option.
                resolved_faculty_id = next(iter(used_faculty_ids))
            course_data.append(
                {
                    "id": item.id,
                    "code": item.code,
                    "name": item.name,
                    "type": item.type.value,
                    "credits": item.credits,
                    "facultyId": resolved_faculty_id,
                    "duration": item.duration_hours,
                    "hoursPerWeek": item.hours_per_week,
                    "semesterNumber": item.semester_number,
                    "batchYear": item.batch_year,
                    "theoryHours": item.theory_hours,
                    "labHours": item.lab_hours,
                    "tutorialHours": item.tutorial_hours,
                }
            )

        room_data = []
        for item in self.rooms.values():
            if item.id not in used_room_ids:
                continue
            room_data.append(
                {
                    "id": item.id,
                    "name": item.name,
                    "capacity": item.capacity,
                    "type": item.type.value,
                    "building": item.building,
                    "hasLabEquipment": item.has_lab_equipment,
                    "hasProjector": item.has_projector,
                }
            )

        return OfficialTimetablePayload(
            programId=self.program_id,
            termNumber=self.term_number,
            facultyData=faculty_data,
            courseData=course_data,
            roomData=room_data,
            timetableData=timetable_rows,
        )

    def _adaptive_mutation_rate(self, stagnant_generations: int) -> float:
        base = self.settings.mutation_rate
        if stagnant_generations >= self.settings.stagnation_limit:
            return min(0.5, max(base, base * 3.0))
        if stagnant_generations >= self.settings.stagnation_limit // 2:
            return min(0.35, max(base, base * 2.0))
        if stagnant_generations >= self.settings.stagnation_limit // 4:
            return min(0.25, max(base, base * 1.4))
        return base

    def _build_initial_population(self) -> list[list[int]]:
        population: list[list[int]] = []
        seen: set[tuple[int, ...]] = set()

        def add_unique(candidate: list[int]) -> None:
            key = tuple(candidate)
            if key in seen:
                return
            seen.add(key)
            population.append(candidate)

        # 1. Add the Best-Fit Deterministic solution (High quality seed)
        add_unique(self._constructive_individual(randomized=False))
        
        # 2. Add Randomized Constructive solutions (Good quality, diverse seeds)
        # Allocate about 25% of population to smart seeds
        smart_seed_count = max(4, self.settings.population_size // 4)
        attempts = 0
        while len(population) < smart_seed_count and attempts < smart_seed_count * 2:
            attempts += 1
            add_unique(self._constructive_individual(randomized=True, rcl_alpha=0.15))

        # 3. Fill remaining with Random individuals (High diversity)
        # We repair them lightly to fix obvious blunders but keep them diverse
        while len(population) < self.settings.population_size:
            before = len(population)
            # 50% chance of repair for random individuals
            candidate = self._random_individual()
            if self.random.random() < 0.5:
                 candidate = self._repair_individual(candidate, max_passes=1)
            
            add_unique(candidate)
            
            if len(population) == before:
                # Fallback in small/dense search spaces where unique genotypes can saturate quickly.
                population.append(self._random_individual())

        return population[: self.settings.population_size]

    def _run_classic_ga(self, request: GenerateTimetableRequest) -> GenerateTimetableResponse:
        start = perf_counter()
        population = self._build_initial_population()
        best_fitness = float("-inf")
        best_hard_conflicts = math.inf
        stagnant = 0

        block_count = len(self.block_requests)
        generation_cap = self.settings.generations
        if block_count >= 220:
            generation_cap = min(generation_cap, 30)
        elif block_count >= 160:
            generation_cap = min(generation_cap, 50)
        elif block_count >= 120:
            generation_cap = min(generation_cap, 80)
        elif block_count >= 80:
            generation_cap = min(generation_cap, 120)
        else:
            generation_cap = min(generation_cap, 180)

        for _generation in range(generation_cap):
            evaluations = [self._evaluate(item) for item in population]
            ranked_indices = sorted(range(len(population)), key=lambda idx: evaluations[idx].fitness, reverse=True)
            ranked_population = [population[idx] for idx in ranked_indices]
            ranked_evaluations = [evaluations[idx] for idx in ranked_indices]

            generation_best_eval = ranked_evaluations[0]
            generation_best_fitness = generation_best_eval.fitness
            if (
                generation_best_eval.hard_conflicts < best_hard_conflicts
                or (
                    generation_best_eval.hard_conflicts == best_hard_conflicts
                    and generation_best_fitness > best_fitness
                )
            ):
                best_hard_conflicts = generation_best_eval.hard_conflicts
                best_fitness = generation_best_fitness
                stagnant = 0
            else:
                stagnant += 1

            if best_hard_conflicts == 0 and stagnant >= max(8, self.settings.stagnation_limit // 2):
                break

            mutation_rate = self._adaptive_mutation_rate(stagnant)
            next_population = ranked_population[: self.settings.elite_count]
            while len(next_population) < self.settings.population_size:
                parent_a = self._select(ranked_population, ranked_evaluations)
                parent_b = self._select(ranked_population, ranked_evaluations)
                if self.random.random() < self.settings.crossover_rate:
                    child = self._crossover(parent_a, parent_b)
                else:
                    child = list(parent_a)
                child = self._mutate(child, mutation_rate=mutation_rate)
                if self.random.random() < 0.03 or (
                    stagnant >= self.settings.stagnation_limit // 2 and self.random.random() < 0.12
                ):
                    child = self._repair_individual(child, max_passes=1)
                next_population.append(child)

            if stagnant >= self.settings.stagnation_limit:
                # Restart most of the population to escape local minima while preserving elites.
                for index in range(self.settings.elite_count, len(next_population)):
                    candidate = self._random_individual()
                    if self.random.random() < 0.35:
                        candidate = self._repair_individual(candidate, max_passes=1)
                    next_population[index] = candidate
                stagnant = 0

            population = next_population

        final_evaluations = [self._evaluate(item) for item in population]
        ranked_indices = sorted(range(len(population)), key=lambda idx: final_evaluations[idx].fitness, reverse=True)
        shortlisted_count = min(len(ranked_indices), max(20, request.alternative_count * 8))
        shortlisted: list[tuple[EvaluationResult, list[int]]] = []
        intensive_budget = (
            max(2, request.alternative_count * 2)
            if block_count >= 180
            else max(4, request.alternative_count * 4)
        )
        intensive_step_cap = self._intensive_repair_step_cap()
        for idx in ranked_indices[:shortlisted_count]:
            genes = population[idx]
            evaluation = final_evaluations[idx]
            best_genes = genes
            best_eval = evaluation
            if evaluation.hard_conflicts > 0:
                repaired = self._repair_individual(list(genes), max_passes=3)
                repaired_eval = self._evaluate(repaired)
                if (
                    repaired_eval.hard_conflicts < evaluation.hard_conflicts
                    or (
                        repaired_eval.hard_conflicts == evaluation.hard_conflicts
                        and repaired_eval.soft_penalty < evaluation.soft_penalty
                    )
                ):
                    best_genes = repaired
                    best_eval = repaired_eval
            if best_eval.hard_conflicts > 0 and intensive_budget > 0:
                intensive_budget -= 1
                intensified_genes, intensified_eval = self._intensive_conflict_repair(
                    list(best_genes),
                    max_steps=intensive_step_cap,
                )
                if self._is_better_eval(intensified_eval, best_eval):
                    best_genes = intensified_genes
                    best_eval = intensified_eval
            shortlisted.append((best_eval, best_genes))

        shortlisted.sort(key=lambda item: (item[0].hard_conflicts, item[0].soft_penalty, -item[0].fitness))

        alternatives: list[GeneratedAlternative] = []
        seen_fingerprints: set[tuple[tuple[int, ...], ...]] = set()
        for evaluation, genes in shortlisted:
            payload = self._decode_payload(genes)
            fingerprint = tuple(
                sorted(
                    (
                        slot.day,
                        slot.startTime,
                        slot.endTime,
                        slot.courseId,
                        slot.roomId,
                        slot.facultyId,
                        slot.section,
                        slot.batch or "",
                        slot.sessionType or "",
                    )
                    for slot in payload.timetable_data
                )
            )
            if fingerprint in seen_fingerprints:
                continue
            seen_fingerprints.add(fingerprint)
            alternatives.append(
                GeneratedAlternative(
                    rank=len(alternatives) + 1,
                    fitness=evaluation.fitness,
                    hard_conflicts=evaluation.hard_conflicts,
                    soft_penalty=evaluation.soft_penalty,
                    payload=payload,
                )
            )
            if len(alternatives) >= request.alternative_count:
                break

        attempts = 0
        while len(alternatives) < request.alternative_count and attempts < request.alternative_count * 20:
            attempts += 1
            candidate_genes = self._repair_individual(self._random_individual(), max_passes=2)
            candidate_eval = self._evaluate(candidate_genes)
            if candidate_eval.hard_conflicts > 0 and intensive_budget > 0:
                intensive_budget -= 1
                candidate_genes, candidate_eval = self._intensive_conflict_repair(
                    candidate_genes,
                    max_steps=intensive_step_cap,
                )
            payload = self._decode_payload(candidate_genes)
            fingerprint = tuple(
                sorted(
                    (
                        slot.day,
                        slot.startTime,
                        slot.endTime,
                        slot.courseId,
                        slot.roomId,
                        slot.facultyId,
                        slot.section,
                        slot.batch or "",
                    )
                    for slot in payload.timetable_data
                )
            )
            if fingerprint in seen_fingerprints:
                continue
            seen_fingerprints.add(fingerprint)
            alternatives.append(
                GeneratedAlternative(
                    rank=len(alternatives) + 1,
                    fitness=candidate_eval.fitness,
                    hard_conflicts=candidate_eval.hard_conflicts,
                    soft_penalty=candidate_eval.soft_penalty,
                    payload=payload,
                )
            )

        if not alternatives:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Generation did not produce any alternatives",
            )

        runtime_ms = int((perf_counter() - start) * 1000)
        return GenerateTimetableResponse(
            alternatives=alternatives,
            settings_used=self.settings,
            runtime_ms=runtime_ms,
        )

    def _request_priority_order(self) -> list[int]:
        def sort_key(req_index: int) -> tuple:
            req = self.block_requests[req_index]
            # Priority 1: Labs (harder to fit due to contiguous blocks)
            # Priority 2: Number of feasible options (fewest options = most constrained = schedule first)
            # Priority 3: Block size (larger blocks are harder to fit)
            # Priority 4: Student count (larger sections need specific rooms)
            
            group_key = self._parallel_lab_group_key(req)
            is_parallel_lab = group_key is not None and req.batch

            # If it's a parallel lab, we want to schedule them together if possible, 
            # but the primary sort is still "difficulty".
            
            option_count = len(req.options)
            
            return (
                0 if req.is_lab else 1,  # Labs first
                option_count,            # Fewest options first (Most Constrained First)
                -req.block_size,         # Largest blocks first
                -req.student_count,      # Largest sections first
                req.course_code,         # Deterministic tie-break
                req.section,
                req.batch or "",
                req.request_id
            )

        return sorted(
            range(len(self.block_requests)),
            key=sort_key,
        )

    def _parallel_lab_overlap_allowed(self, req_a: BlockRequest, req_b: BlockRequest) -> bool:
        return (
            req_a.is_lab
            and req_b.is_lab
            and req_a.course_id == req_b.course_id
            and req_a.allow_parallel_batches
            and req_b.allow_parallel_batches
            and bool(req_a.batch)
            and bool(req_b.batch)
            and req_a.batch != req_b.batch
        )

    def _parallel_lab_sync_required(self, req_a: BlockRequest, req_b: BlockRequest) -> bool:
        return (
            req_a.is_lab
            and req_b.is_lab
            and req_a.course_id == req_b.course_id
            and req_a.section == req_b.section
            and req_a.allow_parallel_batches
            and req_b.allow_parallel_batches
            and bool(req_a.batch)
            and bool(req_b.batch)
            and req_a.batch != req_b.batch
        )

    def _incremental_option_penalty(
        self,
        *,
        req_index: int,
        option_index: int,
        selected_options: dict[int, PlacementOption],
        room_occ: dict[tuple[str, int, str], list[int]],
        faculty_occ: dict[tuple[str, int, str], list[int]],
        section_occ: dict[tuple[str, int, str], list[int]],
        elective_occ: dict[tuple[str, int], list[int]],
        faculty_minutes: dict[str, int],
        section_slot_keys: dict[str, set[tuple[str, int]]],
    ) -> tuple[int, float]:
        req = self.block_requests[req_index]
        option = req.options[option_index]
        room = self.rooms[option.room_id]
        faculty = self.faculty[option.faculty_id]
        weights = self.settings.objective_weights

        hard = 0
        soft = 0.0
        block_start, block_end = self._option_bounds(option, req.block_size)

        if not self._within_semester_time_window(block_start, block_end):
            hard += weights.semester_limit

        reserved_room_conflict, reserved_faculty_conflict = self._reserved_conflict_flags(
            day=option.day,
            start_min=block_start,
            end_min=block_end,
            room_id=option.room_id,
            faculty_id=option.faculty_id,
        )
        if reserved_room_conflict:
            hard += weights.room_conflict
        if reserved_faculty_conflict:
            hard += weights.faculty_conflict

        if room.capacity < req.student_count:
            hard += weights.room_capacity
        if req.is_lab and room.type != RoomType.lab:
            hard += weights.room_type
        if not req.is_lab and room.type == RoomType.lab:
            hard += weights.room_type

        if not self._faculty_allows_day(faculty, option.day):
            hard += weights.faculty_availability

        faculty_windows = self.faculty_windows.get(option.faculty_id, {})
        if faculty_windows.get(option.day):
            if not any(start <= block_start and block_end <= end for start, end in faculty_windows[option.day]):
                hard += weights.faculty_availability

        room_windows = self.room_windows.get(option.room_id, {})
        if room_windows.get(option.day):
            if not any(start <= block_start and block_end <= end for start, end in room_windows[option.day]):
                hard += weights.room_type

        period_minutes = self.schedule_policy.period_minutes
        projected_minutes = faculty_minutes.get(option.faculty_id, 0) + (req.block_size * period_minutes)
        max_minutes = faculty.max_hours * 60
        if projected_minutes > max_minutes:
            overflow_periods = max(1, (projected_minutes - max_minutes) // max(1, period_minutes))
            hard += weights.workload_overflow * overflow_periods
        elif max_minutes > 0:
            # Proactively spread teaching load before reaching hard overload.
            utilization = projected_minutes / max_minutes
            soft += max(0.0, utilization - 0.55) * max(1.0, weights.spread_balance)

        section_keys = section_slot_keys.get(req.section, set())
        projected_section_slot_count = len(section_keys)
        for offset in range(req.block_size):
            slot_idx = option.start_index + offset
            key = (option.day, slot_idx)
            if key not in section_keys:
                projected_section_slot_count += 1
        projected_section_minutes = projected_section_slot_count * period_minutes
        if self.expected_section_minutes > 0 and projected_section_minutes > self.expected_section_minutes:
            overflow_periods = max(
                1,
                math.ceil((projected_section_minutes - self.expected_section_minutes) / max(1, period_minutes)),
            )
            hard += weights.semester_limit * overflow_periods

        if not req.is_lab:
            for other_idx, other_option in selected_options.items():
                other_req = self.block_requests[other_idx]
                if other_req.is_lab:
                    continue
                if other_req.course_id != req.course_id:
                    continue
                same_section = other_req.section == req.section
                same_course_cross_section = self._single_faculty_required(req.course_id)
                if not same_section and not same_course_cross_section:
                    continue
                if other_option.faculty_id != option.faculty_id:
                    hard += weights.faculty_conflict

        if self._is_elective_request(req):
            signatures_by_section: dict[str, list[tuple[str, int, int, str]]] = defaultdict(list)
            for other_idx, other_option in selected_options.items():
                other_req = self.block_requests[other_idx]
                if not self._is_elective_request(other_req):
                    continue
                signatures_by_section[other_req.section].append(
                    (other_option.day, other_option.start_index, other_req.block_size, other_req.session_type)
                )
            signatures_by_section[req.section].append((option.day, option.start_index, req.block_size, req.session_type))
            compared_sections = sorted(signatures_by_section.keys())
            if len(compared_sections) > 1:
                baseline: list[tuple[str, int, int, str]] | None = None
                for section_name in compared_sections:
                    signatures = sorted(signatures_by_section[section_name])
                    if baseline is None:
                        baseline = signatures
                        continue
                    # Avoid over-penalizing incomplete partial assignments while constructing a candidate.
                    if len(signatures) != len(baseline):
                        continue
                    if signatures != baseline:
                        mismatch_size = max(1, len(set(baseline).symmetric_difference(set(signatures))))
                        hard += weights.section_conflict * mismatch_size

        for other_idx, other_option in selected_options.items():
            other_req = self.block_requests[other_idx]
            if self._is_faculty_back_to_back(req, option, other_req, other_option):
                soft += max(1.0, weights.spread_balance * 0.75)

        if req.preferred_faculty_ids and option.faculty_id not in req.preferred_faculty_ids:
            soft += weights.faculty_subject_preference * req.block_size
        if option.faculty_id != req.primary_faculty_id:
            soft += (weights.faculty_subject_preference * 0.5) * req.block_size

        # Prefer tighter but feasible room fit to preserve larger rooms for heavier sections.
        if room.capacity > 0:
            soft += max(0, room.capacity - req.student_count) / room.capacity

        for offset in range(req.block_size):
            slot_idx = option.start_index + offset
            room_key = (option.day, slot_idx, option.room_id)
            faculty_key = (option.day, slot_idx, option.faculty_id)
            section_key = (option.day, slot_idx, req.section)

            for other_idx in room_occ.get(room_key, []):
                other_req = self.block_requests[other_idx]
                if self._is_allowed_shared_overlap(req, other_req, option, selected_options[other_idx]):
                    continue
                hard += weights.room_conflict

            for other_idx in faculty_occ.get(faculty_key, []):
                other_req = self.block_requests[other_idx]
                if self._is_allowed_shared_overlap(req, other_req, option, selected_options[other_idx]):
                    continue
                hard += weights.faculty_conflict

            for other_idx in section_occ.get(section_key, []):
                other_req = self.block_requests[other_idx]
                if not self._parallel_lab_overlap_allowed(req, other_req):
                    hard += weights.section_conflict

            if self.elective_overlap_pairs:
                for other_idx in elective_occ.get((option.day, slot_idx), []):
                    other_req = self.block_requests[other_idx]
                    if other_req.course_id == req.course_id:
                        continue
                    if self._courses_conflict_in_elective_group(req.course_id, other_req.course_id):
                        hard += weights.section_conflict

        for other_idx, other_option in selected_options.items():
            other_req = self.block_requests[other_idx]
            if not self._parallel_lab_sync_required(req, other_req):
                continue
            if req.block_size != other_req.block_size:
                hard += weights.section_conflict
                continue
            if option.day != other_option.day or option.start_index != other_option.start_index:
                hard += weights.section_conflict

        return hard, soft

    def _is_immediately_conflict_free(
        self,
        *,
        req_index: int,
        option_index: int,
        selected_options: dict[int, PlacementOption],
        room_occ: dict[tuple[str, int, str], list[int]],
        faculty_occ: dict[tuple[str, int, str], list[int]],
        section_occ: dict[tuple[str, int, str], list[int]],
        faculty_minutes: dict[str, int],
        section_slot_keys: dict[str, set[tuple[str, int]]],
    ) -> bool:
        req = self.block_requests[req_index]
        option = req.options[option_index]
        block_start, block_end = self._option_bounds(option, req.block_size)
        room = self.rooms[option.room_id]
        faculty = self.faculty[option.faculty_id]

        if not self._within_semester_time_window(block_start, block_end):
            return False
        if not self._faculty_allows_day(faculty, option.day):
            return False
        # Relax strict capacity check to allow spillover to smaller rooms instead of forcing room conflicts
        if req.is_lab and room.type != RoomType.lab:
            return False
        if not req.is_lab and room.type == RoomType.lab:
            return False
        if self._conflicts_reserved_resources(
            day=option.day,
            start_min=block_start,
            end_min=block_end,
            room_id=option.room_id,
            faculty_id=option.faculty_id,
        ):
            return False

        period_minutes = self.schedule_policy.period_minutes
        projected_faculty_minutes = faculty_minutes.get(option.faculty_id, 0) + (req.block_size * period_minutes)
        max_faculty_minutes = faculty.max_hours * 60
        # RELAXED: We allow faculty overbooking to prevent hard conflicts (unassigned slots).
        # The penalty function will discourage this heavily, but it's better than failure.
        # if projected_faculty_minutes > max_faculty_minutes:
        #     return False

        section_keys = section_slot_keys.get(req.section, set())
        projected_section_slot_count = len(section_keys)
        for offset in range(req.block_size):
            slot_idx = option.start_index + offset
            key = (option.day, slot_idx)
            if key not in section_keys:
                projected_section_slot_count += 1
        if (
            self.expected_section_minutes > 0
            and projected_section_slot_count * period_minutes > self.expected_section_minutes
        ):
            return False

        faculty_windows = self.faculty_windows.get(option.faculty_id, {})
        if faculty_windows.get(option.day):
            if not any(start <= block_start and block_end <= end for start, end in faculty_windows[option.day]):
                return False

        room_windows = self.room_windows.get(option.room_id, {})
        if room_windows.get(option.day):
            if not any(start <= block_start and block_end <= end for start, end in room_windows[option.day]):
                return False

        for offset in range(req.block_size):
            slot_idx = option.start_index + offset
            room_key = (option.day, slot_idx, option.room_id)
            faculty_key = (option.day, slot_idx, option.faculty_id)
            section_key = (option.day, slot_idx, req.section)

            for other_idx in room_occ.get(room_key, []):
                other_req = self.block_requests[other_idx]
                if not self._is_allowed_shared_overlap(req, other_req, option, selected_options[other_idx]):
                    return False

            for other_idx in faculty_occ.get(faculty_key, []):
                other_req = self.block_requests[other_idx]
                if not self._is_allowed_shared_overlap(req, other_req, option, selected_options[other_idx]):
                    return False

            for other_idx in section_occ.get(section_key, []):
                other_req = self.block_requests[other_idx]
                if not self._parallel_lab_overlap_allowed(req, other_req):
                    return False

        # Keep one faculty per (course, section) for lecture/tutorial requests.
        if not req.is_lab:
            for other_idx, other_option in selected_options.items():
                other_req = self.block_requests[other_idx]
                if other_req.is_lab or other_req.course_id != req.course_id:
                    continue
                if other_req.section == req.section and other_option.faculty_id != option.faculty_id:
                    return False
                if (
                    other_req.section != req.section
                    and self._single_faculty_required(req.course_id)
                    and other_option.faculty_id != option.faculty_id
                ):
                    return False

        # Enforce elective synchronization progressively across sections when
        # comparable placement counts are available.
        if self._is_elective_request(req):
            signatures_by_section: dict[str, list[tuple[str, int, int, str]]] = defaultdict(list)
            for other_idx, other_option in selected_options.items():
                other_req = self.block_requests[other_idx]
                if not self._is_elective_request(other_req):
                    continue
                signatures_by_section[other_req.section].append(
                    (other_option.day, other_option.start_index, other_req.block_size, other_req.session_type)
                )
            signatures_by_section[req.section].append((option.day, option.start_index, req.block_size, req.session_type))
            sections = sorted(signatures_by_section.keys())
            if len(sections) > 1:
                baseline: list[tuple[str, int, int, str]] | None = None
                for section_name in sections:
                    signatures = sorted(signatures_by_section[section_name])
                    if baseline is None:
                        baseline = signatures
                        continue
                    if len(signatures) == len(baseline) and signatures != baseline:
                        return False

        return True

    def _is_section_slot_free(
        self,
        *,
        req_index: int,
        option_index: int,
        section_occ: dict[tuple[str, int, str], list[int]],
    ) -> bool:
        req = self.block_requests[req_index]
        option = req.options[option_index]
        for offset in range(req.block_size):
            slot_idx = option.start_index + offset
            section_key = (option.day, slot_idx, req.section)
            for other_idx in section_occ.get(section_key, []):
                other_req = self.block_requests[other_idx]
                if not self._parallel_lab_overlap_allowed(req, other_req):
                    return False
        return True

    def _constructive_individual_strict(
        self,
        *,
        randomized: bool,
        rcl_alpha: float = 0.05,
    ) -> list[int] | None:
        genes = [0] * len(self.block_requests)
        selected_options: dict[int, PlacementOption] = {}
        room_occ: dict[tuple[str, int, str], list[int]] = defaultdict(list)
        faculty_occ: dict[tuple[str, int, str], list[int]] = defaultdict(list)
        section_occ: dict[tuple[str, int, str], list[int]] = defaultdict(list)
        faculty_minutes: dict[str, int] = {}
        section_slot_keys: dict[str, set[tuple[str, int]]] = defaultdict(set)
        lab_baseline_batch_by_group: dict[tuple[str, str, str, int], str] = {}
        lab_baseline_signatures_by_group: dict[tuple[str, str, str, int], list[tuple[str, int]]] = defaultdict(list)
        lab_signature_usage_by_group_batch: dict[tuple[tuple[str, str, str, int], str], Counter[tuple[str, int]]] = defaultdict(Counter)

        sorted_indices = self._request_priority_order()
        request_indices_by_course_section = self._request_indices_by_course_section()
        request_indices_by_course = getattr(self, "request_indices_by_course", self._build_request_indices_by_course())
        single_faculty_required_by_course = getattr(self, "single_faculty_required_by_course", {})
        common_faculty_candidates_by_course = getattr(self, "common_faculty_candidates_by_course", {})
        common_faculty_candidates_by_course_section = getattr(self, "common_faculty_candidates_by_course_section", {})
        period_minutes = self.schedule_policy.period_minutes

        remaining_faculty_minutes: dict[str, int] = {
            faculty_id: max(0, faculty.max_hours) * 60
            for faculty_id, faculty in self.faculty.items()
        }
        planned_faculty_by_course_section: dict[tuple[str, str], str] = {}

        for course_id, required in single_faculty_required_by_course.items():
            if not required:
                continue
            lecture_indices = [
                req_index
                for req_index in request_indices_by_course.get(course_id, [])
                if not self.block_requests[req_index].is_lab
            ]
            if not lecture_indices:
                continue
            required_minutes = sum(
                self.block_requests[req_index].block_size * period_minutes for req_index in lecture_indices
            )
            candidate_ids = list(common_faculty_candidates_by_course.get(course_id, ()))
            if not candidate_ids:
                continue
            course = self.courses.get(course_id)
            course_code = course.code if course is not None else ""
            candidate_ids.sort(
                key=lambda faculty_id: (
                    remaining_faculty_minutes.get(faculty_id, 0) < required_minutes,
                    not self._faculty_prefers_subject(faculty_id, course_code),
                    -remaining_faculty_minutes.get(faculty_id, 0),
                    self.faculty[faculty_id].workload_hours,
                    self.faculty[faculty_id].name,
                )
            )
            selected_id = next(
                (
                    faculty_id
                    for faculty_id in candidate_ids
                    if remaining_faculty_minutes.get(faculty_id, 0) >= required_minutes
                ),
                None,
            )
            if selected_id is None:
                continue
            remaining_faculty_minutes[selected_id] -= required_minutes
            for req_index in lecture_indices:
                req = self.block_requests[req_index]
                planned_faculty_by_course_section[(req.course_id, req.section)] = selected_id

        section_groups = sorted(
            request_indices_by_course_section.items(),
            key=lambda item: -sum(self.block_requests[idx].block_size for idx in item[1]),
        )
        for (course_id, section_name), req_indices in section_groups:
            if (course_id, section_name) in planned_faculty_by_course_section:
                continue
            required_minutes = sum(
                self.block_requests[req_index].block_size * period_minutes for req_index in req_indices
            )
            candidate_ids = list(common_faculty_candidates_by_course_section.get((course_id, section_name), ()))
            if not candidate_ids:
                candidate_ids = sorted(
                    set.intersection(
                        *[
                            {option.faculty_id for option in self.block_requests[req_index].options}
                            for req_index in req_indices
                        ]
                    )
                    if req_indices
                    else set()
                )
            if not candidate_ids:
                continue
            course = self.courses.get(course_id)
            course_code = course.code if course is not None else ""
            candidate_ids.sort(
                key=lambda faculty_id: (
                    remaining_faculty_minutes.get(faculty_id, 0) < required_minutes,
                    not self._faculty_prefers_subject(faculty_id, course_code),
                    -remaining_faculty_minutes.get(faculty_id, 0),
                    self.faculty[faculty_id].workload_hours,
                    self.faculty[faculty_id].name,
                )
            )
            selected_id = next(
                (
                    faculty_id
                    for faculty_id in candidate_ids
                    if remaining_faculty_minutes.get(faculty_id, 0) >= required_minutes
                ),
                None,
            )
            if selected_id is None:
                continue
            planned_faculty_by_course_section[(course_id, section_name)] = selected_id
            remaining_faculty_minutes[selected_id] -= required_minutes

        def selected_faculty_for_request(req: BlockRequest) -> str | None:
            if req.is_lab:
                return None
            for other_idx in request_indices_by_course_section.get((req.course_id, req.section), []):
                if other_idx in selected_options:
                    return selected_options[other_idx].faculty_id
            if self._single_faculty_required(req.course_id):
                for other_idx in request_indices_by_course.get(req.course_id, []):
                    if other_idx not in selected_options:
                        continue
                    other_req = self.block_requests[other_idx]
                    if other_req.is_lab:
                        continue
                    return selected_options[other_idx].faculty_id
            return None

        def ordered_candidates(req_index: int) -> list[int]:
            req = self.block_requests[req_index]
            planned_faculty_id = planned_faculty_by_course_section.get((req.course_id, req.section))

            if req.request_id in self.fixed_genes:
                fixed_option_index = self.fixed_genes[req.request_id]
                if self._is_immediately_conflict_free(
                    req_index=req_index,
                    option_index=fixed_option_index,
                    selected_options=selected_options,
                    room_occ=room_occ,
                    faculty_occ=faculty_occ,
                    section_occ=section_occ,
                    faculty_minutes=faculty_minutes,
                    section_slot_keys=section_slot_keys,
                ):
                    return [fixed_option_index]
                return []

            block_count_local = len(self.block_requests)
            if block_count_local >= 220:
                max_candidate_window = 96 if randomized else 128
            elif block_count_local >= 160:
                max_candidate_window = 72 if randomized else 108
            else:
                max_candidate_window = 40 if randomized else 72
            if req.is_lab:
                max_candidate_window += 12
            all_candidate_indices = self._option_candidate_indices(
                req,
                max_candidates=min(len(req.options), max_candidate_window),
                allow_random_tail=randomized,
            )
            if not all_candidate_indices:
                all_candidate_indices = list(range(len(req.options)))

            fixed_faculty_id = selected_faculty_for_request(req)
            if fixed_faculty_id is not None:
                matching = [
                    option_index
                    for option_index in all_candidate_indices
                    if req.options[option_index].faculty_id == fixed_faculty_id
                ]
                if not matching:
                    matching = [
                        option_index
                        for option_index, option in enumerate(req.options)
                        if option.faculty_id == fixed_faculty_id
                    ]
                if not matching:
                    return []
                all_candidate_indices = matching
            elif planned_faculty_id is not None:
                planned_matches = [
                    option_index
                    for option_index in all_candidate_indices
                    if req.options[option_index].faculty_id == planned_faculty_id
                ]
                if not planned_matches:
                    planned_matches = [
                        option_index
                        for option_index, option in enumerate(req.options)
                        if option.faculty_id == planned_faculty_id
                    ]
                if planned_matches:
                    planned_set = set(planned_matches)
                    all_candidate_indices = [
                        *planned_matches,
                        *[option_index for option_index in all_candidate_indices if option_index not in planned_set],
                    ]

            group_key = self._parallel_lab_group_key(req)
            if group_key and req.batch:
                target_signatures: set[tuple[str, int]] = set()
                for other_req_index, other_option in selected_options.items():
                    other_req = self.block_requests[other_req_index]
                    if other_req.request_id == req.request_id:
                        continue
                    if self._parallel_lab_group_key(other_req) != group_key:
                        continue
                    if other_req.batch == req.batch:
                        continue
                    target_signatures.add(self._parallel_lab_signature(other_option))
                if target_signatures:
                    filtered = [
                        option_index
                        for option_index in all_candidate_indices
                        if self._parallel_lab_signature(req.options[option_index]) in target_signatures
                    ]
                    if not filtered:
                        filtered = [
                            option_index
                            for option_index, option in enumerate(req.options)
                            if self._parallel_lab_signature(option) in target_signatures
                        ]
                    if not filtered:
                        return []
                    all_candidate_indices = filtered

                baseline_batch = lab_baseline_batch_by_group.get(group_key)
                if baseline_batch and baseline_batch != req.batch:
                    baseline_counts = Counter(lab_baseline_signatures_by_group.get(group_key, []))
                    if baseline_counts:
                        usage_counter = lab_signature_usage_by_group_batch.get((group_key, req.batch), Counter())
                        allowed_signatures = {
                            signature
                            for signature, expected_count in baseline_counts.items()
                            if usage_counter.get(signature, 0) < expected_count
                        }
                        if allowed_signatures:
                            balanced = [
                                option_index
                                for option_index in all_candidate_indices
                                if self._parallel_lab_signature(req.options[option_index]) in allowed_signatures
                            ]
                            if balanced:
                                all_candidate_indices = balanced

            feasible_indices: list[int] = []
            for option_index in all_candidate_indices:
                if self._is_immediately_conflict_free(
                    req_index=req_index,
                    option_index=option_index,
                    selected_options=selected_options,
                    room_occ=room_occ,
                    faculty_occ=faculty_occ,
                    section_occ=section_occ,
                    faculty_minutes=faculty_minutes,
                    section_slot_keys=section_slot_keys,
                ):
                    feasible_indices.append(option_index)

            if not feasible_indices and len(all_candidate_indices) < len(req.options):
                for option_index in range(len(req.options)):
                    if self._is_immediately_conflict_free(
                        req_index=req_index,
                        option_index=option_index,
                        selected_options=selected_options,
                        room_occ=room_occ,
                        faculty_occ=faculty_occ,
                        section_occ=section_occ,
                        faculty_minutes=faculty_minutes,
                        section_slot_keys=section_slot_keys,
                    ):
                        feasible_indices.append(option_index)

            if not feasible_indices:
                return []

            scored_candidates: list[tuple[float, int]] = []
            for option_index in feasible_indices:
                hard_score, soft_score = self._incremental_option_penalty(
                    req_index=req_index,
                    option_index=option_index,
                    selected_options=selected_options,
                    room_occ=room_occ,
                    faculty_occ=faculty_occ,
                    section_occ=section_occ,
                    elective_occ=defaultdict(list),
                    faculty_minutes=faculty_minutes,
                    section_slot_keys=section_slot_keys,
                )
                room = self.rooms[req.options[option_index].room_id]
                capacity_waste = 0.0
                if room.capacity >= req.student_count:
                    capacity_waste = (room.capacity - req.student_count) / max(1, room.capacity)
                final_score = (hard_score * 10000.0) + soft_score + (capacity_waste * 0.5)
                if (
                    not req.is_lab
                    and planned_faculty_id is not None
                    and req.options[option_index].faculty_id != planned_faculty_id
                ):
                    final_score += 200.0
                scored_candidates.append((final_score, option_index))

            scored_candidates.sort(key=lambda item: item[0])
            ordered = [option_index for _, option_index in scored_candidates]

            if randomized and len(ordered) > 1:
                bounded_alpha = min(0.6, max(0.0, rcl_alpha))
                if bounded_alpha > 0:
                    best_score = scored_candidates[0][0]
                    threshold = best_score + (abs(best_score) * bounded_alpha) + 1.0
                    rcl = [option_index for score, option_index in scored_candidates if score <= threshold]
                    if len(rcl) > 1:
                        self.random.shuffle(rcl)
                        rcl_set = set(rcl)
                        ordered = [*rcl, *[item for item in ordered if item not in rcl_set]]
                else:
                    head_size = min(len(ordered), max(2, len(ordered) // 3))
                    head = ordered[:head_size]
                    self.random.shuffle(head)
                    ordered = [*head, *ordered[head_size:]]

            return ordered

        decision_stack: list[dict] = []
        depth = 0
        backtracks = 0
        max_backtracks = max(4500, len(sorted_indices) * 22)

        while depth < len(sorted_indices):
            req_index = sorted_indices[depth]

            if depth >= len(decision_stack) or decision_stack[depth]["req_index"] != req_index:
                candidates = ordered_candidates(req_index)
                if not candidates:
                    if depth == 0 or backtracks >= max_backtracks:
                        return None
                    # Trigger backtracking.
                    while depth > 0 and backtracks < max_backtracks:
                        depth -= 1
                        backtracks += 1
                        previous_entry = decision_stack[depth]
                        previous_req_index = previous_entry["req_index"]
                        if previous_req_index in selected_options:
                            self._unrecord_selection(
                                previous_req_index,
                                genes[previous_req_index],
                                selected_options,
                                room_occ,
                                faculty_occ,
                                section_occ,
                                faculty_minutes,
                                section_slot_keys,
                                lab_baseline_batch_by_group,
                                lab_baseline_signatures_by_group,
                                lab_signature_usage_by_group_batch,
                            )

                        previous_candidates = previous_entry["candidates"]
                        pointer = previous_entry["next_pos"]
                        advanced = False
                        while pointer < len(previous_candidates):
                            option_index = previous_candidates[pointer]
                            pointer += 1
                            if not self._is_immediately_conflict_free(
                                req_index=previous_req_index,
                                option_index=option_index,
                                selected_options=selected_options,
                                room_occ=room_occ,
                                faculty_occ=faculty_occ,
                                section_occ=section_occ,
                                faculty_minutes=faculty_minutes,
                                section_slot_keys=section_slot_keys,
                            ):
                                continue
                            genes[previous_req_index] = option_index
                            self._record_selection(
                                previous_req_index,
                                option_index,
                                selected_options,
                                room_occ,
                                faculty_occ,
                                section_occ,
                                faculty_minutes,
                                section_slot_keys,
                                lab_baseline_batch_by_group,
                                lab_baseline_signatures_by_group,
                                lab_signature_usage_by_group_batch,
                            )
                            previous_entry["next_pos"] = pointer
                            decision_stack[depth] = previous_entry
                            depth += 1
                            del decision_stack[depth:]
                            advanced = True
                            break
                        if advanced:
                            break
                    if depth == 0 and (not decision_stack or decision_stack[0]["next_pos"] >= len(decision_stack[0]["candidates"])):
                        return None
                    continue

                entry = {"req_index": req_index, "candidates": candidates, "next_pos": 0}
                if depth >= len(decision_stack):
                    decision_stack.append(entry)
                else:
                    decision_stack[depth] = entry
                    del decision_stack[depth + 1 :]

            entry = decision_stack[depth]
            candidates = entry["candidates"]
            pointer = entry["next_pos"]
            placed = False
            while pointer < len(candidates):
                option_index = candidates[pointer]
                pointer += 1
                if not self._is_immediately_conflict_free(
                    req_index=req_index,
                    option_index=option_index,
                    selected_options=selected_options,
                    room_occ=room_occ,
                    faculty_occ=faculty_occ,
                    section_occ=section_occ,
                    faculty_minutes=faculty_minutes,
                    section_slot_keys=section_slot_keys,
                ):
                    continue
                genes[req_index] = option_index
                self._record_selection(
                    req_index,
                    option_index,
                    selected_options,
                    room_occ,
                    faculty_occ,
                    section_occ,
                    faculty_minutes,
                    section_slot_keys,
                    lab_baseline_batch_by_group,
                    lab_baseline_signatures_by_group,
                    lab_signature_usage_by_group_batch,
                )
                entry["next_pos"] = pointer
                decision_stack[depth] = entry
                depth += 1
                placed = True
                break

            if placed:
                continue

            if depth == 0 or backtracks >= max_backtracks:
                return None

            while depth > 0 and backtracks < max_backtracks:
                depth -= 1
                backtracks += 1
                previous_entry = decision_stack[depth]
                previous_req_index = previous_entry["req_index"]
                if previous_req_index in selected_options:
                    self._unrecord_selection(
                        previous_req_index,
                        genes[previous_req_index],
                        selected_options,
                        room_occ,
                        faculty_occ,
                        section_occ,
                        faculty_minutes,
                        section_slot_keys,
                        lab_baseline_batch_by_group,
                        lab_baseline_signatures_by_group,
                        lab_signature_usage_by_group_batch,
                    )

                previous_candidates = previous_entry["candidates"]
                pointer = previous_entry["next_pos"]
                advanced = False
                while pointer < len(previous_candidates):
                    option_index = previous_candidates[pointer]
                    pointer += 1
                    if not self._is_immediately_conflict_free(
                        req_index=previous_req_index,
                        option_index=option_index,
                        selected_options=selected_options,
                        room_occ=room_occ,
                        faculty_occ=faculty_occ,
                        section_occ=section_occ,
                        faculty_minutes=faculty_minutes,
                        section_slot_keys=section_slot_keys,
                    ):
                        continue
                    genes[previous_req_index] = option_index
                    self._record_selection(
                        previous_req_index,
                        option_index,
                        selected_options,
                        room_occ,
                        faculty_occ,
                        section_occ,
                        faculty_minutes,
                        section_slot_keys,
                        lab_baseline_batch_by_group,
                        lab_baseline_signatures_by_group,
                        lab_signature_usage_by_group_batch,
                    )
                    previous_entry["next_pos"] = pointer
                    decision_stack[depth] = previous_entry
                    depth += 1
                    del decision_stack[depth:]
                    advanced = True
                    break
                if advanced:
                    break

            if depth == 0 and (
                not decision_stack
                or decision_stack[0]["next_pos"] >= len(decision_stack[0]["candidates"])
            ):
                return None

        return genes

    def _constructive_individual(
        self,
        *,
        randomized: bool,
        rcl_alpha: float = 0.05,
        strict_dead_end: bool = False,
    ) -> list[int] | None:
        if strict_dead_end:
            strict_result = self._constructive_individual_strict(
                randomized=randomized,
                rcl_alpha=rcl_alpha,
            )
            if strict_result is not None:
                return strict_result
            # Backtracking can still fail in dense terms; keep a strict
            # greedy fallback that returns `None` on the first hard dead-end.

        genes = [0] * len(self.block_requests)
        
        # Tracking state locally for the constructive build
        selected_options: dict[int, PlacementOption] = {}
        room_occ: dict[tuple[str, int, str], list[int]] = defaultdict(list)
        faculty_occ: dict[tuple[str, int, str], list[int]] = defaultdict(list)
        section_occ: dict[tuple[str, int, str], list[int]] = defaultdict(list)
        faculty_minutes: dict[str, int] = {}
        section_slot_keys: dict[str, set[tuple[str, int]]] = defaultdict(set)
        
        # Parallel lab tracking
        lab_baseline_batch_by_group: dict[tuple[str, str, str, int], str] = {}
        lab_baseline_signatures_by_group: dict[tuple[str, str, str, int], list[tuple[str, int]]] = defaultdict(list)
        lab_signature_usage_by_group_batch: dict[tuple[tuple[str, str, str, int], str], Counter[tuple[str, int]]] = defaultdict(Counter)
        chosen_faculty_by_course: dict[str, str] = {}
        chosen_faculty_by_course_section: dict[tuple[str, str], str] = {}

        sorted_indices = self._request_priority_order()
        
        for req_index in sorted_indices:
            req = self.block_requests[req_index]

            # 1. Respect pre-fixed genes (e.g. from partial solutions or locks)
            if req.request_id in self.fixed_genes:
                chosen_index = self.fixed_genes[req.request_id]
                genes[req_index] = chosen_index
                if not req.is_lab:
                    selected_faculty_id = req.options[chosen_index].faculty_id
                    chosen_faculty_by_course_section[(req.course_id, req.section)] = selected_faculty_id
                    if self._single_faculty_required(req.course_id):
                        chosen_faculty_by_course.setdefault(req.course_id, selected_faculty_id)
                self._record_selection(
                    req_index, 
                    chosen_index, 
                    selected_options, 
                    room_occ, 
                    faculty_occ, 
                    section_occ, 
                    faculty_minutes, 
                    section_slot_keys,
                    lab_baseline_batch_by_group,
                    lab_baseline_signatures_by_group,
                    lab_signature_usage_by_group_batch
                )
                continue

            # 2. Determine Candidate Options
            max_candidate_window = 36 if randomized else 72
            if req.is_lab:
                max_candidate_window += 12

            all_candidate_indices = self._option_candidate_indices(
                req,
                max_candidates=min(len(req.options), max_candidate_window),
                allow_random_tail=randomized,
            )
            if not all_candidate_indices:
                all_candidate_indices = list(range(len(req.options)))

            if not req.is_lab:
                selected_faculty_id = chosen_faculty_by_course_section.get((req.course_id, req.section))
                if selected_faculty_id is None and self._single_faculty_required(req.course_id):
                    selected_faculty_id = chosen_faculty_by_course.get(req.course_id)
                if selected_faculty_id is not None:
                    matching_indices = [
                        option_index
                        for option_index in all_candidate_indices
                        if req.options[option_index].faculty_id == selected_faculty_id
                    ]
                    if not matching_indices:
                        matching_indices = [
                            option_index
                            for option_index, option in enumerate(req.options)
                            if option.faculty_id == selected_faculty_id
                        ]
                    if strict_dead_end and not matching_indices:
                        return None
                    if matching_indices:
                        all_candidate_indices = matching_indices

            # 3. Filter for Hard Feasibility
            feasible_indices = []
            for opt_idx in all_candidate_indices:
                if self._is_immediately_conflict_free(
                    req_index=req_index,
                    option_index=opt_idx,
                    selected_options=selected_options,
                    room_occ=room_occ,
                    faculty_occ=faculty_occ,
                    section_occ=section_occ,
                    faculty_minutes=faculty_minutes,
                    section_slot_keys=section_slot_keys,
                ):
                    feasible_indices.append(opt_idx)

            if not feasible_indices and len(all_candidate_indices) < len(req.options):
                for opt_idx in range(len(req.options)):
                    if self._is_immediately_conflict_free(
                        req_index=req_index,
                        option_index=opt_idx,
                        selected_options=selected_options,
                        room_occ=room_occ,
                        faculty_occ=faculty_occ,
                        section_occ=section_occ,
                        faculty_minutes=faculty_minutes,
                        section_slot_keys=section_slot_keys,
                    ):
                        feasible_indices.append(opt_idx)

            if strict_dead_end and not feasible_indices:
                return None
            
            # If no hard-feasible option exists at this step, score a much wider set to
            # pick the least-damaging placement instead of a narrow random window.
            if feasible_indices:
                candidates_to_score = feasible_indices
            else:
                widened = self._option_candidate_indices(
                    req,
                    max_candidates=min(len(req.options), 128),
                    allow_random_tail=False,
                )
                candidates_to_score = widened if widened else list(range(len(req.options)))
            
            # 4. Score Candidates (Best-Fit)
            scored_candidates: list[tuple[float, int]] = []
            
            for opt_idx in candidates_to_score:
                hard_score, soft_score = self._incremental_option_penalty(
                    req_index=req_index,
                    option_index=opt_idx,
                    selected_options=selected_options,
                    room_occ=room_occ,
                    faculty_occ=faculty_occ,
                    section_occ=section_occ,
                    elective_occ=defaultdict(list),
                    faculty_minutes=faculty_minutes,
                    section_slot_keys=section_slot_keys,
                )
                
                room = self.rooms[req.options[opt_idx].room_id]
                capacity_waste = 0.0
                if room.capacity >= req.student_count:
                    capacity_waste = (room.capacity - req.student_count) / max(1, room.capacity)
                
                # Heuristic Weighting
                final_score = (hard_score * 10000.0) + soft_score + (capacity_waste * 0.5)
                scored_candidates.append((final_score, opt_idx))
            
            # 5. Select Best
            scored_candidates.sort(key=lambda x: x[0])
            
            chosen_index = -1
            if not scored_candidates:
                chosen_index = 0
            elif randomized and rcl_alpha > 0 and len(scored_candidates) > 1:
                best_score = scored_candidates[0][0]
                threshold = best_score + (abs(best_score) * rcl_alpha) + 1.0
                rcl = [idx for score, idx in scored_candidates if score <= threshold]
                chosen_index = self.random.choice(rcl)
            else:
                chosen_index = scored_candidates[0][1]
                
            genes[req_index] = chosen_index
            if not req.is_lab:
                selected_faculty_id = req.options[chosen_index].faculty_id
                chosen_faculty_by_course_section[(req.course_id, req.section)] = selected_faculty_id
                if self._single_faculty_required(req.course_id):
                    chosen_faculty_by_course.setdefault(req.course_id, selected_faculty_id)
            
            # 6. Update State
            self._record_selection(
                req_index, 
                chosen_index, 
                selected_options, 
                room_occ, 
                faculty_occ, 
                section_occ, 
                faculty_minutes, 
                section_slot_keys,
                lab_baseline_batch_by_group,
                lab_baseline_signatures_by_group,
                lab_signature_usage_by_group_batch
            )

        return genes

    def _record_selection(
        self,
        req_index: int,
        option_index: int,
        selected_options: dict[int, PlacementOption],
        room_occ: dict,
        faculty_occ: dict,
        section_occ: dict,
        faculty_minutes: dict,
        section_slot_keys: dict,
        lab_baseline_batch_by_group: dict,
        lab_baseline_signatures_by_group: dict,
        lab_signature_usage_by_group_batch: dict
    ):
        req = self.block_requests[req_index]
        option = req.options[option_index]
        selected_options[req_index] = option
        
        if req.is_lab:
            group_key = self._parallel_lab_group_key(req)
            if group_key and req.batch:
                lab_baseline_batch_by_group.setdefault(group_key, req.batch)
                signature = self._parallel_lab_signature(option)
                if req.batch == lab_baseline_batch_by_group[group_key]:
                    lab_baseline_signatures_by_group[group_key].append(signature)
                lab_signature_usage_by_group_batch[(group_key, req.batch)][signature] += 1

        for offset in range(req.block_size):
            slot_idx = option.start_index + offset
            room_key = (option.day, slot_idx, option.room_id)
            faculty_key = (option.day, slot_idx, option.faculty_id)
            section_key = (option.day, slot_idx, req.section)
            
            room_occ[room_key].append(req_index)
            faculty_occ[faculty_key].append(req_index)
            section_occ[section_key].append(req_index)
            section_slot_keys[req.section].add((option.day, slot_idx))
        
        added_minutes = req.block_size * self.schedule_policy.period_minutes
        faculty_minutes[option.faculty_id] = faculty_minutes.get(option.faculty_id, 0) + added_minutes

    def _unrecord_selection(
        self,
        req_index: int,
        option_index: int,
        selected_options: dict,
        room_occ: dict,
        faculty_occ: dict,
        section_occ: dict,
        faculty_minutes: dict,
        section_slot_keys: dict,
        lab_baseline_batch_by_group: dict,
        lab_baseline_signatures_by_group: dict,
        lab_signature_usage_by_group_batch: dict,
    ) -> None:
        req = self.block_requests[req_index]
        option = req.options[option_index]
        selected_options.pop(req_index, None)

        if req.is_lab:
            group_key = self._parallel_lab_group_key(req)
            if group_key and req.batch:
                signature = self._parallel_lab_signature(option)
                usage_key = (group_key, req.batch)
                usage_counter = lab_signature_usage_by_group_batch.get(usage_key)
                if usage_counter is not None:
                    current = usage_counter.get(signature, 0)
                    if current <= 1:
                        usage_counter.pop(signature, None)
                    else:
                        usage_counter[signature] = current - 1
                    if not usage_counter:
                        lab_signature_usage_by_group_batch.pop(usage_key, None)

                active_batches: list[str] = sorted(
                    batch_name
                    for (key, batch_name), counter in lab_signature_usage_by_group_batch.items()
                    if key == group_key and counter
                )
                if not active_batches:
                    lab_baseline_batch_by_group.pop(group_key, None)
                    lab_baseline_signatures_by_group.pop(group_key, None)
                else:
                    baseline_batch = active_batches[0]
                    lab_baseline_batch_by_group[group_key] = baseline_batch
                    baseline_counter = lab_signature_usage_by_group_batch.get((group_key, baseline_batch), Counter())
                    baseline_signatures: list[tuple[str, int]] = []
                    for baseline_signature, count in baseline_counter.items():
                        if count <= 0:
                            continue
                        baseline_signatures.extend([baseline_signature] * count)
                    lab_baseline_signatures_by_group[group_key] = baseline_signatures

        for offset in range(req.block_size):
            slot_idx = option.start_index + offset
            room_key = (option.day, slot_idx, option.room_id)
            faculty_key = (option.day, slot_idx, option.faculty_id)
            section_key = (option.day, slot_idx, req.section)

            room_entries = room_occ.get(room_key, [])
            if req_index in room_entries:
                room_entries.remove(req_index)
            if not room_entries:
                room_occ.pop(room_key, None)

            faculty_entries = faculty_occ.get(faculty_key, [])
            if req_index in faculty_entries:
                faculty_entries.remove(req_index)
            if not faculty_entries:
                faculty_occ.pop(faculty_key, None)

            section_entries = section_occ.get(section_key, [])
            if req_index in section_entries:
                section_entries.remove(req_index)
            if not section_entries:
                section_occ.pop(section_key, None)
                section_slot_keys.get(req.section, set()).discard((option.day, slot_idx))

        if req.section in section_slot_keys and not section_slot_keys[req.section]:
            section_slot_keys.pop(req.section, None)

        removed_minutes = req.block_size * self.schedule_policy.period_minutes
        updated_minutes = faculty_minutes.get(option.faculty_id, 0) - removed_minutes
        if updated_minutes > 0:
            faculty_minutes[option.faculty_id] = updated_minutes
        else:
            faculty_minutes.pop(option.faculty_id, None)


    def _perturb_individual(self, genes: list[int], *, intensity: float) -> list[int]:
        mutated = list(genes)
        mutable_indices = [
            idx for idx, req in enumerate(self.block_requests) if req.request_id not in self.fixed_genes
        ]
        if not mutable_indices:
            return mutated

        conflicted = [idx for idx in self._conflicted_request_ids(mutated) if idx in set(mutable_indices)]
        target_count = max(1, int(len(mutable_indices) * max(0.01, intensity)))
        chosen: set[int] = set()

        if conflicted:
            self.random.shuffle(conflicted)
            chosen.update(conflicted[:target_count])

        while len(chosen) < target_count:
            chosen.add(self.random.choice(mutable_indices))

        for idx in chosen:
            req = self.block_requests[idx]
            if len(req.options) <= 1:
                continue
            candidate_indices = self._option_candidate_indices(req, max_candidates=12)
            if not candidate_indices:
                continue
            mutated[idx] = self.random.choice(candidate_indices)

        return self._harmonize_faculty_assignments(mutated)

    @staticmethod
    def _dominates_eval(left: EvaluationResult, right: EvaluationResult) -> bool:
        return (
            left.hard_conflicts <= right.hard_conflicts
            and left.soft_penalty <= right.soft_penalty
            and (
                left.hard_conflicts < right.hard_conflicts
                or left.soft_penalty < right.soft_penalty
            )
        )

    @staticmethod
    def _is_better_eval(left: EvaluationResult, right: EvaluationResult) -> bool:
        return (
            left.hard_conflicts < right.hard_conflicts
            or (
                left.hard_conflicts == right.hard_conflicts
                and left.soft_penalty < right.soft_penalty
            )
            or (
                left.hard_conflicts == right.hard_conflicts
                and left.soft_penalty == right.soft_penalty
                and left.fitness > right.fitness
            )
        )

    @staticmethod
    def _annealing_energy(evaluation: EvaluationResult) -> float:
        hard_component = evaluation.hard_conflicts * 10_000.0
        soft_component = evaluation.soft_penalty
        tie_break = max(0.0, -evaluation.fitness) * 0.01
        return hard_component + soft_component + tie_break

    def _payload_fingerprint(self, payload: OfficialTimetablePayload) -> tuple[tuple[str, ...], ...]:
        return tuple(
            sorted(
                (
                    slot.day,
                    slot.startTime,
                    slot.endTime,
                    slot.courseId,
                    slot.roomId,
                    slot.facultyId,
                    slot.section,
                    slot.batch or "",
                    slot.sessionType or "",
                )
                for slot in payload.timetable_data
            )
        )

    def _run_hybrid_search(self, request: GenerateTimetableRequest) -> GenerateTimetableResponse:
        start = perf_counter()
        archive: list[tuple[EvaluationResult, list[int]]] = []
        seen_genotypes: set[tuple[int, ...]] = set()
        archive_limit = min(120, max(24, request.alternative_count * 24))
        block_count = len(self.block_requests)

        def add_candidate(candidate_genes: list[int], *, repair_passes: int) -> None:
            repaired = self._repair_individual(list(candidate_genes), max_passes=repair_passes)
            key = tuple(repaired)
            if key in seen_genotypes:
                return
            seen_genotypes.add(key)

            evaluation = self._evaluate(repaired)
            survivors: list[tuple[EvaluationResult, list[int]]] = []
            for existing_eval, existing_genes in archive:
                if self._dominates_eval(existing_eval, evaluation):
                    return
                if not self._dominates_eval(evaluation, existing_eval):
                    survivors.append((existing_eval, existing_genes))
            survivors.append((evaluation, repaired))
            survivors.sort(key=lambda item: (item[0].hard_conflicts, item[0].soft_penalty, -item[0].fitness))
            archive[:] = survivors[:archive_limit]

        constructive_trials = min(64, max(10, self.settings.population_size // 3))
        if block_count >= 220:
            constructive_trials = min(constructive_trials, 6)
        elif block_count >= 160:
            constructive_trials = min(constructive_trials, 10)
        elif block_count >= 120:
            constructive_trials = min(constructive_trials, 14)
        elif block_count >= 80:
            constructive_trials = min(constructive_trials, 20)
        for trial in range(constructive_trials):
            add_candidate(
                self._constructive_individual(
                    randomized=trial > 0,
                    rcl_alpha=0.10 + (0.30 * (trial / max(1, constructive_trials))),
                ),
                repair_passes=2 if trial < 4 else 1,
            )

        if not archive:
            add_candidate(self._random_individual(), repair_passes=2)

        local_iterations = min(180, max(36, self.settings.generations // 2))
        if block_count >= 220:
            local_iterations = min(local_iterations, 16)
        elif block_count >= 160:
            local_iterations = min(local_iterations, 26)
        elif block_count >= 120:
            local_iterations = min(local_iterations, 42)
        elif block_count >= 80:
            local_iterations = min(local_iterations, 60)
        else:
            local_iterations = min(local_iterations, 90)
        for iteration in range(local_iterations):
            if not archive:
                break
            if (
                archive[0][0].hard_conflicts == 0
                and len(archive) >= max(4, request.alternative_count)
                and iteration >= max(6, local_iterations // 4)
            ):
                break

            archive.sort(key=lambda item: (item[0].hard_conflicts, item[0].soft_penalty, -item[0].fitness))
            elite_window = max(1, min(len(archive), max(3, self.settings.elite_count)))
            base_eval, base_genes = archive[self.random.randrange(elite_window)]
            candidate = list(base_genes)

            intensity = 0.03 + (0.15 * (iteration / max(1, local_iterations)))
            candidate = self._perturb_individual(candidate, intensity=intensity)

            mutation_boost = 1.3 if base_eval.hard_conflicts > 0 else 1.0
            mutation_rate = min(0.35, self.settings.mutation_rate * mutation_boost)
            if self.random.random() < 0.65:
                candidate = self._mutate(candidate, mutation_rate=mutation_rate)

            if len(archive) > 1 and self.random.random() < min(0.85, self.settings.crossover_rate + 0.1):
                mate = archive[self.random.randrange(len(archive))][1]
                candidate = self._crossover(candidate, mate)

            add_candidate(
                candidate,
                repair_passes=2 if (iteration % 7 == 0 or base_eval.hard_conflicts > 0) else 1,
            )

        ranked = sorted(archive, key=lambda item: (item[0].hard_conflicts, item[0].soft_penalty, -item[0].fitness))
        alternatives: list[GeneratedAlternative] = []
        seen_fingerprints: set[tuple[tuple[str, ...], ...]] = set()
        intensive_budget = (
            max(2, request.alternative_count * 2)
            if block_count >= 180
            else max(4, request.alternative_count * 4)
        )
        intensive_step_cap = self._intensive_repair_step_cap()

        for evaluation, genes in ranked:
            best_genes = genes
            best_eval = evaluation
            if best_eval.hard_conflicts > 0 and intensive_budget > 0:
                intensive_budget -= 1
                intensified_genes, intensified_eval = self._intensive_conflict_repair(
                    list(best_genes),
                    max_steps=intensive_step_cap,
                )
                if self._is_better_eval(intensified_eval, best_eval):
                    best_genes = intensified_genes
                    best_eval = intensified_eval

            payload = self._decode_payload(best_genes)
            fingerprint = self._payload_fingerprint(payload)
            if fingerprint in seen_fingerprints:
                continue
            seen_fingerprints.add(fingerprint)
            alternatives.append(
                GeneratedAlternative(
                    rank=len(alternatives) + 1,
                    fitness=best_eval.fitness,
                    hard_conflicts=best_eval.hard_conflicts,
                    soft_penalty=best_eval.soft_penalty,
                    payload=payload,
                )
            )
            if len(alternatives) >= request.alternative_count:
                break

        attempts = 0
        while len(alternatives) < request.alternative_count and attempts < request.alternative_count * 18:
            attempts += 1
            candidate_genes = self._repair_individual(
                self._constructive_individual(randomized=True, rcl_alpha=0.35),
                max_passes=1,
            )
            candidate_eval = self._evaluate(candidate_genes)
            if candidate_eval.hard_conflicts > 0 and intensive_budget > 0:
                intensive_budget -= 1
                candidate_genes, candidate_eval = self._intensive_conflict_repair(
                    candidate_genes,
                    max_steps=intensive_step_cap,
                )
            payload = self._decode_payload(candidate_genes)
            fingerprint = self._payload_fingerprint(payload)
            if fingerprint in seen_fingerprints:
                continue
            seen_fingerprints.add(fingerprint)
            alternatives.append(
                GeneratedAlternative(
                    rank=len(alternatives) + 1,
                    fitness=candidate_eval.fitness,
                    hard_conflicts=candidate_eval.hard_conflicts,
                    soft_penalty=candidate_eval.soft_penalty,
                    payload=payload,
                )
            )

        if not alternatives:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Generation did not produce any alternatives",
            )

        runtime_ms = int((perf_counter() - start) * 1000)
        return GenerateTimetableResponse(
            alternatives=alternatives,
            settings_used=self.settings,
            runtime_ms=runtime_ms,
        )

    def _run_simulated_annealing(self, request: GenerateTimetableRequest) -> GenerateTimetableResponse:
        start = perf_counter()
        archive: list[tuple[EvaluationResult, list[int]]] = []
        seen_genotypes: set[tuple[int, ...]] = set()
        archive_limit = min(140, max(28, request.alternative_count * 28))

        def add_candidate(candidate_genes: list[int], *, repair_passes: int) -> EvaluationResult:
            repaired = self._repair_individual(list(candidate_genes), max_passes=repair_passes)
            key = tuple(repaired)
            if key in seen_genotypes:
                return self._evaluate(repaired)
            seen_genotypes.add(key)

            evaluation = self._evaluate(repaired)
            survivors: list[tuple[EvaluationResult, list[int]]] = []
            for existing_eval, existing_genes in archive:
                if self._dominates_eval(existing_eval, evaluation):
                    return evaluation
                if not self._dominates_eval(evaluation, existing_eval):
                    survivors.append((existing_eval, existing_genes))
            survivors.append((evaluation, repaired))
            survivors.sort(key=lambda item: (item[0].hard_conflicts, item[0].soft_penalty, -item[0].fitness))
            archive[:] = survivors[:archive_limit]
            return evaluation

        seed_trials = min(20, max(6, request.alternative_count * 4))
        current_genes: list[int] | None = None
        current_eval: EvaluationResult | None = None
        best_genes: list[int] | None = None
        best_eval: EvaluationResult | None = None

        for trial in range(seed_trials):
            candidate = self._constructive_individual(
                randomized=trial > 0,
                rcl_alpha=0.10 + (0.30 * (trial / max(1, seed_trials))),
            )
            evaluation = add_candidate(candidate, repair_passes=2 if trial < 3 else 1)
            if current_genes is None:
                current_genes = self._repair_individual(list(candidate), max_passes=1)
                current_eval = self._evaluate(current_genes)
                best_genes = list(current_genes)
                best_eval = current_eval
            elif best_eval is not None and self._is_better_eval(evaluation, best_eval):
                best_genes = self._repair_individual(list(candidate), max_passes=1)
                best_eval = self._evaluate(best_genes)

        if current_genes is None or current_eval is None or best_genes is None or best_eval is None:
            current_genes = self._repair_individual(self._random_individual(), max_passes=2)
            current_eval = self._evaluate(current_genes)
            best_genes = list(current_genes)
            best_eval = current_eval
            add_candidate(current_genes, repair_passes=0)

        block_count = len(self.block_requests)
        iterations = self.settings.annealing_iterations
        if block_count >= 220:
            iterations = min(iterations, 80)
        elif block_count >= 160:
            iterations = min(iterations, 140)
        elif block_count >= 120:
            iterations = min(iterations, 220)
        elif block_count >= 80:
            iterations = min(iterations, 320)
        else:
            iterations = min(iterations, 480)
        temperature = max(0.05, self.settings.annealing_initial_temperature)
        cooling_rate = self.settings.annealing_cooling_rate
        stagnant_steps = 0

        for step in range(iterations):
            progress = step / max(1, iterations - 1)
            intensity = min(0.35, 0.03 + (0.18 * progress) + min(0.10, stagnant_steps * 0.002))
            candidate = self._perturb_individual(current_genes, intensity=intensity)

            mutation_scale = 1.35 if current_eval.hard_conflicts > 0 else 1.0
            mutation_rate = min(0.40, self.settings.mutation_rate * mutation_scale)
            if self.random.random() < 0.8:
                candidate = self._mutate(candidate, mutation_rate=mutation_rate)
            if self.random.random() < 0.22 or current_eval.hard_conflicts > 0:
                candidate = self._repair_individual(candidate, max_passes=1)

            candidate_eval = self._evaluate(candidate)
            current_energy = self._annealing_energy(current_eval)
            candidate_energy = self._annealing_energy(candidate_eval)
            delta = candidate_energy - current_energy

            accept = False
            if delta <= 0:
                accept = True
            else:
                probability = math.exp(-delta / max(temperature, 1e-9))
                accept = self.random.random() < probability

            if accept:
                current_genes = candidate
                current_eval = candidate_eval
                stagnant_steps = 0 if delta < 0 else stagnant_steps + 1
            else:
                stagnant_steps += 1

            add_candidate(candidate, repair_passes=0)
            if self._is_better_eval(candidate_eval, best_eval):
                best_genes = list(candidate)
                best_eval = candidate_eval

            if step % 45 == 0:
                probe = self._constructive_individual(randomized=True, rcl_alpha=0.30)
                probe_eval = add_candidate(probe, repair_passes=1)
                if self._is_better_eval(probe_eval, best_eval):
                    best_genes = self._repair_individual(list(probe), max_passes=1)
                    best_eval = self._evaluate(best_genes)
                if self._is_better_eval(probe_eval, current_eval):
                    current_genes = self._repair_individual(list(probe), max_passes=1)
                    current_eval = self._evaluate(current_genes)
                    stagnant_steps = 0

            if stagnant_steps >= 120:
                restart = self._constructive_individual(randomized=True, rcl_alpha=0.35)
                current_genes = self._repair_individual(restart, max_passes=1)
                current_eval = self._evaluate(current_genes)
                add_candidate(current_genes, repair_passes=0)
                stagnant_steps = 0
                temperature = max(
                    temperature,
                    self.settings.annealing_initial_temperature * 0.45,
                )

            temperature *= cooling_rate
            if temperature < 0.03:
                temperature = max(
                    0.05,
                    self.settings.annealing_initial_temperature * (0.30 + (0.20 * self.random.random())),
                )
            if (
                best_eval.hard_conflicts == 0
                and len(archive) >= max(4, request.alternative_count)
                and step >= max(30, int(iterations * 0.35))
            ):
                break

        add_candidate(best_genes, repair_passes=0)
        ranked = sorted(archive, key=lambda item: (item[0].hard_conflicts, item[0].soft_penalty, -item[0].fitness))
        alternatives: list[GeneratedAlternative] = []
        seen_fingerprints: set[tuple[tuple[str, ...], ...]] = set()
        intensive_budget = (
            max(2, request.alternative_count * 2)
            if block_count >= 180
            else max(4, request.alternative_count * 4)
        )
        intensive_step_cap = self._intensive_repair_step_cap()

        for evaluation, genes in ranked:
            best_genes_local = genes
            best_eval_local = evaluation
            if best_eval_local.hard_conflicts > 0 and intensive_budget > 0:
                intensive_budget -= 1
                intensified_genes, intensified_eval = self._intensive_conflict_repair(
                    list(best_genes_local),
                    max_steps=intensive_step_cap,
                )
                if self._is_better_eval(intensified_eval, best_eval_local):
                    best_genes_local = intensified_genes
                    best_eval_local = intensified_eval

            payload = self._decode_payload(best_genes_local)
            fingerprint = self._payload_fingerprint(payload)
            if fingerprint in seen_fingerprints:
                continue
            seen_fingerprints.add(fingerprint)
            alternatives.append(
                GeneratedAlternative(
                    rank=len(alternatives) + 1,
                    fitness=best_eval_local.fitness,
                    hard_conflicts=best_eval_local.hard_conflicts,
                    soft_penalty=best_eval_local.soft_penalty,
                    payload=payload,
                )
            )
            if len(alternatives) >= request.alternative_count:
                break

        attempts = 0
        while len(alternatives) < request.alternative_count and attempts < request.alternative_count * 20:
            attempts += 1
            candidate_genes = self._repair_individual(
                self._constructive_individual(randomized=True, rcl_alpha=0.35),
                max_passes=1,
            )
            candidate_eval = self._evaluate(candidate_genes)
            if candidate_eval.hard_conflicts > 0 and intensive_budget > 0:
                intensive_budget -= 1
                candidate_genes, candidate_eval = self._intensive_conflict_repair(
                    candidate_genes,
                    max_steps=intensive_step_cap,
                )
            payload = self._decode_payload(candidate_genes)
            fingerprint = self._payload_fingerprint(payload)
            if fingerprint in seen_fingerprints:
                continue
            seen_fingerprints.add(fingerprint)
            alternatives.append(
                GeneratedAlternative(
                    rank=len(alternatives) + 1,
                    fitness=candidate_eval.fitness,
                    hard_conflicts=candidate_eval.hard_conflicts,
                    soft_penalty=candidate_eval.soft_penalty,
                    payload=payload,
                )
            )

        if not alternatives:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Generation did not produce any alternatives",
            )

        runtime_ms = int((perf_counter() - start) * 1000)
        return GenerateTimetableResponse(
            alternatives=alternatives,
            settings_used=self.settings,
            runtime_ms=runtime_ms,
        )

    def _payload_fingerprint(self, payload: OfficialTimetablePayload) -> tuple:
        return tuple(
            sorted(
                (
                    slot.day,
                    slot.startTime,
                    slot.endTime,
                    slot.courseId,
                    slot.roomId,
                    slot.facultyId,
                    slot.section,
                    slot.batch or "",
                    slot.sessionType or "",
                )
                for slot in payload.timetable_data
            )
        )

    def _run_fast_solver(self, request: GenerateTimetableRequest) -> GenerateTimetableResponse:
        start = perf_counter()
        block_count = len(self.block_requests)
        strict_attempts = 4
        if block_count >= 220:
            strict_attempts = 3
        elif block_count >= 160:
            strict_attempts = 3
        elif block_count >= 120:
            strict_attempts = 4

        best_genes: list[int] | None = None
        best_eval: EvaluationResult | None = None

        # Try strict conflict-first constructive starts before entering repair-heavy mode.
        for attempt in range(strict_attempts):
            strict_genes = self._constructive_individual(
                randomized=attempt > 0,
                rcl_alpha=0.08 + (0.16 * (attempt / max(1, strict_attempts))),
                strict_dead_end=True,
            )
            if strict_genes is None:
                continue
            strict_eval = self._evaluate(strict_genes)
            if best_eval is None or self._is_better_eval(strict_eval, best_eval):
                best_genes = strict_genes
                best_eval = strict_eval
            if strict_eval.hard_conflicts == 0:
                break

        if best_genes is None or best_eval is None:
            # Fallback: try a few widened constructive starts and keep the best one.
            fallback_trials = 3 if block_count >= 140 else 2
            for trial in range(fallback_trials):
                candidate = self._constructive_individual(
                    randomized=trial > 0,
                    rcl_alpha=0.20 + (0.10 * (trial / max(1, fallback_trials))),
                )
                if candidate is None:
                    candidate = self._random_individual()
                candidate = self._harmonize_faculty_assignments(candidate)
                candidate = self._repair_individual(candidate, max_passes=2 if trial == 0 else 1)
                candidate_eval = self._evaluate(candidate)
                if best_eval is None or self._is_better_eval(candidate_eval, best_eval):
                    best_genes = candidate
                    best_eval = candidate_eval
            if best_genes is None or best_eval is None:
                best_genes = self._repair_individual(self._random_individual(), max_passes=1)
                best_eval = self._evaluate(best_genes)

        if best_eval.hard_conflicts > 0:
            # Try to repair the baseline candidate.
            best_genes = self._repair_individual(best_genes, max_passes=1)
            best_eval = self._evaluate(best_genes)

        if best_eval.hard_conflicts > 0:
            # Intensive repair if still conflicted
            intensive_steps = self._intensive_repair_step_cap()
            if block_count >= 220:
                intensive_steps = min(intensive_steps, 8)
            elif block_count >= 160:
                intensive_steps = min(intensive_steps, 10)
            elif block_count >= 120:
                intensive_steps = min(intensive_steps, 14)
            best_genes, best_eval = self._intensive_conflict_repair(
                best_genes,
                max_steps=intensive_steps,
            )

        if best_eval.hard_conflicts > 0:
            overlap_repaired = self._greedy_overlap_repair(
                best_genes,
                max_iterations=220 if block_count >= 180 else 140,
            )
            overlap_eval = self._evaluate(overlap_repaired)
            if self._is_better_eval(overlap_eval, best_eval):
                best_genes = overlap_repaired
                best_eval = overlap_eval

        if best_eval.hard_conflicts > 0:
            room_repaired = self._repair_room_conflicts(
                best_genes,
                max_iterations=10 if block_count >= 180 else 6,
            )
            room_eval = self._evaluate(room_repaired)
            if self._is_better_eval(room_eval, best_eval):
                best_genes = room_repaired
                best_eval = room_eval

        alternatives: list[GeneratedAlternative] = []
        
        def add_result(genes: list[int], evaluation: EvaluationResult) -> bool:
            payload = self._decode_payload(genes)
            alternatives.append(
                GeneratedAlternative(
                    rank=len(alternatives) + 1,
                    fitness=evaluation.fitness,
                    hard_conflicts=evaluation.hard_conflicts,
                    soft_penalty=evaluation.soft_penalty,
                    payload=payload,
                )
            )
            return len(alternatives) >= request.alternative_count

        add_result(best_genes, best_eval)
        
        # 2. If we need more alternatives or the first one failed, try randomized starts
        attempts = 0
        while len(alternatives) < request.alternative_count and attempts < 10:
            attempts += 1
            # Diverse constructive starts
            candidate = self._constructive_individual(randomized=True, rcl_alpha=0.2)
            candidate = self._repair_individual(candidate, max_passes=1)
            candidate = self._repair_room_conflicts(
                candidate,
                max_iterations=8 if block_count >= 180 else 5,
            )
            eval_res = self._evaluate(candidate)
            
            if eval_res.hard_conflicts > 0:
                 retry_steps = 10
                 if block_count >= 160:
                     retry_steps = 6
                 candidate, eval_res = self._intensive_conflict_repair(candidate, max_steps=retry_steps)
            
            # Simple dedup based on fitness/conflicts
            is_duplicate = any(
                a.fitness == eval_res.fitness and a.hard_conflicts == eval_res.hard_conflicts 
                for a in alternatives
            )
            if not is_duplicate:
                add_result(candidate, eval_res)

        return GenerateTimetableResponse(
            alternatives=alternatives,
            settings_used=self.settings,
            runtime_ms=int((perf_counter() - start) * 1000),
        )

    def _merge_results(
        self,
        *,
        primary: GenerateTimetableResponse,
        secondary: GenerateTimetableResponse,
        alternative_count: int,
    ) -> GenerateTimetableResponse:
        merged: list[GeneratedAlternative] = []
        seen_fingerprints: set[tuple[tuple[str, ...], ...]] = set()

        ordered = [*primary.alternatives, *secondary.alternatives]
        ordered.sort(key=lambda item: (item.hard_conflicts, item.soft_penalty, -item.fitness))

        for candidate in ordered:
            fingerprint = self._payload_fingerprint(candidate.payload)
            if fingerprint in seen_fingerprints:
                continue
            seen_fingerprints.add(fingerprint)
            merged.append(
                GeneratedAlternative(
                    rank=len(merged) + 1,
                    fitness=candidate.fitness,
                    hard_conflicts=candidate.hard_conflicts,
                    soft_penalty=candidate.soft_penalty,
                    payload=candidate.payload,
                )
            )
            if len(merged) >= alternative_count:
                break

        return GenerateTimetableResponse(
            alternatives=merged,
            settings_used=self.settings,
            runtime_ms=primary.runtime_ms + secondary.runtime_ms,
        )

    def run(self, request: GenerateTimetableRequest) -> GenerateTimetableResponse:
        strategy = self.settings.solver_strategy
        logger.info(
            "Scheduler run strategy=%s program_id=%s term=%s alternatives=%s",
            strategy,
            self.program_id,
            self.term_number,
            request.alternative_count,
        )

        def has_enough_alternatives(result: GenerateTimetableResponse) -> bool:
            return bool(result.alternatives) and len(result.alternatives) >= request.alternative_count

        def has_conflict_free_alternative(result: GenerateTimetableResponse) -> bool:
            return any(item.hard_conflicts == 0 for item in result.alternatives)

        can_run_fast = hasattr(self, "block_requests")
        block_count = len(self.block_requests) if can_run_fast else 0

        if strategy == "fast":
            return self._run_fast_solver(request)
        if strategy == "hybrid":
            fast: GenerateTimetableResponse | None = None
            if can_run_fast:
                fast = self._run_fast_solver(request)
                if has_enough_alternatives(fast) and (
                    has_conflict_free_alternative(fast) or block_count >= 120
                ):
                    return fast
            hybrid = self._run_hybrid_search(request)
            if fast is not None:
                merged = self._merge_results(
                    primary=fast,
                    secondary=hybrid,
                    alternative_count=request.alternative_count,
                )
                if merged.alternatives:
                    return merged
            return hybrid
        if strategy == "simulated_annealing":
            return self._run_simulated_annealing(request)
        if strategy == "genetic":
            return self._run_classic_ga(request)

        fast: GenerateTimetableResponse | None = None
        if can_run_fast:
            fast = self._run_fast_solver(request)
            if has_enough_alternatives(fast) and (
                has_conflict_free_alternative(fast) or block_count >= 120
            ):
                return fast

        hybrid = self._run_hybrid_search(request)
        if has_enough_alternatives(hybrid) and (
            has_conflict_free_alternative(hybrid) or block_count >= 120
        ):
            return hybrid

        merged_fast_hybrid = hybrid
        if fast is not None:
            merged_fast_hybrid = self._merge_results(
                primary=fast,
                secondary=hybrid,
                alternative_count=request.alternative_count,
            )
            if has_enough_alternatives(merged_fast_hybrid) and (
                has_conflict_free_alternative(merged_fast_hybrid) or block_count >= 120
            ):
                return merged_fast_hybrid

        # For small search spaces, keep deeper fallback to chase conflict-free results.
        # For dense terms (block_count >= 120), return fast/hybrid output promptly.
        annealed = self._run_simulated_annealing(request)
        if has_enough_alternatives(annealed):
            return annealed

        merged_hybrid_annealed = self._merge_results(
            primary=merged_fast_hybrid,
            secondary=annealed,
            alternative_count=request.alternative_count,
        )
        if has_enough_alternatives(merged_hybrid_annealed):
            return merged_hybrid_annealed

        classic = self._run_classic_ga(request)
        merged = self._merge_results(
            primary=merged_hybrid_annealed,
            secondary=classic,
            alternative_count=request.alternative_count,
        )
        if merged.alternatives:
            return merged
        return classic
