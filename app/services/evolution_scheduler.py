from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import logging
import math
from pathlib import Path
import random
import threading
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Literal

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.core.exceptions import SchedulerError

from app.models.course import Course, CourseType
from app.models.faculty import Faculty
from app.models.institution_settings import InstitutionSettings
from app.models.program_constraint import ProgramConstraint
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
HYPERPARAMETER_CACHE_FILE = Path(__file__).resolve().parents[2] / ".ga_hyperparameter_cache.json"
THREE_SLOT_PRACTICAL_COURSE_CODES = {"23MEE115", "23ECE285"}
THREE_SLOT_PRACTICAL_NAME_MARKERS = {
    "manufacturing practice",
    "digital electronics laboratory",
}
VIRTUAL_RESOURCE_HASH_SIZE = 12
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


def minutes_to_time(value: int) -> str:
    hours = value // 60
    minutes = value % 60
    return f"{hours:02d}:{minutes:02d}"


def normalize_day(value: str) -> str:
    return DAY_SHORT_MAP.get(value, value)


def _is_removed_legacy_slot_range(start: int, end: int) -> bool:
    return (start, end) in REMOVED_LEGACY_SLOT_RANGES


def _is_canonical_lunch_range(start: int, end: int) -> bool:
    return start == CANONICAL_LUNCH_START_MINUTES and end == CANONICAL_LUNCH_END_MINUTES


def _overlaps_canonical_lunch(start: int, end: int) -> bool:
    return start < CANONICAL_LUNCH_END_MINUTES and end > CANONICAL_LUNCH_START_MINUTES


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
    requires_faculty: bool = True
    requires_room: bool = True


@dataclass
class EvaluationResult:
    fitness: float
    hard_conflicts: int
    soft_penalty: float
    workload_balance_penalty: float = 0.0
    objectives: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class SearchHyperParameters:
    population_size: int
    generations: int
    mutation_rate: float
    crossover_rate: float
    elite_count: int
    tournament_size: int
    stagnation_limit: int
    annealing_iterations: int
    annealing_initial_temperature: float
    annealing_cooling_rate: float


class EvolutionaryScheduler:
    """
    Fully automatic timetable optimizer:
    1. GA-based hyperparameter tuning (meta-optimization).
    2. MOEA search for exploration (Pareto-driven population evolution).
    3. Simulated Annealing for exploitation (local intensification on promising solutions).
    """

    _hyperparameter_cache_lock = threading.Lock()
    _hyperparameter_cache_loaded = False
    _hyperparameter_cache_by_scenario: dict[str, SearchHyperParameters] = {}
    program_constraint: ProgramConstraint | None = None

    def __init__(
        self,
        *,
        db: Session,
        program_id: str,
        term_number: int,
        settings: GenerationSettingsBase,
        reserved_resource_slots: list[dict] | None = None,
        progress_reporter: Callable[[dict[str, Any]], None] | None = None,
        hyperparameter_cache_scope: str = "single",
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
        self.progress_reporter = progress_reporter
        self.hyperparameter_cache_scope = hyperparameter_cache_scope

        self.program_constraint = self._load_program_constraint()
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
        self.rooms = {
            room.id: room
            for room in self.db.execute(select(Room).where(Room.program_id == self.program_id)).scalars().all()
        }
        if not self.rooms:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No rooms available for generation")
        self._virtual_room_ids: set[str] = set()

        room_type_counts = Counter(r.type for r in self.rooms.values())
        logger.info("Rooms loaded: total=%d, counts=%s", len(self.rooms), dict(room_type_counts))

        self.faculty = {
            item.id: item
            for item in self.db.execute(select(Faculty).where(Faculty.program_id == self.program_id)).scalars().all()
        }
        if not self.faculty:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No faculty available for generation")
        self._virtual_faculty_ids: set[str] = set()
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
        self.three_term_horizon_terms = self._resolve_three_term_horizon_terms()
        self.faculty_three_term_baseline_minutes = self._load_three_term_baseline_minutes()

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

    @staticmethod
    def _normalize_project_phase_text(value: str | None) -> str:
        raw = (value or "").lower().replace("-", " ").replace("_", " ")
        return " ".join(raw.split())

    def _is_project_phase_course(self, course: Course | None) -> bool:
        if course is None:
            return False
        name = self._normalize_project_phase_text(getattr(course, "name", ""))
        code = self._normalize_project_phase_text(getattr(course, "code", ""))
        return "project phase" in name or "project phase" in code

    def _is_project_phase_request(self, req: BlockRequest) -> bool:
        return self._is_project_phase_course(self.courses.get(req.course_id))

    @staticmethod
    def _normalize_course_identity_text(value: str | None) -> str:
        return " ".join((value or "").strip().lower().replace("-", " ").replace("_", " ").split())

    def _is_three_slot_practical_course(self, course: Course | None) -> bool:
        if course is None:
            return False
        code = str(getattr(course, "code", "") or "").strip().upper()
        if code in THREE_SLOT_PRACTICAL_COURSE_CODES:
            return True
        name = self._normalize_course_identity_text(getattr(course, "name", ""))
        return any(marker in name for marker in THREE_SLOT_PRACTICAL_NAME_MARKERS)

    def _practical_block_slot_size_for_course(self, course: Course | None) -> int:
        if self.program_constraint is not None and not self.program_constraint.enforce_lab_contiguous_blocks:
            return 1
        if course is None:
            return max(1, int(self.schedule_policy.lab_contiguous_slots or 2))
        configured = getattr(course, "practical_contiguous_slots", None)
        if configured is not None:
            block_size = max(1, int(configured))
        elif self._is_three_slot_practical_course(course):
            block_size = 3
        else:
            # Fallback to institution-level default when course-level value is not provided.
            block_size = max(1, int(self.schedule_policy.lab_contiguous_slots or 2))
        practical_hours = max(0, int(course.lab_hours or 0))
        if practical_hours > 0:
            block_size = min(block_size, practical_hours)
        return max(1, block_size)

    @staticmethod
    def _computed_course_credits(course: Course) -> float:
        """
        Compute academic credits from LTP split:
        Credits = L + T + (P / 2)
        where:
          L = theory hours, T = tutorial hours, P = lab/practical hours.
        """
        theory = max(0, int(course.theory_hours or 0))
        tutorial = max(0, int(course.tutorial_hours or 0))
        practical = max(0, int(course.lab_hours or 0))
        return float(theory + tutorial + (practical / 2.0))

    def _weekly_period_units_for_course(self, course: Course) -> int:
        split_units = (
            max(0, int(course.theory_hours or 0))
            + max(0, int(course.tutorial_hours or 0))
            + max(0, int(course.lab_hours or 0))
        )
        if split_units > 0:
            return split_units
        return max(0, int(course.hours_per_week or 0))

    def _resolve_three_term_horizon_terms(self) -> list[int]:
        window_size = 3
        if self.program_constraint is not None:
            window_size = max(1, int(self.program_constraint.temporal_window_semesters or 3))
        terms: list[int] = []
        next_term = self.term_number
        while next_term <= 20 and len(terms) < window_size:
            terms.append(next_term)
            next_term += 2

        prev_term = self.term_number - 2
        while prev_term >= 1 and len(terms) < window_size:
            terms.append(prev_term)
            prev_term -= 2

        return sorted(set(terms))

    def _load_three_term_baseline_minutes(self) -> dict[str, int]:
        baseline_minutes: dict[str, int] = {faculty_id: 0 for faculty_id in self.faculty.keys()}
        prior_terms = [term for term in self.three_term_horizon_terms if term != self.term_number]
        if not prior_terms:
            return baseline_minutes

        section_counts_by_term: dict[int, int] = defaultdict(int)
        for term_number, section_count in (
            self.db.execute(
                select(ProgramSection.term_number, ProgramSection.id)
                .where(
                    ProgramSection.program_id == self.program_id,
                    ProgramSection.term_number.in_(prior_terms),
                )
            )
            .all()
        ):
            if section_count:
                section_counts_by_term[term_number] += 1

        rows = (
            self.db.execute(
                select(ProgramCourse).where(
                    ProgramCourse.program_id == self.program_id,
                    ProgramCourse.term_number.in_(prior_terms),
                )
            )
            .scalars()
            .all()
        )
        if not rows:
            return baseline_minutes

        rows_by_term: dict[int, list[ProgramCourse]] = defaultdict(list)
        for row in rows:
            rows_by_term[row.term_number].append(row)

        active_rows: list[ProgramCourse] = []
        for term_rows in rows_by_term.values():
            required = [item for item in term_rows if item.is_required]
            active_rows.extend(required if required else term_rows)

        if not active_rows:
            return baseline_minutes

        prior_course_ids = sorted({row.course_id for row in active_rows})
        prior_courses = {
            item.id: item
            for item in self.db.execute(select(Course).where(Course.id.in_(prior_course_ids))).scalars().all()
        }
        if not prior_courses:
            return baseline_minutes

        period_minutes = max(1, self.schedule_policy.period_minutes)
        for row in active_rows:
            course = prior_courses.get(row.course_id)
            if course is None or not course.faculty_id:
                continue
            if course.faculty_id not in baseline_minutes:
                continue
            weekly_units = self._weekly_period_units_for_course(course)
            if weekly_units <= 0:
                continue
            section_multiplier = max(1, section_counts_by_term.get(row.term_number, 0))
            practical_units = max(0, int(course.lab_hours or 0))
            lecture_units = max(0, weekly_units - practical_units)
            practical_multiplier = max(1, row.lab_batch_count) if practical_units > 0 else 1
            effective_units = lecture_units + (practical_units * practical_multiplier)
            baseline_minutes[course.faculty_id] += effective_units * section_multiplier * period_minutes

        return baseline_minutes

    def _three_term_workload_balance_penalty(self, current_minutes: dict[str, int]) -> float:
        baseline_minutes = getattr(self, "faculty_three_term_baseline_minutes", {}) or {}
        active_faculty_ids = [
            faculty_id
            for faculty_id, faculty in self.faculty.items()
            if self._effective_faculty_max_hours(faculty) > 0
        ]
        if len(active_faculty_ids) <= 1:
            return 0.0

        combined_minutes: list[float] = []
        for faculty_id in active_faculty_ids:
            combined_minutes.append(
                float(baseline_minutes.get(faculty_id, 0) + current_minutes.get(faculty_id, 0))
            )
        if not combined_minutes:
            return 0.0

        average = sum(combined_minutes) / len(combined_minutes)
        if average <= 0:
            return 0.0

        mad = sum(abs(value - average) for value in combined_minutes) / len(combined_minutes)
        period = max(1.0, float(self.schedule_policy.period_minutes))
        return mad / period

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
        # Project phases are synchronized across sections as one combined event.
        sync_scope = "__project_phase__" if self._is_project_phase_request(req) else req.section
        return (req.course_id, sync_scope, req.session_type, req.block_size)

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
        check_room: bool = True,
        check_faculty: bool = True,
    ) -> tuple[bool, bool]:
        room_conflict = False
        faculty_conflict = False
        for reserved_start, reserved_end, reserved_room_id, reserved_faculty_id in self.reserved_resource_slots_by_day.get(day, []):
            overlaps = start_min < reserved_end and reserved_start < end_min
            if not overlaps:
                continue
            if check_room and reserved_room_id and reserved_room_id == room_id:
                room_conflict = True
            if check_faculty and reserved_faculty_id and reserved_faculty_id == faculty_id:
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
        check_room: bool = True,
        check_faculty: bool = True,
    ) -> bool:
        room_conflict, faculty_conflict = self._reserved_conflict_flags(
            day=day,
            start_min=start_min,
            end_min=end_min,
            room_id=room_id,
            faculty_id=faculty_id,
            check_room=check_room,
            check_faculty=check_faculty,
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

    def _load_program_constraint(self) -> ProgramConstraint | None:
        return (
            self.db.execute(select(ProgramConstraint).where(ProgramConstraint.program_id == self.program_id))
            .scalars()
            .first()
        )

    def _effective_faculty_max_hours(self, faculty: Faculty) -> int:
        configured_cap = 0
        program_constraint = getattr(self, "program_constraint", None)
        if program_constraint is not None:
            configured_cap = max(0, int(program_constraint.faculty_max_hours_per_week or 0))
        faculty_cap = max(0, int(faculty.max_hours or 0))
        if configured_cap and faculty_cap:
            return min(configured_cap, faculty_cap)
        return configured_cap or faculty_cap

    def _faculty_min_target_hours(self, faculty: Faculty) -> int:
        configured_min = 0
        program_constraint = getattr(self, "program_constraint", None)
        if program_constraint is not None:
            configured_min = max(0, int(program_constraint.faculty_min_hours_per_week or 0))
        faculty_target = max(0, int(faculty.workload_hours or 0))
        return max(configured_min, faculty_target)

    def _overlaps_non_teaching_window(self, *, day: str, start_min: int, end_min: int) -> bool:
        windows = getattr(self, "non_teaching_windows_by_day", {}).get(day, [])
        return any(start_min < window_end and end_min > window_start for window_start, window_end in windows)

    def _build_day_slots(self) -> dict[str, list[SlotSegment]]:
        self.non_teaching_windows_by_day: dict[str, list[tuple[int, int]]] = {}
        configured_daily_slots = []
        if self.program_constraint is not None:
            configured_daily_slots = list(self.program_constraint.daily_time_slots or [])

        if configured_daily_slots:
            normalized_slots: list[tuple[int, int, str]] = []
            for item in configured_daily_slots:
                try:
                    start_raw = str(item.get("start_time", "")).strip()
                    end_raw = str(item.get("end_time", "")).strip()
                    tag_raw = str(item.get("tag", "teaching")).strip().lower() or "teaching"
                    if tag_raw not in {"teaching", "block", "break", "lunch"}:
                        tag_raw = "teaching"
                    start = parse_time_to_minutes(start_raw)
                    end = parse_time_to_minutes(end_raw)
                except Exception:
                    continue
                if end <= start:
                    continue
                if _is_removed_legacy_slot_range(start, end):
                    continue
                if _overlaps_canonical_lunch(start, end) and not _is_canonical_lunch_range(start, end):
                    continue
                if _is_canonical_lunch_range(start, end):
                    tag_raw = "lunch"
                elif tag_raw == "lunch":
                    continue
                normalized_slots.append((start, end, tag_raw))

            if not any(_is_canonical_lunch_range(start, end) for start, end, _tag in normalized_slots):
                normalized_slots.append((CANONICAL_LUNCH_START_MINUTES, CANONICAL_LUNCH_END_MINUTES, "lunch"))

            normalized_slots.sort(key=lambda item: item[0])
            day_slots: dict[str, list[SlotSegment]] = {}
            enabled_days = [entry.day for entry in self.working_hours if entry.enabled]
            for day in enabled_days:
                teaching: list[SlotSegment] = []
                blocked: list[tuple[int, int]] = []
                for start, end, tag in normalized_slots:
                    if tag == "teaching":
                        teaching.append(SlotSegment(start=start, end=end))
                    else:
                        blocked.append((start, end))
                if teaching:
                    day_slots[day] = teaching
                if blocked:
                    self.non_teaching_windows_by_day[day] = blocked

            if day_slots:
                return day_slots

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
                self.non_teaching_windows_by_day[entry.day] = [
                    (parse_time_to_minutes(item.start_time), parse_time_to_minutes(item.end_time))
                    for item in self.schedule_policy.breaks
                ]
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
        configured_period_units = 0
        for program_course in self.program_courses:
            course = self.courses.get(program_course.course_id)
            if course is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Course mapping missing for course id {program_course.course_id}",
                )
            configured_period_units += self._weekly_period_units_for_course(course)

        if configured_period_units <= 0:
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
        enforce_student_credit_load = True
        if self.program_constraint is not None:
            enforce_student_credit_load = bool(self.program_constraint.enforce_student_credit_load)
        if enforce_student_credit_load and term is not None and term.credits_required > 0:
            total_credits = 0.0
            for program_course in self.program_courses:
                course = self.courses.get(program_course.course_id)
                if course:
                    total_credits += self._computed_course_credits(course)

            if not math.isclose(total_credits, float(term.credits_required), abs_tol=0.01):
                logger.warning(
                    "Curriculum credit mismatch in term %s: Computed LTP credits %.2f "
                    "(L+T+P/2) but term requires %.2f",
                    self.term_number,
                    total_credits,
                    float(term.credits_required),
                )

        # TARGET HOURS: Sum of effective weekly period units is what must be scheduled.
        target_hours = configured_period_units

        return target_hours * self.schedule_policy.period_minutes

    def _validate_course_credit_alignment(self) -> None:
        enforce_ltp_split = True
        if self.program_constraint is not None:
            enforce_ltp_split = bool(self.program_constraint.enforce_ltp_split)
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
            if enforce_ltp_split and total_split != course.hours_per_week:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"ShedForge Data Parity Mismatch: Course {course.code} "
                        f"Hours/Week ({course.hours_per_week}) must exactly match "
                        f"the sum of theory+lab+tutorial hours ({total_split})."
                    ),
                )

            computed_credits = self._computed_course_credits(course)
            if computed_credits <= 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"Course {course.code} has invalid LTP split for credit computation "
                        "(L+T+P/2 must be > 0)."
                    ),
                )

    def _validate_total_faculty_capacity(self) -> None:
        virtual_faculty_ids = getattr(self, "_virtual_faculty_ids", set())
        total_required_minutes = sum(
            req.block_size * self.schedule_policy.period_minutes
            for req in self.block_requests
            if self._request_requires_faculty(req)
        )
        total_capacity_minutes = sum(
            self._effective_faculty_max_hours(item) * 60
            for faculty_id, item in self.faculty.items()
            if faculty_id not in virtual_faculty_ids
        )
        if total_required_minutes <= 0:
            return
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
        total_credits = 0.0
        for pc in self.program_courses:
            course = self.courses.get(pc.course_id)
            if course:
                configured_hours += self._weekly_period_units_for_course(course)
                total_credits += self._computed_course_credits(course)
        
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
        enforce_student_credit_load = True
        if self.program_constraint is not None:
            enforce_student_credit_load = bool(self.program_constraint.enforce_student_credit_load)
        if (
            enforce_student_credit_load
            and term
            and term.credits_required > 0
            and not math.isclose(total_credits, float(term.credits_required), abs_tol=0.01)
        ):
            logger.warning(
                "Credit load mismatch for term %s: Computed LTP credits total %.2f "
                "(L+T+P/2), but term requires %.2f",
                self.term_number,
                total_credits,
                float(term.credits_required),
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

    def _room_candidates_for(
        self,
        course: Course,
        *,
        session_type: Literal["theory", "tutorial", "lab"] | None = None,
    ) -> list[Room]:
        practical_session = session_type == "lab"
        if practical_session:
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

    @staticmethod
    def _intervals_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
        return start_a < end_b and start_b < end_a

    def _faculty_windows_allow_block(self, faculty_id: str, day: str, start_min: int, end_min: int) -> bool:
        windows_by_day = self.faculty_windows.get(faculty_id, {})
        day_windows = windows_by_day.get(day, [])
        if day_windows:
            return any(window_start <= start_min and end_min <= window_end for window_start, window_end in day_windows)
        faculty = self.faculty.get(faculty_id)
        return not bool(faculty and faculty.availability_windows)

    def _faculty_has_schedule_overlap(
        self,
        *,
        faculty_id: str,
        day: str,
        start_min: int,
        end_min: int,
        faculty_schedule: dict[str, list[tuple[str, int, int]]],
    ) -> bool:
        for scheduled_day, scheduled_start, scheduled_end in faculty_schedule.get(faculty_id, []):
            if scheduled_day != day:
                continue
            if self._intervals_overlap(start_min, end_min, scheduled_start, scheduled_end):
                return True
        return False

    def _assistant_candidates_for_course(self, course: Course | None, *, primary_faculty_id: str) -> list[str]:
        ordered: list[str] = []
        if course is not None:
            for faculty_id in self._faculty_candidates_for_course(course):
                if faculty_id == primary_faculty_id or faculty_id in ordered:
                    continue
                ordered.append(faculty_id)

        fallback = sorted(
            [faculty_id for faculty_id in self.faculty.keys() if faculty_id != primary_faculty_id and faculty_id not in ordered],
            key=lambda faculty_id: (
                self.faculty[faculty_id].workload_hours,
                -self.faculty[faculty_id].max_hours,
                self.faculty[faculty_id].name,
                faculty_id,
            ),
        )
        ordered.extend(fallback)
        return ordered

    def _select_assisting_faculty_ids(
        self,
        *,
        course: Course | None,
        primary_faculty_id: str,
        day: str,
        start_min: int,
        end_min: int,
        faculty_schedule: dict[str, list[tuple[str, int, int]]],
        required_count: int = 2,
        allow_relaxed_fallback: bool = False,
    ) -> tuple[str, ...]:
        if required_count <= 0:
            return tuple()

        ordered_candidates = self._assistant_candidates_for_course(course, primary_faculty_id=primary_faculty_id)
        if not ordered_candidates:
            return tuple()

        chosen: list[str] = []

        def try_fill(*, enforce_overlap_free: bool, enforce_day_windows: bool) -> None:
            for candidate_id in ordered_candidates:
                if len(chosen) >= required_count:
                    return
                if candidate_id in chosen:
                    continue

                faculty = self.faculty[candidate_id]
                if enforce_day_windows:
                    if not self._faculty_allows_day(faculty, day):
                        continue
                    if not self._faculty_windows_allow_block(candidate_id, day, start_min, end_min):
                        continue
                if enforce_overlap_free and self._faculty_has_schedule_overlap(
                    faculty_id=candidate_id,
                    day=day,
                    start_min=start_min,
                    end_min=end_min,
                    faculty_schedule=faculty_schedule,
                ):
                    continue
                chosen.append(candidate_id)

        # 1) Strict: available day/window and no schedule overlap.
        try_fill(enforce_overlap_free=True, enforce_day_windows=True)
        # Optional fallbacks are disabled by default so assistants obey
        # the same overlap/availability expectations as primary faculty.
        if allow_relaxed_fallback:
            if len(chosen) < required_count:
                try_fill(enforce_overlap_free=False, enforce_day_windows=True)
            if len(chosen) < required_count:
                try_fill(enforce_overlap_free=False, enforce_day_windows=False)

        if len(chosen) < required_count:
            logger.warning(
                "Insufficient assisting faculty for practical block | program_id=%s term=%s course=%s day=%s start=%s end=%s required=%s selected=%s",
                self.program_id,
                self.term_number,
                course.code if course is not None else "unknown",
                day,
                start_min,
                end_min,
                required_count,
                len(chosen),
            )

        return tuple(chosen[:required_count])

    def _assign_assisting_faculty(
        self,
        *,
        selected_assignments: list[tuple[int, BlockRequest, PlacementOption]],
        faculty_schedule: dict[str, list[tuple[str, int, int]]],
        required_count: int = 2,
        allow_relaxed_fallback: bool = False,
    ) -> tuple[dict[int, tuple[str, ...]], int]:
        assisting_faculty_by_request: dict[int, tuple[str, ...]] = {}
        missing_assistant_slots = 0

        for req_index, req, option in selected_assignments:
            if (
                req.session_type != "lab"
                or req.block_size != 2
                or not self._request_requires_faculty(req)
            ):
                continue

            course = self.courses.get(req.course_id)
            block_start, block_end = self._option_bounds(option, req.block_size)
            assisting_faculty_ids = self._select_assisting_faculty_ids(
                course=course,
                primary_faculty_id=option.faculty_id,
                day=option.day,
                start_min=block_start,
                end_min=block_end,
                faculty_schedule=faculty_schedule,
                required_count=required_count,
                allow_relaxed_fallback=allow_relaxed_fallback,
            )
            if assisting_faculty_ids:
                assisting_faculty_by_request[req_index] = assisting_faculty_ids
                for assistant_faculty_id in assisting_faculty_ids:
                    faculty_schedule[assistant_faculty_id].append((option.day, block_start, block_end))

            if len(assisting_faculty_ids) < required_count:
                missing_assistant_slots += required_count - len(assisting_faculty_ids)

        return assisting_faculty_by_request, missing_assistant_slots

    @staticmethod
    def _faculty_minutes_from_schedule(
        faculty_schedule: dict[str, list[tuple[str, int, int]]],
    ) -> dict[str, int]:
        minutes: dict[str, int] = defaultdict(int)
        for faculty_id, entries in faculty_schedule.items():
            total = 0
            for _day, start_min, end_min in entries:
                if end_min > start_min:
                    total += end_min - start_min
            minutes[faculty_id] = total
        return minutes

    def _build_research_slot_payloads(
        self,
        *,
        faculty_schedule: dict[str, list[tuple[str, int, int]]],
        faculty_minutes: dict[str, int],
    ) -> tuple[list[dict], list[dict], list[dict]]:
        program_constraint = self.program_constraint
        if program_constraint is None or not bool(program_constraint.auto_assign_research_slots):
            return [], [], []

        period_minutes = max(1, int(self.schedule_policy.period_minutes))
        virtual_faculty_ids = getattr(self, "_virtual_faculty_ids", set())

        research_slots: list[dict] = []
        research_courses: list[dict] = []
        research_rooms: list[dict] = []

        for faculty in sorted(self.faculty.values(), key=lambda item: (item.name.lower(), item.id)):
            if faculty.id in virtual_faculty_ids:
                continue

            max_hours = self._effective_faculty_max_hours(faculty)
            target_hours = self._faculty_min_target_hours(faculty)
            if max_hours <= 0 or target_hours <= 0:
                continue

            desired_minutes = min(max_hours, target_hours) * 60
            current_minutes = faculty_minutes.get(faculty.id, 0)
            if current_minutes >= desired_minutes:
                continue

            needed_periods = math.ceil((desired_minutes - current_minutes) / period_minutes)
            if needed_periods <= 0:
                continue

            assigned_segments: list[tuple[str, int, int]] = []
            for day, segments in self.day_slots.items():
                for segment in segments:
                    start_min, end_min = segment.start, segment.end
                    if not self._faculty_allows_day(faculty, day):
                        continue
                    if not self._faculty_windows_allow_block(faculty.id, day, start_min, end_min):
                        continue
                    if self._faculty_has_schedule_overlap(
                        faculty_id=faculty.id,
                        day=day,
                        start_min=start_min,
                        end_min=end_min,
                        faculty_schedule=faculty_schedule,
                    ):
                        continue

                    _room_reserved, faculty_reserved = self._reserved_conflict_flags(
                        day=day,
                        start_min=start_min,
                        end_min=end_min,
                        room_id="",
                        faculty_id=faculty.id,
                        check_room=False,
                        check_faculty=True,
                    )
                    if faculty_reserved:
                        continue

                    assigned_segments.append((day, start_min, end_min))
                    faculty_schedule[faculty.id].append((day, start_min, end_min))
                    faculty_minutes[faculty.id] = faculty_minutes.get(faculty.id, 0) + (end_min - start_min)

                    if len(assigned_segments) >= needed_periods:
                        break
                if len(assigned_segments) >= needed_periods:
                    break

            if not assigned_segments:
                continue
            if len(assigned_segments) < needed_periods:
                logger.info(
                    "Research slot backfill is partial for faculty %s: needed=%s assigned=%s",
                    faculty.id,
                    needed_periods,
                    len(assigned_segments),
                )

            digest = hashlib.blake2b(
                f"research|{self.program_id}|{self.term_number}|{faculty.id}".encode("utf-8"),
                digest_size=8,
            ).hexdigest()
            course_id = f"res-c-{digest}"[:36]
            room_id = f"res-r-{digest}"[:36]
            section_name = f"RS-{faculty.id[:6]}".upper()
            weekly_periods = len(assigned_segments)

            research_rooms.append(
                {
                    "id": room_id,
                    "name": f"Research Desk ({faculty.name})"[:100],
                    "capacity": 1,
                    "type": "seminar",
                    "building": "Research",
                    "hasLabEquipment": False,
                    "hasProjector": False,
                }
            )
            research_courses.append(
                {
                    "id": course_id,
                    "code": f"RS-{digest[:6].upper()}",
                    "name": f"Research Slot - {faculty.name}"[:200],
                    "type": "theory",
                    "credits": float(weekly_periods),
                    "facultyId": faculty.id,
                    "duration": 1,
                    "hoursPerWeek": weekly_periods,
                    "semesterNumber": self.term_number,
                    "batchYear": 1,
                    "theoryHours": weekly_periods,
                    "labHours": 0,
                    "tutorialHours": 0,
                    "batchSegregation": False,
                    "practicalContiguousSlots": 1,
                    "assignFaculty": True,
                    "assignClassroom": False,
                    "defaultRoomId": room_id,
                    "electiveCategory": "Research",
                }
            )

            for index, (day, start_min, end_min) in enumerate(assigned_segments):
                slot_digest = hashlib.blake2b(
                    f"{course_id}|{day}|{start_min}|{end_min}|{index}".encode("utf-8"),
                    digest_size=6,
                ).hexdigest()
                research_slots.append(
                    {
                        "id": f"res-{slot_digest}"[:36],
                        "day": day,
                        "startTime": minutes_to_time(start_min),
                        "endTime": minutes_to_time(end_min),
                        "courseId": course_id,
                        "roomId": room_id,
                        "facultyId": faculty.id,
                        "section": section_name,
                        "batch": None,
                        "studentCount": 1,
                        "sessionType": "theory",
                        "assistantFacultyIds": [],
                    }
                )

        return research_slots, research_courses, research_rooms

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

    @staticmethod
    def _request_requires_faculty(req: BlockRequest) -> bool:
        return bool(getattr(req, "requires_faculty", True))

    @staticmethod
    def _request_requires_room(req: BlockRequest) -> bool:
        return bool(getattr(req, "requires_room", True))

    def _virtual_resource_id(self, *, kind: Literal["faculty", "room"], key: str) -> str:
        digest = hashlib.blake2b(
            f"{kind}|{self.program_id}|{self.term_number}|{key}".encode("utf-8"),
            digest_size=VIRTUAL_RESOURCE_HASH_SIZE,
        ).hexdigest()
        prefix = "nr-f" if kind == "faculty" else "nr-r"
        return f"{prefix}-{digest}"

    def _ensure_virtual_faculty(self, faculty_id: str) -> str:
        if faculty_id in self.faculty:
            return faculty_id
        virtual = Faculty(
            id=faculty_id,
            program_id=self.program_id,
            name="No Faculty Required",
            email=f"{faculty_id}@shedforge.app",
            department="N/A",
            workload_hours=0,
            max_hours=0,
            availability=[],
            availability_windows=[],
            preferred_subject_codes=[],
            semester_preferences={},
        )
        self.faculty[faculty_id] = virtual
        self._virtual_faculty_ids.add(faculty_id)
        self.faculty_windows[faculty_id] = {}
        self.faculty_preferred_subject_codes[faculty_id] = set()
        return faculty_id

    def _ensure_virtual_room(self, room_id: str, *, session_type: Literal["theory", "tutorial", "lab"]) -> str:
        if room_id in self.rooms:
            return room_id
        room_type = RoomType.lab if session_type == "lab" else RoomType.lecture
        virtual = Room(
            id=room_id,
            program_id=self.program_id,
            name="No Classroom Required",
            building="N/A",
            capacity=1000,
            type=room_type,
            has_lab_equipment=session_type == "lab",
            has_projector=False,
            availability_windows=[],
        )
        self.rooms[room_id] = virtual
        self._virtual_room_ids.add(room_id)
        self.room_windows[room_id] = {}
        return room_id

    def _faculty_candidates_for_course(self, course: Course) -> list[str]:
        virtual_faculty_ids = getattr(self, "_virtual_faculty_ids", set())
        candidate_pool_ids = [
            faculty_id for faculty_id in self.faculty.keys() if faculty_id not in virtual_faculty_ids
        ]
        ordered_ids: list[str] = []
        if (
            course.faculty_id
            and course.faculty_id in self.faculty
            and course.faculty_id not in virtual_faculty_ids
        ):
            ordered_ids.append(course.faculty_id)

        preferred_ids = sorted(
            [
                item.id
                for item in self.faculty.values()
                if item.id in candidate_pool_ids
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
            [item_id for item_id in candidate_pool_ids if item_id not in ordered_ids],
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

        # Preferences are guidance only; keep a wide candidate pool so workload balancing can reassign when needed.
        if len(ordered_ids) <= 32:
            return ordered_ids
        candidate_cap = max(32, math.ceil(len(ordered_ids) * 0.90))
        return ordered_ids[: min(len(ordered_ids), candidate_cap)]

    def _build_block_requests(self) -> list[BlockRequest]:
        requests: list[BlockRequest] = []
        request_id = 0

        for program_course in self.program_courses:
            course = self.courses.get(program_course.course_id)
            if course is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Course mapping missing for course id {program_course.course_id}",
                )
            if bool(getattr(course, "assign_faculty", True)) and course.faculty_id and course.faculty_id not in self.faculty:
                logger.warning(
                    "Course %s has stale faculty assignment %s; falling back to candidate faculty pool",
                    course.code,
                    course.faculty_id,
                )
            requires_faculty = bool(getattr(course, "assign_faculty", True))
            requires_room = bool(getattr(course, "assign_classroom", True))
            if requires_faculty:
                faculty_candidate_ids = self._faculty_candidates_for_course(course)
                primary_faculty_id = (
                    course.faculty_id if course.faculty_id and course.faculty_id in self.faculty else faculty_candidate_ids[0]
                )
                preferred_faculty_ids = tuple(
                    item_id for item_id in faculty_candidate_ids if self._faculty_prefers_subject(item_id, course.code)
                )
            else:
                faculty_candidate_ids = []
                primary_faculty_id = ""
                preferred_faculty_ids = tuple()

            max_daily_slots = max((len(slots) for slots in self.day_slots.values()), default=0)
            request_templates: list[tuple[Literal["theory", "tutorial", "lab"], int, int]] = []
            is_project_phase = self._is_project_phase_course(course)
            if course.theory_hours > 0:
                request_templates.append(("theory", 1, int(course.theory_hours)))
            if course.tutorial_hours > 0:
                request_templates.append(("tutorial", 1, int(course.tutorial_hours)))

            practical_hours = max(0, int(course.lab_hours or 0))
            if practical_hours > 0:
                preferred_practical_block = self._practical_block_slot_size_for_course(course)
                full_blocks, remainder = divmod(practical_hours, preferred_practical_block)
                if full_blocks > 0:
                    request_templates.append(("lab", preferred_practical_block, full_blocks))
                if remainder > 0:
                    # Preserve exact weekly load even when practical hours are not a multiple of the preferred block size.
                    request_templates.append(("lab", remainder, 1))
                    logger.warning(
                        "Practical block remainder for course %s: %s practical hour(s) mapped as one contiguous block",
                        course.code,
                        remainder,
                    )

            if not request_templates:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Course {course.code} has no valid hours to schedule.",
                )

            for section in self.sections:
                for session_type, block_size, blocks_needed in request_templates:
                    session_is_lab = session_type == "lab"
                    batch_segregation_enabled = bool(getattr(course, "batch_segregation", True))
                    if session_is_lab:
                        if batch_segregation_enabled:
                            batch_count = max(1, int(program_course.lab_batch_count or 1))
                            student_per_batch = max(1, math.ceil(section.capacity / batch_count))
                            batch_labels = [f"B{index + 1}" for index in range(batch_count)]
                        else:
                            student_per_batch = max(1, section.capacity)
                            batch_labels = [None]
                    else:
                        student_per_batch = max(1, section.capacity)
                        batch_labels = [None]

                    if requires_room:
                        room_candidates = self._room_candidates_for(course, session_type=session_type)
                        request_room_candidates = self._select_room_candidates_for_request(
                            room_candidates=room_candidates,
                            student_count=student_per_batch,
                            is_lab=session_is_lab,
                            seed=f"{course.code}|{section.name}|{session_type}",
                        )
                    else:
                        room_candidates = []
                        request_room_candidates = []

                    for batch in batch_labels:
                        if not requires_faculty:
                            faculty_resource_key = f"{course.id}|{section.name}|{session_type}|{batch or 'all'}"
                            faculty_placeholder_id = self._ensure_virtual_faculty(
                                self._virtual_resource_id(kind="faculty", key=faculty_resource_key)
                            )
                            faculty_candidate_ids = [faculty_placeholder_id]
                            primary_faculty_id = faculty_placeholder_id
                            preferred_faculty_ids = tuple()
                        if not requires_room:
                            room_resource_key = f"{course.id}|{section.name}|{session_type}|{batch or 'all'}"
                            room_placeholder_id = self._ensure_virtual_room(
                                self._virtual_resource_id(kind="room", key=room_resource_key),
                                session_type=session_type,
                            )
                            request_room_candidates = [self.rooms[room_placeholder_id]]

                        if block_size > max_daily_slots:
                            raise HTTPException(
                                status_code=status.HTTP_400_BAD_REQUEST,
                                detail=(
                                    f"No feasible placement options for course {course.code}: required contiguous block size "
                                    f"({block_size}) exceeds available daily slots ({max_daily_slots}). "
                                    "Adjust working-hour windows."
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
                                            if requires_room and enforce_room_windows and not self._room_is_available(room, day, block_start, block_end):
                                                continue
                                            for faculty_id in faculty_option_order:
                                                if requires_faculty:
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
                                                    check_room=requires_room,
                                                    check_faculty=requires_faculty,
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
                                    is_lab=session_type == "lab",
                                    session_type=session_type,
                                    allow_parallel_batches=(
                                        session_type == "lab"
                                        and batch_segregation_enabled
                                        and (program_course.allow_parallel_batches or is_project_phase)
                                    ),
                                    room_candidate_ids=tuple(room.id for room in request_room_candidates),
                                    options=tuple(options),
                                    requires_faculty=requires_faculty,
                                    requires_room=requires_room,
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
            if course is not None and not bool(getattr(course, "assign_faculty", True)):
                requirements[course_id] = False
                continue
            if course is None or not course.faculty_id or course.faculty_id not in self.faculty:
                requirements[course_id] = False
                continue

            dedicated_faculty = self.faculty[course.faculty_id]
            dedicated_capacity_minutes = self._effective_faculty_max_hours(dedicated_faculty) * 60
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
        if not bool(getattr(course, "assign_faculty", True)):
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
            if req.is_lab or not self._request_requires_faculty(req) or not self._single_faculty_required(req.course_id):
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
                if req.is_lab or not self._request_requires_faculty(req) or req.request_id in self.fixed_genes:
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
                if not req.is_lab and self._request_requires_faculty(req):
                    selected_faculty_id = req.options[fixed_option_index].faculty_id
                    chosen_faculty_by_course_section[(req.course_id, req.section)] = selected_faculty_id
                    if self._single_faculty_required(req.course_id):
                        chosen_faculty_by_course.setdefault(req.course_id, selected_faculty_id)
            else:
                if not req.is_lab and self._request_requires_faculty(req):
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
                        common_faculty_ids = getattr(
                            self,
                            "common_faculty_candidates_by_course",
                            {},
                        ).get(req.course_id, ())
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
            requires_room = self._request_requires_room(req)
            requires_faculty = self._request_requires_faculty(req)
            room = self.rooms[option.room_id] if requires_room else None
            faculty = self.faculty[option.faculty_id] if requires_faculty else None

            block_start, block_end = self._option_bounds(option, req.block_size)

            if not self._within_semester_time_window(block_start, block_end):
                conflicted.add(req_index)

            if self._conflicts_reserved_resources(
                day=option.day,
                start_min=block_start,
                end_min=block_end,
                room_id=option.room_id,
                faculty_id=option.faculty_id,
                check_room=requires_room,
                check_faculty=requires_faculty,
            ):
                conflicted.add(req_index)

            if requires_room and room is not None:
                if room.capacity < req.student_count:
                    conflicted.add(req_index)
                if req.is_lab and room.type != RoomType.lab:
                    conflicted.add(req_index)
                if not req.is_lab and room.type == RoomType.lab:
                    conflicted.add(req_index)

            if requires_faculty and faculty is not None and not self._faculty_allows_day(faculty, option.day):
                conflicted.add(req_index)

            if requires_faculty and self.faculty_windows.get(option.faculty_id, {}).get(option.day):
                if not any(
                    start <= block_start and block_end <= end
                    for start, end in self.faculty_windows[option.faculty_id][option.day]
                ):
                    conflicted.add(req_index)

            if requires_room and self.room_windows.get(option.room_id, {}).get(option.day):
                if not any(
                    start <= block_start and block_end <= end
                    for start, end in self.room_windows[option.room_id][option.day]
                ):
                    conflicted.add(req_index)

            section_day_req_ids.setdefault((req.section, option.day), set()).add(req_index)
            section_req_ids.setdefault(req.section, set()).add(req_index)
            if requires_faculty:
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
                if requires_room:
                    room_occ.setdefault(room_key, []).append(req_index)
                if requires_faculty:
                    faculty_occ.setdefault(faculty_key, []).append(req_index)
                section_occ.setdefault(section_key, []).append(req_index)
                elective_occ.setdefault((option.day, slot_idx), []).append(req_index)
                section_day_slots.setdefault((req.section, option.day), set()).add(slot_idx)
                if requires_faculty:
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
                    if not (self._request_requires_room(left_req) and self._request_requires_room(right_req)):
                        continue
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
                    if not (self._request_requires_faculty(left_req) and self._request_requires_faculty(right_req)):
                        continue
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

        parallel_lab_signatures: dict[tuple[str, str, str, int], dict[str, list[tuple[str, int, int]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        parallel_lab_req_ids: dict[tuple[str, str, str, int], dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        for req_index, req in enumerate(self.block_requests):
            if not req.is_lab or not req.allow_parallel_batches or not req.batch:
                continue
            option = selected_options[req_index]
            group_key = self._parallel_lab_group_key(req)
            if group_key is None:
                continue
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
            lecture_req_indices = [
                idx for idx in req_indices if self._request_requires_faculty(self.block_requests[idx])
            ]
            if len(lecture_req_indices) <= 1:
                continue
            assigned_faculty_ids = {selected_options[idx].faculty_id for idx in lecture_req_indices}
            if len(assigned_faculty_ids) > 1:
                conflicted.update(lecture_req_indices)

        for course_id, req_indices in self.request_indices_by_course.items():
            if not self._single_faculty_required(course_id):
                continue
            lecture_req_indices = [
                idx
                for idx in req_indices
                if not self.block_requests[idx].is_lab and self._request_requires_faculty(self.block_requests[idx])
            ]
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

            enforce_student_credit_load = True
            if self.program_constraint is not None:
                enforce_student_credit_load = bool(self.program_constraint.enforce_student_credit_load)
            if enforce_student_credit_load and self.expected_section_minutes > 0:
                for section, request_ids in section_req_ids.items():
                    minutes = weekly_minutes_by_section.get(section, 0)
                    if minutes != self.expected_section_minutes:
                        conflicted.update(request_ids)

        # Workload caps are hard constraints for publishable schedules.
        virtual_faculty_ids = getattr(self, "_virtual_faculty_ids", set())
        for faculty_id, minutes in faculty_minutes.items():
            if faculty_id in virtual_faculty_ids:
                continue
            faculty = self.faculty.get(faculty_id)
            if faculty is None:
                continue
            max_minutes = self._effective_faculty_max_hours(faculty) * 60
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
                if self._request_requires_room(req):
                    room_occ[(option.day, slot_idx, option.room_id)].append(req_index)
                if self._request_requires_faculty(req):
                    faculty_occ[(option.day, slot_idx, option.faculty_id)].append(req_index)
                section_occ[(option.day, slot_idx, req.section)].append(req_index)

        for values in room_occ.values():
            if len(values) <= 1:
                continue
            for left_index, left_req_idx in enumerate(values):
                for right_req_idx in values[left_index + 1 :]:
                    left_req = self.block_requests[left_req_idx]
                    right_req = self.block_requests[right_req_idx]
                    if not (self._request_requires_room(left_req) and self._request_requires_room(right_req)):
                        continue
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
                    if not (self._request_requires_faculty(left_req) and self._request_requires_faculty(right_req)):
                        continue
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
                if not req.is_lab and self._request_requires_faculty(req):
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
            requires_room = self._request_requires_room(req)
            requires_faculty = self._request_requires_faculty(req)
            for offset in range(req.block_size):
                slot_idx = option.start_index + offset
                if requires_room:
                    room_occ[(option.day, slot_idx, option.room_id)].append(req_index)
                if requires_faculty:
                    faculty_occ[(option.day, slot_idx, option.faculty_id)].append(req_index)
                section_occ[(option.day, slot_idx, req.section)].append(req_index)

        def unregister(req_index: int, option: PlacementOption) -> None:
            req = self.block_requests[req_index]
            requires_room = self._request_requires_room(req)
            requires_faculty = self._request_requires_faculty(req)
            for offset in range(req.block_size):
                slot_idx = option.start_index + offset
                room_key = (option.day, slot_idx, option.room_id)
                faculty_key = (option.day, slot_idx, option.faculty_id)
                section_key = (option.day, slot_idx, req.section)
                if requires_room and req_index in room_occ.get(room_key, []):
                    room_occ[room_key].remove(req_index)
                    if not room_occ[room_key]:
                        room_occ.pop(room_key, None)
                if requires_faculty and req_index in faculty_occ.get(faculty_key, []):
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
            requires_room = self._request_requires_room(req)
            requires_faculty = self._request_requires_faculty(req)
            room_hits = 0
            faculty_hits = 0
            section_hits = 0
            for offset in range(req.block_size):
                slot_idx = option.start_index + offset
                room_key = (option.day, slot_idx, option.room_id)
                faculty_key = (option.day, slot_idx, option.faculty_id)
                section_key = (option.day, slot_idx, req.section)
                if requires_room:
                    for other_idx in room_occ.get(room_key, []):
                        if other_idx == req_index:
                            continue
                        other_req = self.block_requests[other_idx]
                        if not self._request_requires_room(other_req):
                            continue
                        if self._is_allowed_shared_overlap(req, other_req, option, selected_options[other_idx]):
                            continue
                        room_hits += 1
                if requires_faculty:
                    for other_idx in faculty_occ.get(faculty_key, []):
                        if other_idx == req_index:
                            continue
                        other_req = self.block_requests[other_idx]
                        if not self._request_requires_faculty(other_req):
                            continue
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
                        if not (self._request_requires_room(left_req) and self._request_requires_room(right_req)):
                            continue
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
                        if not (self._request_requires_faculty(left_req) and self._request_requires_faculty(right_req)):
                            continue
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
                if not req.is_lab and self._request_requires_faculty(req):
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
                if not self._request_requires_room(req):
                    continue
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
                if not self._request_requires_room(req):
                    continue
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
        selected_assignments: list[tuple[int, BlockRequest, PlacementOption]] = []
        faculty_schedule: dict[str, list[tuple[str, int, int]]] = defaultdict(list)

        for req_index, req in enumerate(self.block_requests):
            option = req.options[genes[req_index]]
            selected_options[req_index] = option
            selected_assignments.append((req_index, req, option))
            requires_room = self._request_requires_room(req)
            requires_faculty = self._request_requires_faculty(req)
            room = self.rooms[option.room_id] if requires_room else None
            faculty = self.faculty[option.faculty_id] if requires_faculty else None
            block_start, block_end = self._option_bounds(option, req.block_size)
            if requires_faculty:
                faculty_schedule[option.faculty_id].append((option.day, block_start, block_end))

            if not self._within_semester_time_window(block_start, block_end):
                hard += weights.semester_limit
            if self._overlaps_non_teaching_window(day=option.day, start_min=block_start, end_min=block_end):
                hard += weights.semester_limit * 4

            reserved_room_conflict, reserved_faculty_conflict = self._reserved_conflict_flags(
                day=option.day,
                start_min=block_start,
                end_min=block_end,
                room_id=option.room_id,
                faculty_id=option.faculty_id,
                check_room=requires_room,
                check_faculty=requires_faculty,
            )
            if requires_room and reserved_room_conflict:
                hard += weights.room_conflict
            if requires_faculty and reserved_faculty_conflict:
                hard += weights.faculty_conflict

            if requires_room and room is not None:
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
                if requires_room:
                    room_occ.setdefault(room_key, []).append(req_index)
                if requires_faculty:
                    faculty_occ.setdefault(faculty_key, []).append(req_index)
                section_occ.setdefault(section_key, []).append(req_index)
                elective_occ.setdefault((option.day, slot_idx), []).append(req_index)
                section_day_slots.setdefault((req.section, option.day), set()).add(slot_idx)
                if requires_faculty:
                    faculty_minutes[option.faculty_id] = (
                        faculty_minutes.get(option.faculty_id, 0) + self.schedule_policy.period_minutes
                    )
            if requires_faculty:
                faculty_day_req_indices.setdefault((option.faculty_id, option.day), []).append(req_index)
            if self._is_elective_request(req):
                elective_signatures_by_section[req.section].append(
                    (option.day, option.start_index, req.block_size, req.session_type)
                )

            if requires_faculty and faculty is not None and not self._faculty_allows_day(faculty, option.day):
                hard += weights.faculty_availability

            if requires_faculty and self.faculty_windows.get(option.faculty_id, {}).get(option.day):
                if not any(
                    start <= block_start and block_end <= end
                    for start, end in self.faculty_windows[option.faculty_id][option.day]
                ):
                    hard += weights.faculty_availability

            if requires_room and self.room_windows.get(option.room_id, {}).get(option.day):
                if not any(
                    start <= block_start and block_end <= end
                    for start, end in self.room_windows[option.room_id][option.day]
                ):
                    hard += weights.room_type

            if requires_faculty:
                if req.preferred_faculty_ids and option.faculty_id not in req.preferred_faculty_ids:
                    soft += (weights.faculty_subject_preference * 0.20) * req.block_size
                if req.primary_faculty_id and option.faculty_id != req.primary_faculty_id:
                    soft += (weights.faculty_subject_preference * 0.08) * req.block_size

        assisting_faculty_by_request, missing_assistant_slots = self._assign_assisting_faculty(
            selected_assignments=selected_assignments,
            faculty_schedule=faculty_schedule,
            required_count=2,
            allow_relaxed_fallback=False,
        )
        if missing_assistant_slots > 0:
            hard += weights.faculty_conflict * missing_assistant_slots

        for req_index, req, option in selected_assignments:
            assistant_faculty_ids = assisting_faculty_by_request.get(req_index, tuple())
            if not assistant_faculty_ids:
                continue

            block_start, block_end = self._option_bounds(option, req.block_size)
            for assistant_faculty_id in assistant_faculty_ids:
                assistant = self.faculty.get(assistant_faculty_id)
                if assistant is not None and not self._faculty_allows_day(assistant, option.day):
                    hard += weights.faculty_availability
                if self.faculty_windows.get(assistant_faculty_id, {}).get(option.day):
                    if not any(
                        start <= block_start and block_end <= end
                        for start, end in self.faculty_windows[assistant_faculty_id][option.day]
                    ):
                        hard += weights.faculty_availability

                _room_reserved, assistant_reserved = self._reserved_conflict_flags(
                    day=option.day,
                    start_min=block_start,
                    end_min=block_end,
                    room_id=option.room_id,
                    faculty_id=assistant_faculty_id,
                    check_room=False,
                    check_faculty=True,
                )
                if assistant_reserved:
                    hard += weights.faculty_conflict

                faculty_day_req_indices.setdefault((assistant_faculty_id, option.day), []).append(req_index)
                for offset in range(req.block_size):
                    slot_idx = option.start_index + offset
                    faculty_key = (option.day, slot_idx, assistant_faculty_id)
                    faculty_occ.setdefault(faculty_key, []).append(req_index)
                    faculty_minutes[assistant_faculty_id] = (
                        faculty_minutes.get(assistant_faculty_id, 0) + self.schedule_policy.period_minutes
                    )

        for values in room_occ.values():
            if len(values) <= 1:
                continue
            for left_index, left_req_idx in enumerate(values):
                for right_req_idx in values[left_index + 1 :]:
                    left_req = self.block_requests[left_req_idx]
                    right_req = self.block_requests[right_req_idx]
                    if not (self._request_requires_room(left_req) and self._request_requires_room(right_req)):
                        continue
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
                    if not (self._request_requires_faculty(left_req) and self._request_requires_faculty(right_req)):
                        continue
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

        parallel_lab_signatures: dict[tuple[str, str, str, int], dict[str, list[tuple[str, int, int]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for req_index, req in enumerate(self.block_requests):
            if not req.is_lab or not req.allow_parallel_batches or not req.batch:
                continue
            option = selected_options[req_index]
            group_key = self._parallel_lab_group_key(req)
            if group_key is None:
                continue
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
            applicable_req_indices = [
                idx for idx in lecture_req_indices if self._request_requires_faculty(self.block_requests[idx])
            ]
            if len(applicable_req_indices) <= 1:
                continue
            assigned_faculty_ids = [selected_options[idx].faculty_id for idx in applicable_req_indices]
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
            lecture_req_indices = [
                idx
                for idx in req_indices
                if not self.block_requests[idx].is_lab and self._request_requires_faculty(self.block_requests[idx])
            ]
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

            enforce_student_credit_load = True
            if self.program_constraint is not None:
                enforce_student_credit_load = bool(self.program_constraint.enforce_student_credit_load)
            if enforce_student_credit_load and self.expected_section_minutes > 0:
                period_minutes = max(1, self.schedule_policy.period_minutes)
                all_sections = {req.section for req in self.block_requests}
                for section in all_sections:
                    minutes = weekly_minutes_by_section.get(section, 0)
                    if minutes == self.expected_section_minutes:
                        continue
                    delta = abs(minutes - self.expected_section_minutes)
                    hard += weights.semester_limit * max(1, math.ceil(delta / period_minutes))

        virtual_faculty_ids = getattr(self, "_virtual_faculty_ids", set())
        for faculty_id, faculty in self.faculty.items():
            if faculty_id in virtual_faculty_ids:
                continue
            minutes = faculty_minutes.get(faculty_id, 0)
            max_minutes = self._effective_faculty_max_hours(faculty) * 60
            if minutes > max_minutes:
                overflow_periods = max(1, (minutes - max_minutes) // max(1, self.schedule_policy.period_minutes))
                hard += weights.workload_overflow * overflow_periods
            target_hours = self._faculty_min_target_hours(faculty)
            if target_hours > 0:
                baseline_minutes = self.faculty_three_term_baseline_minutes.get(faculty_id, 0)
                horizon_terms = max(1, len(self.three_term_horizon_terms))
                target_minutes = target_hours * 60 * horizon_terms
                combined_minutes = baseline_minutes + minutes
                if combined_minutes < target_minutes:
                    deficit_periods = max(
                        1,
                        math.ceil((target_minutes - combined_minutes) / max(1, self.schedule_policy.period_minutes)),
                    )
                    underflow_multiplier = 1.0
                    if self.program_constraint is not None and self.program_constraint.auto_assign_research_slots:
                        # Research-hour backfilling relaxes the penalty while still preferring balanced teaching load.
                        underflow_multiplier = 0.35
                    soft += (weights.workload_underflow * deficit_periods) * underflow_multiplier

        workload_balance_penalty = self._three_term_workload_balance_penalty(faculty_minutes)
        soft += workload_balance_penalty * max(1.0, weights.spread_balance)

        sections = {req.section for req in self.block_requests}
        for section in sections:
            day_counts = [len(section_day_slots.get((section, day), set())) for day in self.day_slots]
            if day_counts:
                soft += (max(day_counts) - min(day_counts)) * weights.spread_balance

        objectives = (
            float(hard),
            float(soft),
            float(workload_balance_penalty),
        )
        # Display-only scalar for API/UI compatibility; MOEA decisions use objective vectors.
        fitness = -(objectives[0] + objectives[1] + objectives[2])
        result = EvaluationResult(
            fitness=fitness,
            hard_conflicts=hard,
            soft_penalty=soft,
            workload_balance_penalty=workload_balance_penalty,
            objectives=objectives,
        )
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
                    (
                        -float(self._effective_faculty_max_hours(self.faculty.get(faculty_id)))
                        if faculty_id in self.faculty
                        else 0.0
                    ),
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
            applicable_indices = [
                req_index
                for req_index in req_indices
                if self._request_requires_faculty(self.block_requests[req_index])
            ]
            if len(applicable_indices) <= 1:
                continue

            fixed_faculty_id: str | None = None
            conflicting_fixed = False
            for req_index in applicable_indices:
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
                candidate_ids = {option.faculty_id for option in self.block_requests[applicable_indices[0]].options}
                for req_index in applicable_indices[1:]:
                    candidate_ids &= {option.faculty_id for option in self.block_requests[req_index].options}
            target_faculty_id = choose_target_faculty(
                course_id=course_id,
                req_indices=applicable_indices,
                candidate_ids=candidate_ids,
                fixed_faculty_id=fixed_faculty_id,
            )
            if target_faculty_id is None:
                continue
            align_group(course_id, applicable_indices, target_faculty_id)

        for course_id, required in single_faculty_required.items():
            if not required:
                continue
            lecture_indices = [
                req_index
                for req_index in self.request_indices_by_course.get(course_id, [])
                if not self.block_requests[req_index].is_lab
                and self._request_requires_faculty(self.block_requests[req_index])
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
        selected_assignments: list[tuple[int, BlockRequest, PlacementOption]] = []
        faculty_schedule: dict[str, list[tuple[str, int, int]]] = defaultdict(list)

        for req_index, req in enumerate(self.block_requests):
            option = req.options[genes[req_index]]
            selected_assignments.append((req_index, req, option))
            used_faculty_ids.add(option.faculty_id)
            used_course_ids.add(req.course_id)
            used_room_ids.add(option.room_id)
            selected_faculty_by_course[req.course_id].append(option.faculty_id)
            block_start, block_end = self._option_bounds(option, req.block_size)
            faculty_schedule[option.faculty_id].append((option.day, block_start, block_end))

        assisting_faculty_by_request, _missing_assistant_slots = self._assign_assisting_faculty(
            selected_assignments=selected_assignments,
            faculty_schedule=faculty_schedule,
            required_count=2,
            allow_relaxed_fallback=False,
        )
        for assistant_ids in assisting_faculty_by_request.values():
            for assistant_faculty_id in assistant_ids:
                used_faculty_ids.add(assistant_faculty_id)

        faculty_minutes = self._faculty_minutes_from_schedule(faculty_schedule)
        research_rows, research_courses, research_rooms = self._build_research_slot_payloads(
            faculty_schedule=faculty_schedule,
            faculty_minutes=faculty_minutes,
        )

        for req_index, req, option in selected_assignments:
            assistant_faculty_ids = list(assisting_faculty_by_request.get(req_index, tuple()))
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
                        "assistantFacultyIds": assistant_faculty_ids,
                    }
                )
        for row in research_rows:
            timetable_rows.append(row)
            used_faculty_ids.add(row["facultyId"])
            used_course_ids.add(row["courseId"])
            used_room_ids.add(row["roomId"])

        faculty_data = []
        virtual_faculty_ids = getattr(self, "_virtual_faculty_ids", set())
        for item in self.faculty.values():
            if item.id not in used_faculty_ids and item.id in virtual_faculty_ids:
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
            if not item.assign_faculty:
                resolved_faculty_id = next(iter(assigned_ids), "")
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
                    "batchSegregation": item.batch_segregation,
                    "practicalContiguousSlots": item.practical_contiguous_slots,
                    "assignFaculty": item.assign_faculty,
                    "assignClassroom": item.assign_classroom,
                    "defaultRoomId": item.default_room_id,
                    "electiveCategory": item.elective_category,
                }
            )
        course_data.extend(research_courses)

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
        room_data.extend(research_rooms)

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

    def _report_progress(
        self,
        *,
        stage: str,
        message: str,
        level: str = "info",
        progress_percent: float | None = None,
        metrics: dict[str, Any] | None = None,
        best_genes: list[int] | None = None,
        best_evaluation: EvaluationResult | None = None,
    ) -> None:
        reporter = getattr(self, "progress_reporter", None)
        if reporter is None:
            return
        payload: dict[str, Any] = {
            "stage": stage,
            "level": level,
            "message": message,
        }
        if progress_percent is not None:
            payload["progress_percent"] = float(max(0.0, min(100.0, progress_percent)))
        if metrics:
            payload["metrics"] = metrics
        if best_genes is not None and best_evaluation is not None:
            preview_payload = self._decode_payload(best_genes)
            payload["latest_generation"] = GenerateTimetableResponse(
                alternatives=[
                    GeneratedAlternative(
                        rank=1,
                        fitness=best_evaluation.fitness,
                        hard_conflicts=best_evaluation.hard_conflicts,
                        soft_penalty=best_evaluation.soft_penalty,
                        payload=preview_payload,
                    )
                ],
                settings_used=self.settings,
                runtime_ms=0,
            ).model_dump(by_alias=True)
        try:
            reporter(payload)
        except Exception:
            logger.exception("Failed to publish scheduler progress event")

    def _scenario_fingerprint(self) -> str:
        request_signature = [
            (
                req.course_id,
                req.section,
                req.batch or "",
                req.student_count,
                req.block_size,
                req.is_lab,
                req.session_type,
                req.requires_faculty,
                req.requires_room,
                len(req.options),
                len(req.room_candidate_ids),
            )
            for req in self.block_requests
        ]
        request_signature.sort()
        canonical = {
            "program_id": self.program_id,
            "term_number": self.term_number,
            "expected_section_minutes": self.expected_section_minutes,
            "request_count": len(self.block_requests),
            "requests": request_signature,
            "slot_count_per_day": {day: len(slots) for day, slots in sorted(self.day_slots.items())},
        }
        digest = hashlib.blake2b(
            json.dumps(canonical, sort_keys=True).encode("utf-8"),
            digest_size=16,
        ).hexdigest()
        return digest

    def _hyperparameter_profile_key(self) -> str:
        scope = (getattr(self, "hyperparameter_cache_scope", "") or "single").strip().lower()
        program_id = getattr(self, "program_id", "") or "global"
        if scope in {"cycle", "term", "odd_even"}:
            return f"profile:cycle:{program_id}"
        if scope in {"single", "semester"}:
            return f"profile:single:{program_id}"
        return f"profile:{scope}:{program_id}"

    def _hyperparameter_cache_lookup_keys(self) -> list[tuple[str, str]]:
        """
        Returns candidate keys in priority order.

        Order:
        1) Scenario profile key (`profile:cycle:*` or `profile:single:*`)
        2) Same-program opposite profile as migration fallback
        3) Legacy scenario fingerprint key (pre-profile cache format)
        """
        primary = self._hyperparameter_profile_key()
        program_id = getattr(self, "program_id", "") or "global"

        candidates: list[tuple[str, str]] = [(primary, "profile")]
        if primary.startswith("profile:cycle:"):
            candidates.append((f"profile:single:{program_id}", "profile_fallback_single"))
        elif primary.startswith("profile:single:"):
            candidates.append((f"profile:cycle:{program_id}", "profile_fallback_cycle"))

        legacy = self._scenario_fingerprint()
        candidates.append((legacy, "legacy_scenario"))

        seen: set[str] = set()
        deduped: list[tuple[str, str]] = []
        for key, source in candidates:
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append((key, source))
        return deduped

    @classmethod
    def _ensure_hyperparameter_cache_loaded(cls) -> None:
        if cls._hyperparameter_cache_loaded:
            return
        with cls._hyperparameter_cache_lock:
            if cls._hyperparameter_cache_loaded:
                return
            cls._hyperparameter_cache_by_scenario = {}
            if HYPERPARAMETER_CACHE_FILE.exists():
                try:
                    raw = json.loads(HYPERPARAMETER_CACHE_FILE.read_text(encoding="utf-8"))
                    if isinstance(raw, dict):
                        for scenario_key, values in raw.items():
                            if not isinstance(values, dict):
                                continue
                            try:
                                cls._hyperparameter_cache_by_scenario[scenario_key] = SearchHyperParameters(
                                    population_size=int(values["population_size"]),
                                    generations=int(values["generations"]),
                                    mutation_rate=float(values["mutation_rate"]),
                                    crossover_rate=float(values["crossover_rate"]),
                                    elite_count=int(values["elite_count"]),
                                    tournament_size=int(values["tournament_size"]),
                                    stagnation_limit=int(values["stagnation_limit"]),
                                    annealing_iterations=int(values["annealing_iterations"]),
                                    annealing_initial_temperature=float(values["annealing_initial_temperature"]),
                                    annealing_cooling_rate=float(values["annealing_cooling_rate"]),
                                )
                            except (KeyError, TypeError, ValueError):
                                continue
                except Exception:
                    logger.exception("Failed to load GA hyperparameter cache file")
            cls._hyperparameter_cache_loaded = True

    def _load_cached_hyperparameters_for_scenario(
        self,
    ) -> tuple[SearchHyperParameters | None, str | None, str | None]:
        self._ensure_hyperparameter_cache_loaded()
        with self._hyperparameter_cache_lock:
            for scenario_key, source in self._hyperparameter_cache_lookup_keys():
                cached = self._hyperparameter_cache_by_scenario.get(scenario_key)
                if cached is None:
                    continue
                return SearchHyperParameters(**cached.__dict__), scenario_key, source
        return None, None, None

    def _persist_cached_hyperparameters_for_scenario(
        self,
        params: SearchHyperParameters,
        *,
        cache_key: str | None = None,
    ) -> None:
        self._ensure_hyperparameter_cache_loaded()
        scenario_key = cache_key or self._hyperparameter_profile_key()
        with self._hyperparameter_cache_lock:
            self._hyperparameter_cache_by_scenario[scenario_key] = SearchHyperParameters(**params.__dict__)
            serialized = {
                key: value.__dict__
                for key, value in self._hyperparameter_cache_by_scenario.items()
            }
            try:
                HYPERPARAMETER_CACHE_FILE.write_text(
                    json.dumps(serialized, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
            except Exception:
                logger.exception("Failed to persist GA hyperparameter cache")

    def _current_hyperparameters(self) -> SearchHyperParameters:
        return SearchHyperParameters(
            population_size=self.settings.population_size,
            generations=self.settings.generations,
            mutation_rate=self.settings.mutation_rate,
            crossover_rate=self.settings.crossover_rate,
            elite_count=self.settings.elite_count,
            tournament_size=self.settings.tournament_size,
            stagnation_limit=self.settings.stagnation_limit,
            annealing_iterations=self.settings.annealing_iterations,
            annealing_initial_temperature=self.settings.annealing_initial_temperature,
            annealing_cooling_rate=self.settings.annealing_cooling_rate,
        )

    def _apply_hyperparameters(self, params: SearchHyperParameters) -> None:
        self.settings.population_size = params.population_size
        self.settings.generations = params.generations
        self.settings.mutation_rate = params.mutation_rate
        self.settings.crossover_rate = params.crossover_rate
        self.settings.elite_count = params.elite_count
        self.settings.tournament_size = params.tournament_size
        self.settings.stagnation_limit = params.stagnation_limit
        self.settings.annealing_iterations = params.annealing_iterations
        self.settings.annealing_initial_temperature = params.annealing_initial_temperature
        self.settings.annealing_cooling_rate = params.annealing_cooling_rate

    def _hyperparameter_bounds(self, block_count: int) -> dict[str, tuple[float, float]]:
        if block_count >= 220:
            return {
                "population_size": (36, 72),
                "generations": (20, 50),
                "annealing_iterations": (120, 260),
            }
        if block_count >= 160:
            return {
                "population_size": (44, 86),
                "generations": (24, 64),
                "annealing_iterations": (140, 320),
            }
        if block_count >= 120:
            return {
                "population_size": (52, 102),
                "generations": (30, 82),
                "annealing_iterations": (160, 380),
            }
        if block_count >= 80:
            return {
                "population_size": (58, 118),
                "generations": (34, 96),
                "annealing_iterations": (180, 460),
            }
        return {
            "population_size": (66, 138),
            "generations": (40, 126),
            "annealing_iterations": (220, 620),
        }

    def _coerce_hyperparameters(
        self,
        params: SearchHyperParameters,
        *,
        block_count: int,
    ) -> SearchHyperParameters:
        bounds = self._hyperparameter_bounds(block_count)

        def clamp_int(value: int, low: float, high: float) -> int:
            return int(max(int(low), min(int(high), int(round(value)))))

        def clamp_float(value: float, low: float, high: float) -> float:
            return max(low, min(high, float(value)))

        population_size = clamp_int(params.population_size, *bounds["population_size"])
        generations = clamp_int(params.generations, *bounds["generations"])
        mutation_rate = clamp_float(params.mutation_rate, 0.04, 0.35)
        crossover_rate = clamp_float(params.crossover_rate, 0.55, 0.98)

        elite_upper = max(2, min(18, population_size // 3))
        elite_count = clamp_int(params.elite_count, 2, elite_upper)
        tournament_upper = max(2, min(16, population_size // 2))
        tournament_size = clamp_int(params.tournament_size, 2, tournament_upper)
        stagnation_upper = max(12, generations)
        stagnation_limit = clamp_int(params.stagnation_limit, 8, stagnation_upper)
        annealing_iterations = clamp_int(params.annealing_iterations, *bounds["annealing_iterations"])
        annealing_initial_temperature = clamp_float(params.annealing_initial_temperature, 2.0, 14.0)
        annealing_cooling_rate = clamp_float(params.annealing_cooling_rate, 0.90, 0.9998)

        if elite_count >= population_size:
            elite_count = max(1, population_size - 1)
        if tournament_size > population_size:
            tournament_size = population_size

        return SearchHyperParameters(
            population_size=population_size,
            generations=generations,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            elite_count=elite_count,
            tournament_size=tournament_size,
            stagnation_limit=stagnation_limit,
            annealing_iterations=annealing_iterations,
            annealing_initial_temperature=annealing_initial_temperature,
            annealing_cooling_rate=annealing_cooling_rate,
        )

    def _random_hyperparameters(self, *, block_count: int) -> SearchHyperParameters:
        bounds = self._hyperparameter_bounds(block_count)
        params = SearchHyperParameters(
            population_size=self.random.randint(int(bounds["population_size"][0]), int(bounds["population_size"][1])),
            generations=self.random.randint(int(bounds["generations"][0]), int(bounds["generations"][1])),
            mutation_rate=self.random.uniform(0.05, 0.30),
            crossover_rate=self.random.uniform(0.62, 0.95),
            elite_count=self.random.randint(2, 12),
            tournament_size=self.random.randint(2, 10),
            stagnation_limit=self.random.randint(10, 70),
            annealing_iterations=self.random.randint(
                int(bounds["annealing_iterations"][0]),
                int(bounds["annealing_iterations"][1]),
            ),
            annealing_initial_temperature=self.random.uniform(3.0, 10.0),
            annealing_cooling_rate=self.random.uniform(0.93, 0.9995),
        )
        return self._coerce_hyperparameters(params, block_count=block_count)

    def _crossover_hyperparameters(
        self,
        left: SearchHyperParameters,
        right: SearchHyperParameters,
        *,
        block_count: int,
    ) -> SearchHyperParameters:
        alpha = self.random.uniform(0.25, 0.75)
        child = SearchHyperParameters(
            population_size=int(round((left.population_size * alpha) + (right.population_size * (1.0 - alpha)))),
            generations=int(round((left.generations * alpha) + (right.generations * (1.0 - alpha)))),
            mutation_rate=(left.mutation_rate * alpha) + (right.mutation_rate * (1.0 - alpha)),
            crossover_rate=(left.crossover_rate * alpha) + (right.crossover_rate * (1.0 - alpha)),
            elite_count=int(round((left.elite_count * alpha) + (right.elite_count * (1.0 - alpha)))),
            tournament_size=int(round((left.tournament_size * alpha) + (right.tournament_size * (1.0 - alpha)))),
            stagnation_limit=int(round((left.stagnation_limit * alpha) + (right.stagnation_limit * (1.0 - alpha)))),
            annealing_iterations=int(
                round((left.annealing_iterations * alpha) + (right.annealing_iterations * (1.0 - alpha)))
            ),
            annealing_initial_temperature=(left.annealing_initial_temperature * alpha)
            + (right.annealing_initial_temperature * (1.0 - alpha)),
            annealing_cooling_rate=(left.annealing_cooling_rate * alpha)
            + (right.annealing_cooling_rate * (1.0 - alpha)),
        )
        return self._coerce_hyperparameters(child, block_count=block_count)

    def _mutate_hyperparameters(
        self,
        params: SearchHyperParameters,
        *,
        block_count: int,
        mutation_probability: float,
    ) -> SearchHyperParameters:
        candidate = SearchHyperParameters(**params.__dict__)
        bounds = self._hyperparameter_bounds(block_count)

        if self.random.random() < mutation_probability:
            candidate.population_size += self.random.randint(-18, 18)
        if self.random.random() < mutation_probability:
            candidate.generations += self.random.randint(-16, 16)
        if self.random.random() < mutation_probability:
            candidate.mutation_rate += self.random.uniform(-0.05, 0.05)
        if self.random.random() < mutation_probability:
            candidate.crossover_rate += self.random.uniform(-0.07, 0.07)
        if self.random.random() < mutation_probability:
            candidate.elite_count += self.random.randint(-3, 3)
        if self.random.random() < mutation_probability:
            candidate.tournament_size += self.random.randint(-2, 2)
        if self.random.random() < mutation_probability:
            candidate.stagnation_limit += self.random.randint(-12, 12)
        if self.random.random() < mutation_probability:
            candidate.annealing_iterations += self.random.randint(-80, 80)
        if self.random.random() < mutation_probability:
            candidate.annealing_initial_temperature += self.random.uniform(-1.2, 1.2)
        if self.random.random() < mutation_probability:
            candidate.annealing_cooling_rate += self.random.uniform(-0.02, 0.02)

        candidate.population_size = int(
            max(bounds["population_size"][0], min(bounds["population_size"][1], candidate.population_size))
        )
        candidate.generations = int(max(bounds["generations"][0], min(bounds["generations"][1], candidate.generations)))
        candidate.annealing_iterations = int(
            max(bounds["annealing_iterations"][0], min(bounds["annealing_iterations"][1], candidate.annealing_iterations))
        )
        return self._coerce_hyperparameters(candidate, block_count=block_count)

    @staticmethod
    def _hyperparameter_signature(params: SearchHyperParameters) -> tuple:
        return (
            params.population_size,
            params.generations,
            round(params.mutation_rate, 5),
            round(params.crossover_rate, 5),
            params.elite_count,
            params.tournament_size,
            params.stagnation_limit,
            params.annealing_iterations,
            round(params.annealing_initial_temperature, 3),
            round(params.annealing_cooling_rate, 5),
        )

    @staticmethod
    def _eval_sort_key(evaluation: EvaluationResult) -> tuple[float, float, float]:
        objective_1, objective_2, objective_3 = EvolutionaryScheduler._objective_values(evaluation)
        return (objective_1, objective_2, objective_3)

    def _probe_hyperparameters(self, request: GenerateTimetableRequest) -> EvaluationResult:
        probe_population_size = max(16, min(42, self.settings.population_size // 2))
        probe_generations = max(6, min(22, self.settings.generations // 3))
        probe_sa_iterations = max(8, min(48, self.settings.annealing_iterations // 12))

        population = self._build_initial_population()[:probe_population_size]
        while len(population) < probe_population_size:
            population.append(self._repair_individual(self._random_individual(), max_passes=1))

        evaluations = [self._evaluate(genes) for genes in population]
        best_eval = min(evaluations, key=self._eval_sort_key)
        stagnant = 0

        for generation in range(probe_generations):
            fronts, rank_by_index = self._non_dominated_sort(evaluations)
            crowding_by_index: dict[int, float] = {}
            for front in fronts:
                crowding_by_index.update(self._crowding_distances(front, evaluations))

            ordered_indices = sorted(
                range(len(population)),
                key=lambda index: (
                    rank_by_index.get(index, math.inf),
                    -crowding_by_index.get(index, 0.0),
                    *self._eval_sort_key(evaluations[index]),
                ),
            )
            if not ordered_indices:
                break

            generation_best = evaluations[ordered_indices[0]]
            if self._is_better_eval(generation_best, best_eval):
                best_eval = generation_best
                stagnant = 0
            else:
                stagnant += 1

            mutation_rate = self._adaptive_mutation_rate(stagnant)
            next_population: list[list[int]] = []
            elite_keep = min(len(ordered_indices), max(2, self.settings.elite_count))
            for index in ordered_indices[:elite_keep]:
                next_population.append(list(population[index]))

            top_index = ordered_indices[0]
            refined_genes, refined_eval = self._simulated_annealing_refine(
                population[top_index],
                evaluations[top_index],
                iterations=probe_sa_iterations,
            )
            if self._is_better_eval(refined_eval, best_eval):
                best_eval = refined_eval
            if len(next_population) < probe_population_size:
                next_population.append(refined_genes)

            while len(next_population) < probe_population_size:
                parent_a = self._select_moea_parent(
                    population,
                    evaluations,
                    rank_by_index,
                    crowding_by_index,
                )
                parent_b = self._select_moea_parent(
                    population,
                    evaluations,
                    rank_by_index,
                    crowding_by_index,
                )
                if self.random.random() < self.settings.crossover_rate:
                    child = self._crossover(parent_a, parent_b)
                else:
                    child = list(parent_a)

                if self.random.random() < 0.90:
                    child = self._mutate(child, mutation_rate=mutation_rate)

                if self.random.random() < 0.20:
                    child = self._repair_individual(child, max_passes=1)

                next_population.append(child)

            population = next_population
            evaluations = [self._evaluate(genes) for genes in population]
            candidate_best = min(evaluations, key=self._eval_sort_key)
            if self._is_better_eval(candidate_best, best_eval):
                best_eval = candidate_best

            if best_eval.hard_conflicts == 0 and generation >= max(3, probe_generations // 3):
                break

        return best_eval

    def _score_hyperparameters(self, request: GenerateTimetableRequest) -> tuple[float, float, float]:
        probe = self._probe_hyperparameters(request)
        runtime_budget = float(self.settings.population_size * self.settings.generations)
        obj_hard, obj_soft, obj_workload = self._objective_values(probe)
        return (
            obj_hard,
            obj_soft + obj_workload,
            runtime_budget * 0.001,
        )

    def _tuning_wall_time_limit_seconds(self, block_count: int) -> int:
        # Keep tuning bounded so unattended overnight runs always terminate.
        if block_count >= 220:
            return 1_500
        if block_count >= 160:
            return 1_800
        if block_count >= 120:
            return 2_100
        if block_count >= 80:
            return 2_400
        return 2_700

    def _search_wall_time_limit_seconds(self, block_count: int) -> int:
        # Main MOEA-SA wall-time guard to avoid hanging jobs.
        if block_count >= 220:
            return 2_400
        if block_count >= 160:
            return 3_000
        if block_count >= 120:
            return 3_600
        if block_count >= 80:
            return 4_200
        return 4_800

    def _tune_hyperparameters_with_ga(self, request: GenerateTimetableRequest) -> SearchHyperParameters:
        if not hasattr(self, "block_requests"):
            return self._current_hyperparameters()

        block_count = len(self.block_requests)
        tuning_wall_time_limit_seconds = self._tuning_wall_time_limit_seconds(block_count)
        tuning_started_at = perf_counter()
        tuning_termination = "generation_budget_exhausted"
        profile_key = self._hyperparameter_profile_key()
        profile_scope = profile_key.split(":")[1] if ":" in profile_key else "single"
        lookup_keys = self._hyperparameter_cache_lookup_keys()
        self._report_progress(
            stage="tuning",
            progress_percent=5.0,
            message=f"Checking saved hyperparameter profile: {profile_key}.",
            metrics={
                "profile_key": profile_key,
                "profile_scope": profile_scope,
                "lookup_keys": [item[0] for item in lookup_keys],
            },
        )
        cached, cache_key, cache_source = self._load_cached_hyperparameters_for_scenario()
        if cached is not None:
            tuned_cached = self._coerce_hyperparameters(cached, block_count=block_count)
            self._apply_hyperparameters(tuned_cached)
            migrated = bool(cache_key and cache_key != profile_key)
            if migrated:
                self._persist_cached_hyperparameters_for_scenario(tuned_cached, cache_key=profile_key)
            source_label = cache_source or "profile"
            if migrated:
                hit_message = (
                    "Hyperparameter profile cache HIT via fallback key "
                    f"{cache_key}. Migrated to {profile_key} and reused."
                )
            else:
                hit_message = f"Hyperparameter profile cache HIT. Reusing saved parameters from {profile_key}."
            self._report_progress(
                stage="tuning",
                level="success",
                progress_percent=30.0,
                message=hit_message,
                metrics={
                    "cache_status": "hit",
                    "cache_source": source_label,
                    "cache_key": cache_key or profile_key,
                    "cache_migrated": migrated,
                    "profile_key": profile_key,
                    "profile_scope": profile_scope,
                    **tuned_cached.__dict__,
                },
            )
            return tuned_cached

        baseline = self._coerce_hyperparameters(self._current_hyperparameters(), block_count=block_count)
        settings_snapshot = self._current_hyperparameters()
        objective_cache: dict[tuple, tuple[float, float, float]] = {}

        meta_population_size = 10 if block_count >= 120 else 12
        meta_generations = 5 if block_count >= 160 else 6
        mutation_probability = 0.35
        elite_keep = 2

        self._report_progress(
            stage="tuning",
            progress_percent=5.0,
            message=(
                "Hyperparameter profile cache MISS. "
                f"Creating and saving new profile via GA tuning ({profile_key}): "
                f"population={meta_population_size}, generations={meta_generations}."
            ),
            metrics={
                "cache_status": "miss",
                "profile_key": profile_key,
                "profile_scope": profile_scope,
                "lookup_keys": [item[0] for item in lookup_keys],
                "wall_time_limit_seconds": tuning_wall_time_limit_seconds,
            },
        )

        population: list[SearchHyperParameters] = [baseline]
        while len(population) < meta_population_size:
            population.append(self._random_hyperparameters(block_count=block_count))

        best_params = baseline
        best_score = (float("inf"), float("inf"), float("inf"))

        try:
            for generation_index in range(meta_generations):
                elapsed_tuning_seconds = perf_counter() - tuning_started_at
                if elapsed_tuning_seconds >= tuning_wall_time_limit_seconds:
                    tuning_termination = "wall_time_budget_exhausted"
                    self._report_progress(
                        stage="tuning",
                        level="warn",
                        progress_percent=29.0,
                        message=(
                            "GA tuning wall-time budget reached. "
                            "Stopping early and reusing the best profile found so far."
                        ),
                        metrics={
                            "profile_key": profile_key,
                            "profile_scope": profile_scope,
                            "elapsed_seconds": round(elapsed_tuning_seconds, 2),
                            "wall_time_limit_seconds": tuning_wall_time_limit_seconds,
                            "completed_generations": generation_index,
                        },
                    )
                    break

                scored_population: list[tuple[tuple[float, float, float], SearchHyperParameters]] = []
                for params in population:
                    key = self._hyperparameter_signature(params)
                    if key not in objective_cache:
                        self._apply_hyperparameters(params)
                        objective_cache[key] = self._score_hyperparameters(request)
                    score = objective_cache[key]
                    scored_population.append((score, params))

                scored_population.sort(key=lambda item: item[0])
                if scored_population and scored_population[0][0] < best_score:
                    best_score = scored_population[0][0]
                    best_params = scored_population[0][1]

                if scored_population:
                    current_best = scored_population[0]
                    self._report_progress(
                        stage="tuning",
                        progress_percent=5.0 + (((generation_index + 1) / max(1, meta_generations)) * 25.0),
                        message=(
                            f"GA tuning generation {generation_index + 1}/{meta_generations}: "
                            f"best hard={int(current_best[0][0])}, combined_secondary={current_best[0][1]:.2f}."
                        ),
                        metrics={
                            "generation_index": generation_index + 1,
                            "best_hard_conflicts": int(current_best[0][0]),
                            "best_combined_secondary_objective": round(current_best[0][1], 4),
                            "best_runtime_penalty": round(current_best[0][2], 4),
                            "candidate_population_size": current_best[1].population_size,
                            "candidate_generations": current_best[1].generations,
                            "candidate_mutation_rate": round(current_best[1].mutation_rate, 4),
                            "candidate_crossover_rate": round(current_best[1].crossover_rate, 4),
                        },
                    )

                next_population = [item[1] for item in scored_population[:elite_keep]]
                while len(next_population) < meta_population_size:
                    parent_a = scored_population[self.random.randrange(max(1, len(scored_population) // 2))][1]
                    parent_b = scored_population[self.random.randrange(max(1, len(scored_population) // 2))][1]
                    child = self._crossover_hyperparameters(parent_a, parent_b, block_count=block_count)
                    child = self._mutate_hyperparameters(
                        child,
                        block_count=block_count,
                        mutation_probability=mutation_probability,
                    )
                    next_population.append(child)

                population = next_population
        finally:
            self._apply_hyperparameters(settings_snapshot)

        tuned = self._coerce_hyperparameters(best_params, block_count=block_count)
        self._apply_hyperparameters(tuned)
        self._persist_cached_hyperparameters_for_scenario(tuned)
        logger.info(
            "GA hyperparameter tuning complete program_id=%s term=%s hard=%s combined_secondary=%.2f pop=%s gen=%s mut=%.3f cross=%.3f elite=%s tour=%s stag=%s anneal_iter=%s anneal_temp=%.2f cool=%.5f",
            self.program_id,
            self.term_number,
            int(best_score[0]),
            best_score[1],
            tuned.population_size,
            tuned.generations,
            tuned.mutation_rate,
            tuned.crossover_rate,
            tuned.elite_count,
            tuned.tournament_size,
            tuned.stagnation_limit,
            tuned.annealing_iterations,
            tuned.annealing_initial_temperature,
            tuned.annealing_cooling_rate,
        )
        self._report_progress(
            stage="tuning",
            level="success",
            progress_percent=30.0,
            message=f"GA hyperparameter tuning complete. Saved profile for reuse: {profile_key}.",
            metrics={
                "termination": tuning_termination,
                "cache_status": "saved",
                "profile_key": profile_key,
                "profile_scope": profile_scope,
                "best_hard_conflicts": int(best_score[0]),
                "best_combined_secondary_objective": round(best_score[1], 4),
                "best_runtime_penalty": round(best_score[2], 4),
                "elapsed_seconds": round(perf_counter() - tuning_started_at, 2),
                "wall_time_limit_seconds": tuning_wall_time_limit_seconds,
                **tuned.__dict__,
            },
        )
        return tuned

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
        return self._run_moea_search(
            request,
            use_simulated_annealing=True,
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
        same_sync_scope = req_a.section == req_b.section or (
            self._is_project_phase_request(req_a) and self._is_project_phase_request(req_b)
        )
        return (
            req_a.is_lab
            and req_b.is_lab
            and req_a.course_id == req_b.course_id
            and same_sync_scope
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
        requires_room = self._request_requires_room(req)
        requires_faculty = self._request_requires_faculty(req)
        room = self.rooms[option.room_id] if requires_room else None
        faculty = self.faculty[option.faculty_id] if requires_faculty else None
        weights = self.settings.objective_weights

        hard = 0
        soft = 0.0
        block_start, block_end = self._option_bounds(option, req.block_size)

        if not self._within_semester_time_window(block_start, block_end):
            hard += weights.semester_limit
        if self._overlaps_non_teaching_window(day=option.day, start_min=block_start, end_min=block_end):
            hard += weights.semester_limit * 4

        reserved_room_conflict, reserved_faculty_conflict = self._reserved_conflict_flags(
            day=option.day,
            start_min=block_start,
            end_min=block_end,
            room_id=option.room_id,
            faculty_id=option.faculty_id,
            check_room=requires_room,
            check_faculty=requires_faculty,
        )
        if requires_room and reserved_room_conflict:
            hard += weights.room_conflict
        if requires_faculty and reserved_faculty_conflict:
            hard += weights.faculty_conflict

        if requires_room and room is not None:
            if room.capacity < req.student_count:
                hard += weights.room_capacity
            if req.is_lab and room.type != RoomType.lab:
                hard += weights.room_type
            if not req.is_lab and room.type == RoomType.lab:
                hard += weights.room_type

            room_windows = self.room_windows.get(option.room_id, {})
            if room_windows.get(option.day):
                if not any(start <= block_start and block_end <= end for start, end in room_windows[option.day]):
                    hard += weights.room_type

        if requires_faculty and faculty is not None:
            if not self._faculty_allows_day(faculty, option.day):
                hard += weights.faculty_availability

            faculty_windows = self.faculty_windows.get(option.faculty_id, {})
            if faculty_windows.get(option.day):
                if not any(start <= block_start and block_end <= end for start, end in faculty_windows[option.day]):
                    hard += weights.faculty_availability

        period_minutes = self.schedule_policy.period_minutes
        if requires_faculty and faculty is not None:
            projected_minutes = faculty_minutes.get(option.faculty_id, 0) + (req.block_size * period_minutes)
            max_minutes = self._effective_faculty_max_hours(faculty) * 60
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
        enforce_student_credit_load = True
        if self.program_constraint is not None:
            enforce_student_credit_load = bool(self.program_constraint.enforce_student_credit_load)
        if enforce_student_credit_load and self.expected_section_minutes > 0 and projected_section_minutes > self.expected_section_minutes:
            overflow_periods = max(
                1,
                math.ceil((projected_section_minutes - self.expected_section_minutes) / max(1, period_minutes)),
            )
            hard += weights.semester_limit * overflow_periods

        if not req.is_lab and requires_faculty:
            for other_idx, other_option in selected_options.items():
                other_req = self.block_requests[other_idx]
                if other_req.is_lab:
                    continue
                if not self._request_requires_faculty(other_req):
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

        if requires_faculty:
            for other_idx, other_option in selected_options.items():
                other_req = self.block_requests[other_idx]
                if not self._request_requires_faculty(other_req):
                    continue
                if self._is_faculty_back_to_back(req, option, other_req, other_option):
                    soft += max(1.0, weights.spread_balance * 0.75)

            if req.preferred_faculty_ids and option.faculty_id not in req.preferred_faculty_ids:
                soft += (weights.faculty_subject_preference * 0.20) * req.block_size
            if req.primary_faculty_id and option.faculty_id != req.primary_faculty_id:
                soft += (weights.faculty_subject_preference * 0.08) * req.block_size

        # Prefer tighter but feasible room fit to preserve larger rooms for heavier sections.
        if requires_room and room is not None and room.capacity > 0:
            soft += max(0, room.capacity - req.student_count) / room.capacity

        for offset in range(req.block_size):
            slot_idx = option.start_index + offset
            room_key = (option.day, slot_idx, option.room_id)
            faculty_key = (option.day, slot_idx, option.faculty_id)
            section_key = (option.day, slot_idx, req.section)

            if requires_room:
                for other_idx in room_occ.get(room_key, []):
                    other_req = self.block_requests[other_idx]
                    if not self._request_requires_room(other_req):
                        continue
                    if self._is_allowed_shared_overlap(req, other_req, option, selected_options[other_idx]):
                        continue
                    hard += weights.room_conflict

            if requires_faculty:
                for other_idx in faculty_occ.get(faculty_key, []):
                    other_req = self.block_requests[other_idx]
                    if not self._request_requires_faculty(other_req):
                        continue
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
        requires_room = self._request_requires_room(req)
        requires_faculty = self._request_requires_faculty(req)
        room = self.rooms[option.room_id] if requires_room else None
        faculty = self.faculty[option.faculty_id] if requires_faculty else None

        if not self._within_semester_time_window(block_start, block_end):
            return False
        if requires_faculty and faculty is not None and not self._faculty_allows_day(faculty, option.day):
            return False
        # Relax strict capacity check to allow spillover to smaller rooms instead of forcing room conflicts
        if requires_room and room is not None and req.is_lab and room.type != RoomType.lab:
            return False
        if requires_room and room is not None and not req.is_lab and room.type == RoomType.lab:
            return False
        if self._conflicts_reserved_resources(
            day=option.day,
            start_min=block_start,
            end_min=block_end,
            room_id=option.room_id,
            faculty_id=option.faculty_id,
            check_room=requires_room,
            check_faculty=requires_faculty,
        ):
            return False

        period_minutes = self.schedule_policy.period_minutes
        if requires_faculty and faculty is not None:
            projected_faculty_minutes = faculty_minutes.get(option.faculty_id, 0) + (req.block_size * period_minutes)
            max_faculty_minutes = self._effective_faculty_max_hours(faculty) * 60
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
        enforce_student_credit_load = True
        if self.program_constraint is not None:
            enforce_student_credit_load = bool(self.program_constraint.enforce_student_credit_load)
        if (
            enforce_student_credit_load
            and self.expected_section_minutes > 0
            and projected_section_slot_count * period_minutes > self.expected_section_minutes
        ):
            return False

        if requires_faculty:
            faculty_windows = self.faculty_windows.get(option.faculty_id, {})
            if faculty_windows.get(option.day):
                if not any(start <= block_start and block_end <= end for start, end in faculty_windows[option.day]):
                    return False

        if requires_room:
            room_windows = self.room_windows.get(option.room_id, {})
            if room_windows.get(option.day):
                if not any(start <= block_start and block_end <= end for start, end in room_windows[option.day]):
                    return False

        for offset in range(req.block_size):
            slot_idx = option.start_index + offset
            room_key = (option.day, slot_idx, option.room_id)
            faculty_key = (option.day, slot_idx, option.faculty_id)
            section_key = (option.day, slot_idx, req.section)

            if requires_room:
                for other_idx in room_occ.get(room_key, []):
                    other_req = self.block_requests[other_idx]
                    if not self._request_requires_room(other_req):
                        continue
                    if not self._is_allowed_shared_overlap(req, other_req, option, selected_options[other_idx]):
                        return False

            if requires_faculty:
                for other_idx in faculty_occ.get(faculty_key, []):
                    other_req = self.block_requests[other_idx]
                    if not self._request_requires_faculty(other_req):
                        continue
                    if not self._is_allowed_shared_overlap(req, other_req, option, selected_options[other_idx]):
                        return False

            for other_idx in section_occ.get(section_key, []):
                other_req = self.block_requests[other_idx]
                if not self._parallel_lab_overlap_allowed(req, other_req):
                    return False

        # Keep one faculty per (course, section) for lecture/tutorial requests.
        if not req.is_lab and requires_faculty:
            for other_idx, other_option in selected_options.items():
                other_req = self.block_requests[other_idx]
                if other_req.is_lab or other_req.course_id != req.course_id:
                    continue
                if not self._request_requires_faculty(other_req):
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
            faculty_id: self._effective_faculty_max_hours(faculty) * 60
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
                and self._request_requires_faculty(self.block_requests[req_index])
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
            applicable_req_indices = [
                req_index
                for req_index in req_indices
                if self._request_requires_faculty(self.block_requests[req_index])
            ]
            if not applicable_req_indices:
                continue
            required_minutes = sum(
                self.block_requests[req_index].block_size * period_minutes for req_index in applicable_req_indices
            )
            candidate_ids = list(common_faculty_candidates_by_course_section.get((course_id, section_name), ()))
            if not candidate_ids:
                candidate_ids = sorted(
                    set.intersection(
                        *[
                            {option.faculty_id for option in self.block_requests[req_index].options}
                            for req_index in applicable_req_indices
                        ]
                    )
                    if applicable_req_indices
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
            if req.is_lab or not self._request_requires_faculty(req):
                return None
            for other_idx in request_indices_by_course_section.get((req.course_id, req.section), []):
                if other_idx not in selected_options:
                    continue
                other_req = self.block_requests[other_idx]
                if not self._request_requires_faculty(other_req):
                    continue
                return selected_options[other_idx].faculty_id
            if self._single_faculty_required(req.course_id):
                for other_idx in request_indices_by_course.get(req.course_id, []):
                    if other_idx not in selected_options:
                        continue
                    other_req = self.block_requests[other_idx]
                    if other_req.is_lab or not self._request_requires_faculty(other_req):
                        continue
                    return selected_options[other_idx].faculty_id
            return None

        def ordered_candidates(req_index: int) -> list[int]:
            req = self.block_requests[req_index]
            planned_faculty_id = (
                planned_faculty_by_course_section.get((req.course_id, req.section))
                if self._request_requires_faculty(req)
                else None
            )

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
            if fixed_faculty_id is not None and self._request_requires_faculty(req):
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
            elif planned_faculty_id is not None and self._request_requires_faculty(req):
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
                capacity_waste = 0.0
                if self._request_requires_room(req):
                    room = self.rooms[req.options[option_index].room_id]
                    if room.capacity >= req.student_count:
                        capacity_waste = (room.capacity - req.student_count) / max(1, room.capacity)
                final_score = (hard_score * 10000.0) + soft_score + (capacity_waste * 0.5)
                if (
                    not req.is_lab
                    and self._request_requires_faculty(req)
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
                if not req.is_lab and self._request_requires_faculty(req):
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

            if not req.is_lab and self._request_requires_faculty(req):
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
                
                capacity_waste = 0.0
                if self._request_requires_room(req):
                    room = self.rooms[req.options[opt_idx].room_id]
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
            if not req.is_lab and self._request_requires_faculty(req):
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
        requires_room = self._request_requires_room(req)
        requires_faculty = self._request_requires_faculty(req)
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

            if requires_room:
                room_occ[room_key].append(req_index)
            if requires_faculty:
                faculty_occ[faculty_key].append(req_index)
            section_occ[section_key].append(req_index)
            section_slot_keys[req.section].add((option.day, slot_idx))

        if requires_faculty:
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
        requires_room = self._request_requires_room(req)
        requires_faculty = self._request_requires_faculty(req)
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

            if requires_room:
                room_entries = room_occ.get(room_key, [])
                if req_index in room_entries:
                    room_entries.remove(req_index)
                if not room_entries:
                    room_occ.pop(room_key, None)

            if requires_faculty:
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

        if requires_faculty:
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
        left_obj = EvolutionaryScheduler._objective_values(left)
        right_obj = EvolutionaryScheduler._objective_values(right)
        return (
            all(left_value <= right_value for left_value, right_value in zip(left_obj, right_obj))
            and any(left_value < right_value for left_value, right_value in zip(left_obj, right_obj))
        )

    @staticmethod
    def _is_better_eval(left: EvaluationResult, right: EvaluationResult) -> bool:
        left_obj = EvolutionaryScheduler._objective_values(left)
        right_obj = EvolutionaryScheduler._objective_values(right)
        if left_obj != right_obj:
            return left_obj < right_obj
        return left.fitness > right.fitness

    @staticmethod
    def _objective_values(evaluation: EvaluationResult) -> tuple[float, float, float]:
        default = (0.0, 0.0, 0.0)
        if evaluation.objectives != default:
            return (
                float(evaluation.objectives[0]),
                float(evaluation.objectives[1]),
                float(evaluation.objectives[2]),
            )
        return (
            float(evaluation.hard_conflicts),
            float(evaluation.soft_penalty),
            float(max(0.0, -evaluation.fitness)),
        )

    @staticmethod
    def _annealing_energy(evaluation: EvaluationResult) -> float:
        objective_1, objective_2, objective_3 = EvolutionaryScheduler._objective_values(evaluation)
        return (objective_1 * 10_000.0) + objective_2 + (objective_3 * 5.0)

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
                    ",".join(sorted(slot.assistant_faculty_ids or [])),
                    slot.section,
                    slot.batch or "",
                    slot.sessionType or "",
                )
                for slot in payload.timetable_data
            )
        )

    def _non_dominated_sort(
        self,
        evaluations: list[EvaluationResult],
    ) -> tuple[list[list[int]], dict[int, int]]:
        if not evaluations:
            return [], {}

        domination_sets: list[set[int]] = [set() for _ in evaluations]
        dominated_counts = [0] * len(evaluations)
        fronts: list[list[int]] = [[]]

        for left_index in range(len(evaluations)):
            left_eval = evaluations[left_index]
            for right_index in range(left_index + 1, len(evaluations)):
                right_eval = evaluations[right_index]
                if self._dominates_eval(left_eval, right_eval):
                    domination_sets[left_index].add(right_index)
                    dominated_counts[right_index] += 1
                elif self._dominates_eval(right_eval, left_eval):
                    domination_sets[right_index].add(left_index)
                    dominated_counts[left_index] += 1

            if dominated_counts[left_index] == 0:
                fronts[0].append(left_index)

        rank_by_index: dict[int, int] = {}
        front_cursor = 0
        while front_cursor < len(fronts) and fronts[front_cursor]:
            next_front: list[int] = []
            for current_index in fronts[front_cursor]:
                rank_by_index[current_index] = front_cursor
                for dominated_index in domination_sets[current_index]:
                    dominated_counts[dominated_index] -= 1
                    if dominated_counts[dominated_index] == 0:
                        next_front.append(dominated_index)
            if next_front:
                fronts.append(next_front)
            front_cursor += 1

        return fronts, rank_by_index

    def _crowding_distances(
        self,
        front: list[int],
        evaluations: list[EvaluationResult],
    ) -> dict[int, float]:
        if not front:
            return {}
        if len(front) <= 2:
            return {index: float("inf") for index in front}

        distances = {index: 0.0 for index in front}
        objectives = (
            lambda idx: self._objective_values(evaluations[idx])[0],
            lambda idx: self._objective_values(evaluations[idx])[1],
            lambda idx: self._objective_values(evaluations[idx])[2],
        )

        for objective in objectives:
            ordered = sorted(front, key=objective)
            distances[ordered[0]] = float("inf")
            distances[ordered[-1]] = float("inf")

            minimum = objective(ordered[0])
            maximum = objective(ordered[-1])
            span = maximum - minimum
            if span <= 1e-9:
                continue

            for position in range(1, len(ordered) - 1):
                index = ordered[position]
                if math.isinf(distances[index]):
                    continue
                previous_value = objective(ordered[position - 1])
                next_value = objective(ordered[position + 1])
                distances[index] += (next_value - previous_value) / span

        return distances

    def _moea_parent_better(
        self,
        left_index: int,
        right_index: int,
        *,
        evaluations: list[EvaluationResult],
        rank_by_index: dict[int, int],
        crowding_by_index: dict[int, float],
    ) -> bool:
        left_rank = rank_by_index.get(left_index, math.inf)
        right_rank = rank_by_index.get(right_index, math.inf)
        if left_rank != right_rank:
            return left_rank < right_rank

        left_crowding = crowding_by_index.get(left_index, 0.0)
        right_crowding = crowding_by_index.get(right_index, 0.0)
        if left_crowding != right_crowding:
            return left_crowding > right_crowding

        return self._is_better_eval(evaluations[left_index], evaluations[right_index])

    def _select_moea_parent(
        self,
        population: list[list[int]],
        evaluations: list[EvaluationResult],
        rank_by_index: dict[int, int],
        crowding_by_index: dict[int, float],
    ) -> list[int]:
        tournament_size = min(self.settings.tournament_size, len(population))
        contenders = self.random.sample(range(len(population)), tournament_size)
        best_index = contenders[0]
        for contender_index in contenders[1:]:
            if self._moea_parent_better(
                contender_index,
                best_index,
                evaluations=evaluations,
                rank_by_index=rank_by_index,
                crowding_by_index=crowding_by_index,
            ):
                best_index = contender_index
        return list(population[best_index])

    def _add_pareto_candidate(
        self,
        archive: list[tuple[EvaluationResult, list[int]]],
        seen_genotypes: set[tuple[int, ...]],
        *,
        genes: list[int],
        evaluation: EvaluationResult,
        archive_limit: int,
    ) -> None:
        key = tuple(genes)
        if key in seen_genotypes:
            return
        seen_genotypes.add(key)

        survivors: list[tuple[EvaluationResult, list[int]]] = []
        for existing_eval, existing_genes in archive:
            if self._dominates_eval(existing_eval, evaluation):
                return
            if not self._dominates_eval(evaluation, existing_eval):
                survivors.append((existing_eval, existing_genes))
        survivors.append((evaluation, list(genes)))
        survivors.sort(key=lambda item: self._eval_sort_key(item[0]))
        archive[:] = survivors[:archive_limit]

    def _simulated_annealing_refine(
        self,
        seed_genes: list[int],
        seed_evaluation: EvaluationResult,
        *,
        iterations: int,
    ) -> tuple[list[int], EvaluationResult]:
        current_genes = list(seed_genes)
        current_eval = seed_evaluation
        best_genes = list(seed_genes)
        best_eval = seed_evaluation

        temperature = max(0.05, self.settings.annealing_initial_temperature)
        cooling_rate = self.settings.annealing_cooling_rate
        local_iterations = max(12, iterations)

        for step in range(local_iterations):
            progress = step / max(1, local_iterations - 1)
            intensity = min(0.38, 0.03 + (0.18 * progress))
            candidate = self._perturb_individual(current_genes, intensity=intensity)

            if self.random.random() < 0.85:
                mutation_scale = 1.35 if current_eval.hard_conflicts > 0 else 1.0
                mutation_rate = min(0.45, self.settings.mutation_rate * mutation_scale)
                candidate = self._mutate(candidate, mutation_rate=mutation_rate)
            if self.random.random() < 0.25 or current_eval.hard_conflicts > 0:
                candidate = self._repair_individual(candidate, max_passes=1)

            candidate_eval = self._evaluate(candidate)
            delta = self._annealing_energy(candidate_eval) - self._annealing_energy(current_eval)
            if delta <= 0:
                current_genes = candidate
                current_eval = candidate_eval
            else:
                acceptance_probability = math.exp(-delta / max(temperature, 1e-9))
                if self.random.random() < acceptance_probability:
                    current_genes = candidate
                    current_eval = candidate_eval

            if self._is_better_eval(candidate_eval, best_eval):
                best_genes = list(candidate)
                best_eval = candidate_eval

            temperature *= cooling_rate
            if temperature < 0.02:
                temperature = max(0.05, self.settings.annealing_initial_temperature * 0.30)

            if best_eval.hard_conflicts == 0 and step >= max(10, local_iterations // 3):
                break

        return best_genes, best_eval

    def _build_response_from_archive(
        self,
        *,
        request: GenerateTimetableRequest,
        start_time: float,
        archive: list[tuple[EvaluationResult, list[int]]],
        block_count: int,
        fallback_attempt_multiplier: int,
    ) -> GenerateTimetableResponse:
        ranked = sorted(archive, key=lambda item: self._eval_sort_key(item[0]))
        alternatives: list[GeneratedAlternative] = []
        seen_fingerprints: set[str] = set()
        intensive_budget = (
            max(2, request.alternative_count * 2)
            if block_count >= 180
            else max(4, request.alternative_count * 4)
        )
        intensive_step_cap = self._intensive_repair_step_cap()

        for evaluation, genes in ranked:
            candidate_genes = list(genes)
            candidate_eval = evaluation
            if candidate_eval.hard_conflicts > 0 and intensive_budget > 0:
                intensive_budget -= 1
                intensified_genes, intensified_eval = self._intensive_conflict_repair(
                    candidate_genes,
                    max_steps=intensive_step_cap,
                )
                if self._is_better_eval(intensified_eval, candidate_eval):
                    candidate_genes = intensified_genes
                    candidate_eval = intensified_eval

            payload = self._decode_payload(candidate_genes)
            fingerprint = self._payload_fingerprint(payload)
            fingerprint_key = repr(fingerprint)
            if fingerprint_key in seen_fingerprints:
                continue
            seen_fingerprints.add(fingerprint_key)
            alternatives.append(
                GeneratedAlternative(
                    rank=len(alternatives) + 1,
                    fitness=candidate_eval.fitness,
                    hard_conflicts=candidate_eval.hard_conflicts,
                    soft_penalty=candidate_eval.soft_penalty,
                    payload=payload,
                )
            )
            if len(alternatives) >= request.alternative_count:
                break

        attempts = 0
        max_attempts = request.alternative_count * fallback_attempt_multiplier
        while len(alternatives) < request.alternative_count and attempts < max_attempts:
            attempts += 1
            seed = self._constructive_individual(randomized=True, rcl_alpha=0.35)
            if seed is None:
                seed = self._random_individual()
            candidate_genes = self._repair_individual(seed, max_passes=1)
            candidate_eval = self._evaluate(candidate_genes)
            if candidate_eval.hard_conflicts > 0 and intensive_budget > 0:
                intensive_budget -= 1
                candidate_genes, candidate_eval = self._intensive_conflict_repair(
                    candidate_genes,
                    max_steps=intensive_step_cap,
                )
            payload = self._decode_payload(candidate_genes)
            fingerprint = self._payload_fingerprint(payload)
            fingerprint_key = repr(fingerprint)
            if fingerprint_key in seen_fingerprints:
                continue
            seen_fingerprints.add(fingerprint_key)
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

        runtime_ms = int((perf_counter() - start_time) * 1000)
        return GenerateTimetableResponse(
            alternatives=alternatives,
            settings_used=self.settings,
            runtime_ms=runtime_ms,
        )

    def _run_moea_search(
        self,
        request: GenerateTimetableRequest,
        *,
        use_simulated_annealing: bool,
        auto_tune: bool = True,
    ) -> GenerateTimetableResponse:
        start = perf_counter()
        solver_strategy = str(getattr(self.settings, "solver_strategy", "") or "").strip().lower()
        can_auto_tune = auto_tune and all(
            hasattr(self, attr)
            for attr in ("program_id", "term_number", "expected_section_minutes", "block_requests", "day_slots")
        ) and solver_strategy != "fast"
        self._report_progress(
            stage="initialization",
            progress_percent=0.0,
            message="Preparing MOEA-SA optimizer.",
            metrics={"auto_tune_enabled": can_auto_tune, "sa_enabled": use_simulated_annealing},
        )
        if can_auto_tune:
            self._tune_hyperparameters_with_ga(request)

        block_count = len(self.block_requests)
        population = self._build_initial_population()
        if not population:
            population = [self._repair_individual(self._random_individual(), max_passes=1)]

        evaluations = [self._evaluate(genes) for genes in population]
        archive: list[tuple[EvaluationResult, list[int]]] = []
        seen_genotypes: set[tuple[int, ...]] = set()
        archive_limit = min(180, max(36, request.alternative_count * 36))
        for genes, evaluation in zip(population, evaluations):
            self._add_pareto_candidate(
                archive,
                seen_genotypes,
                genes=genes,
                evaluation=evaluation,
                archive_limit=archive_limit,
            )

        generation_cap = self.settings.generations
        if block_count >= 220:
            generation_cap = min(generation_cap, 24)
        elif block_count >= 160:
            generation_cap = min(generation_cap, 36)
        elif block_count >= 120:
            generation_cap = min(generation_cap, 54)
        elif block_count >= 80:
            generation_cap = min(generation_cap, 78)
        else:
            generation_cap = min(generation_cap, 120)
        generation_cap = max(8, generation_cap)

        sa_iterations = max(16, self.settings.annealing_iterations // 8)
        if block_count >= 220:
            sa_iterations = min(sa_iterations, 28)
        elif block_count >= 160:
            sa_iterations = min(sa_iterations, 36)
        elif block_count >= 120:
            sa_iterations = min(sa_iterations, 48)
        elif block_count >= 80:
            sa_iterations = min(sa_iterations, 64)
        else:
            sa_iterations = min(sa_iterations, 90)

        search_wall_time_limit_seconds = self._search_wall_time_limit_seconds(block_count)
        search_started_at = perf_counter()
        search_deadline = search_started_at + search_wall_time_limit_seconds

        best_eval = min(evaluations, key=self._eval_sort_key)
        stagnant = 0
        termination_reason = "generation_budget_exhausted"
        best_genes = list(population[min(range(len(population)), key=lambda idx: self._eval_sort_key(evaluations[idx]))])

        self._report_progress(
            stage="search",
            progress_percent=35.0,
            message="Starting MOEA exploration and SA exploitation.",
            metrics={
                "population_size": self.settings.population_size,
                "generation_cap": generation_cap,
                "annealing_iterations": sa_iterations,
                "search_wall_time_limit_seconds": search_wall_time_limit_seconds,
            },
            best_genes=best_genes,
            best_evaluation=best_eval,
        )

        for generation in range(generation_cap):
            if perf_counter() >= search_deadline:
                termination_reason = "wall_time_budget_exhausted"
                elapsed_seconds = perf_counter() - search_started_at
                self._report_progress(
                    stage="search",
                    level="warn",
                    progress_percent=90.0,
                    message=(
                        "MOEA-SA wall-time budget reached. "
                        "Stopping with the best feasible archive found so far."
                    ),
                    metrics={
                        "generation_index": generation,
                        "elapsed_seconds": round(elapsed_seconds, 2),
                        "search_wall_time_limit_seconds": search_wall_time_limit_seconds,
                        "hard_conflicts": best_eval.hard_conflicts,
                        "soft_penalty": round(best_eval.soft_penalty, 4),
                        "fitness": round(best_eval.fitness, 4),
                    },
                    best_genes=best_genes,
                    best_evaluation=best_eval,
                )
                break

            fronts, rank_by_index = self._non_dominated_sort(evaluations)
            crowding_by_index: dict[int, float] = {}
            for front in fronts:
                crowding_by_index.update(self._crowding_distances(front, evaluations))

            ordered_indices = sorted(
                range(len(population)),
                key=lambda index: (
                    rank_by_index.get(index, math.inf),
                    -crowding_by_index.get(index, 0.0),
                    *self._eval_sort_key(evaluations[index]),
                ),
            )
            if not ordered_indices:
                break

            archive_window = min(
                len(ordered_indices),
                max(request.alternative_count * 12, self.settings.elite_count * 6),
            )
            for candidate_index in ordered_indices[:archive_window]:
                self._add_pareto_candidate(
                    archive,
                    seen_genotypes,
                    genes=population[candidate_index],
                    evaluation=evaluations[candidate_index],
                    archive_limit=archive_limit,
                )

            generation_best_eval = evaluations[ordered_indices[0]]
            if self._is_better_eval(generation_best_eval, best_eval):
                best_eval = generation_best_eval
                best_genes = list(population[ordered_indices[0]])
                stagnant = 0
            else:
                stagnant += 1

            self._report_progress(
                stage="search",
                progress_percent=35.0 + (((generation + 1) / max(1, generation_cap)) * 55.0),
                message=(
                    f"MOEA generation {generation + 1}/{generation_cap}: "
                    f"best hard={best_eval.hard_conflicts}, soft={best_eval.soft_penalty:.2f}, "
                    f"stagnant={stagnant}."
                ),
                metrics={
                    "generation_index": generation + 1,
                    "hard_conflicts": best_eval.hard_conflicts,
                    "soft_penalty": round(best_eval.soft_penalty, 4),
                    "workload_balance_penalty": round(best_eval.workload_balance_penalty, 4),
                    "fitness": round(best_eval.fitness, 4),
                    "stagnant_generations": stagnant,
                },
                best_genes=best_genes,
                best_evaluation=best_eval,
            )

            if (
                best_eval.hard_conflicts == 0
                and len(archive) >= max(4, request.alternative_count)
                and stagnant >= max(6, self.settings.stagnation_limit // 3)
            ):
                termination_reason = "converged_conflict_free_with_stagnation"
                break

            mutation_rate = self._adaptive_mutation_rate(stagnant)
            next_population: list[list[int]] = []
            elite_keep = min(len(ordered_indices), max(2, self.settings.elite_count))
            for index in ordered_indices[:elite_keep]:
                next_population.append(list(population[index]))

            if use_simulated_annealing and (generation % 2 == 0 or best_eval.hard_conflicts > 0):
                refinement_targets = ordered_indices[: min(len(ordered_indices), max(1, request.alternative_count))]
                for index in refinement_targets:
                    refined_genes, refined_eval = self._simulated_annealing_refine(
                        population[index],
                        evaluations[index],
                        iterations=sa_iterations,
                    )
                    self._add_pareto_candidate(
                        archive,
                        seen_genotypes,
                        genes=refined_genes,
                        evaluation=refined_eval,
                        archive_limit=archive_limit,
                    )
                    if len(next_population) < self.settings.population_size:
                        next_population.append(refined_genes)

            while len(next_population) < self.settings.population_size:
                parent_a = self._select_moea_parent(
                    population,
                    evaluations,
                    rank_by_index,
                    crowding_by_index,
                )
                parent_b = self._select_moea_parent(
                    population,
                    evaluations,
                    rank_by_index,
                    crowding_by_index,
                )
                if self.random.random() < self.settings.crossover_rate:
                    child = self._crossover(parent_a, parent_b)
                else:
                    child = list(parent_a)

                if self.random.random() < 0.85:
                    child = self._mutate(child, mutation_rate=mutation_rate)

                perturb_probability = min(0.40, 0.08 + (0.03 * stagnant))
                if self.random.random() < perturb_probability:
                    progress = generation / max(1, generation_cap - 1)
                    perturb_intensity = min(0.40, 0.05 + (0.17 * progress))
                    child = self._perturb_individual(child, intensity=perturb_intensity)

                if self.random.random() < 0.20 or generation_best_eval.hard_conflicts > 0:
                    child = self._repair_individual(child, max_passes=1)

                next_population.append(child)

            if stagnant >= self.settings.stagnation_limit:
                restart_count = max(1, self.settings.population_size // 4)
                for offset in range(restart_count):
                    restart_index = len(next_population) - 1 - offset
                    replacement = self._random_individual()
                    if self.random.random() < 0.60:
                        replacement = self._repair_individual(replacement, max_passes=1)
                    next_population[restart_index] = replacement
                stagnant = self.settings.stagnation_limit // 3

            population = next_population[: self.settings.population_size]
            evaluations = [self._evaluate(genes) for genes in population]

        if generation_cap <= 0:
            termination_reason = "no_generations_configured"

        for genes, evaluation in zip(population, evaluations):
            self._add_pareto_candidate(
                archive,
                seen_genotypes,
                genes=genes,
                evaluation=evaluation,
                archive_limit=archive_limit,
            )

        result = self._build_response_from_archive(
            request=request,
            start_time=start,
            archive=archive,
            block_count=block_count,
            fallback_attempt_multiplier=20,
        )
        best_result = result.alternatives[0]
        self._report_progress(
            stage="finalization",
            level="success",
            progress_percent=92.0,
            message=(
                "MOEA-SA optimization phase complete. "
                "Applying final post-processing. "
                f"Termination: {termination_reason}. Best hard conflicts: {best_result.hard_conflicts}."
            ),
            metrics={
                "termination_reason": termination_reason,
                "runtime_ms": result.runtime_ms,
                "alternatives": len(result.alternatives),
                "best_hard_conflicts": best_result.hard_conflicts,
                "best_soft_penalty": round(best_result.soft_penalty, 4),
                "best_workload_balance_penalty": round(getattr(best_eval, "workload_balance_penalty", 0.0), 4),
                "best_fitness": round(best_result.fitness, 4),
            },
        )
        return result

    def _run_hybrid_search(self, request: GenerateTimetableRequest) -> GenerateTimetableResponse:
        return self._run_moea_search(
            request,
            use_simulated_annealing=True,
        )

    def _run_simulated_annealing(self, request: GenerateTimetableRequest) -> GenerateTimetableResponse:
        return self._run_moea_search(
            request,
            use_simulated_annealing=True,
        )

    def _run_fast_solver(self, request: GenerateTimetableRequest) -> GenerateTimetableResponse:
        baseline = self._current_hyperparameters()
        fast_profile = self._coerce_hyperparameters(
            SearchHyperParameters(
                population_size=min(self.settings.population_size, 26),
                generations=min(self.settings.generations, 18),
                mutation_rate=self.settings.mutation_rate,
                crossover_rate=self.settings.crossover_rate,
                elite_count=min(self.settings.elite_count, 4),
                tournament_size=min(self.settings.tournament_size, 3),
                stagnation_limit=min(self.settings.stagnation_limit, 10),
                annealing_iterations=min(self.settings.annealing_iterations, 90),
                annealing_initial_temperature=self.settings.annealing_initial_temperature,
                annealing_cooling_rate=self.settings.annealing_cooling_rate,
            ),
            block_count=len(getattr(self, "block_requests", [])),
        )
        self._apply_hyperparameters(fast_profile)
        try:
            return self._run_moea_search(
                request,
                use_simulated_annealing=True,
                auto_tune=False,
            )
        finally:
            self._apply_hyperparameters(baseline)

    def _merge_results(
        self,
        *,
        primary: GenerateTimetableResponse,
        secondary: GenerateTimetableResponse,
        alternative_count: int,
    ) -> GenerateTimetableResponse:
        merged: list[GeneratedAlternative] = []
        seen_fingerprints: set[str] = set()

        ordered = [*primary.alternatives, *secondary.alternatives]
        ordered.sort(key=self._eval_sort_key)

        for candidate in ordered:
            fingerprint = self._payload_fingerprint(candidate.payload)
            fingerprint_key = repr(fingerprint)
            if fingerprint_key in seen_fingerprints:
                continue
            seen_fingerprints.add(fingerprint_key)
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
        logger.info(
            "Scheduler run strategy=moea_sa_auto program_id=%s term=%s alternatives=%s",
            self.program_id,
            self.term_number,
            request.alternative_count,
        )

        if not hasattr(self, "block_requests"):
            # Compatibility fallback for lightweight unit stubs that bypass __init__.
            hybrid = self._run_hybrid_search(request)
            if any(item.hard_conflicts == 0 for item in hybrid.alternatives):
                return hybrid
            annealed = self._run_simulated_annealing(request)
            if annealed.alternatives:
                return annealed
            return hybrid

        return self._run_moea_search(
            request,
            use_simulated_annealing=True,
            auto_tune=True,
        )
