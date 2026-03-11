from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, timezone
import logging
import threading
from time import perf_counter
import uuid
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db, require_roles
from app.db.session import SessionLocal
from app.models.faculty import Faculty
from app.models.institution_settings import InstitutionSettings
from app.models.notification import NotificationType
from app.models.program_constraint import ProgramConstraint
from app.models.program_structure import ProgramTerm
from app.models.timetable import OfficialTimetable
from app.models.timetable_generation import (
    ReevaluationStatus,
    TimetableGenerationSettings,
    TimetableSlotLock,
)
from app.models.timetable_conflict_decision import ConflictDecision, TimetableConflictDecision
from app.models.timetable_version import TimetableVersion
from app.models.user import User, UserRole
from app.schemas.generator import (
    AutoResolvedConflictEntry,
    FacultyWorkloadBridgeSuggestion,
    FacultyWorkloadGapSuggestion,
    GenerateTimetableRequest,
    GenerateTimetableCycleRequest,
    GenerateTimetableCycleResponse,
    GenerateTimetableResponse,
    GenerationJobAccepted,
    GenerationJobStatusOut,
    GeneratedCycleSolution,
    GeneratedCycleSolutionTerm,
    GeneratedCycleTermResult,
    GenerationSettingsBase,
    GenerationSettingsOut,
    GenerationSettingsUpdate,
    OccupancyMatrix,
    ReevaluateTimetableRequest,
    ReevaluateTimetableResponse,
    ReevaluationEventOut,
    SlotLockCreate,
    SlotLockOut,
)
from app.schemas.timetable import OfficialTimetablePayload
from app.schemas.settings import DEFAULT_ACADEMIC_CYCLE, parse_time_to_minutes
from app.services.audit import log_activity
from app.services.evolution_scheduler import EvolutionaryScheduler
from app.services.generation_jobs import generation_job_store
from app.services.notifications import notify_all_users
from app.services.reevaluation import (
    list_reevaluation_events,
    official_scope_impacted,
    resolve_reevaluation_events,
)

router = APIRouter()
logger = logging.getLogger(__name__)


def _slot_assistant_faculty_ids(slot: object) -> tuple[str, ...]:
    raw = getattr(slot, "assistant_faculty_ids", None)
    if raw is None:
        raw = getattr(slot, "assistantFacultyIds", None)
    if not isinstance(raw, list):
        return tuple()
    seen: set[str] = set()
    ordered: list[str] = []
    for item in raw:
        faculty_id = str(item or "").strip()
        if not faculty_id or faculty_id in seen:
            continue
        seen.add(faculty_id)
        ordered.append(faculty_id)
    return tuple(ordered)


def _dedupe_auto_resolved_conflicts(
    entries: list[AutoResolvedConflictEntry],
    *,
    limit: int = 200,
) -> list[AutoResolvedConflictEntry]:
    unique: list[AutoResolvedConflictEntry] = []
    seen_ids: set[str] = set()
    for entry in entries:
        conflict_id = entry.conflict_id.strip()
        if not conflict_id or conflict_id in seen_ids:
            continue
        seen_ids.add(conflict_id)
        unique.append(entry)
        if len(unique) >= limit:
            break
    return unique


def default_generation_settings() -> GenerationSettingsBase:
    return GenerationSettingsBase(solver_strategy="hybrid")


def load_generation_settings(db: Session) -> GenerationSettingsOut:
    record = db.get(TimetableGenerationSettings, 1)
    if record is None:
        defaults = default_generation_settings()
        return GenerationSettingsOut(id=1, **defaults.model_dump())
    return GenerationSettingsOut(
        id=record.id,
        solver_strategy=record.solver_strategy,
        population_size=record.population_size,
        generations=record.generations,
        mutation_rate=record.mutation_rate,
        crossover_rate=record.crossover_rate,
        elite_count=record.elite_count,
        tournament_size=record.tournament_size,
        stagnation_limit=record.stagnation_limit,
        annealing_iterations=record.annealing_iterations,
        annealing_initial_temperature=record.annealing_initial_temperature,
        annealing_cooling_rate=record.annealing_cooling_rate,
        random_seed=record.random_seed,
        objective_weights=record.objective_weights,
    )


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


def _timestamped_generation_label(*, prefix: str, program_id: str, term_number: int) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    compact_program = "".join(ch for ch in program_id if ch.isalnum())[:10] or "program"
    return f"{prefix}-{compact_program}-t{term_number}-{ts}-{suffix}"


def _persist_generated_snapshot_version(
    *,
    db: Session,
    current_user: User,
    generation: GenerateTimetableResponse,
    label_prefix: str = "gen",
    cycle: str | None = None,
    term_number: int | None = None,
) -> str:
    best = generation.alternatives[0]
    resolved_term = term_number or (best.payload.term_number or 0)
    label = _timestamped_generation_label(
        prefix=label_prefix,
        program_id=best.payload.program_id,
        term_number=resolved_term,
    )
    summary = {
        "program_id": best.payload.program_id,
        "term_number": resolved_term,
        "slots": len(best.payload.timetable_data),
        "conflicts": best.hard_conflicts,
        "source": "generator-auto-save",
        "cycle": cycle,
        "auto_saved": True,
    }
    db.add(
        TimetableVersion(
            label=label,
            payload=best.payload.model_dump(by_alias=True),
            summary=summary,
            created_by_id=current_user.id,
        )
    )
    generation.auto_saved_version_label = label
    return label


def _timestamped_cycle_bundle_label(*, program_id: str, cycle: str, term_numbers: list[int]) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    compact_program = "".join(ch for ch in program_id if ch.isalnum())[:10] or "program"
    terms = "-".join(str(item) for item in sorted(set(term_numbers))) or "none"
    return f"cyclegen-{compact_program}-cycle-{cycle}-{terms}-{ts}-{suffix}"


def _build_cycle_combined_payload(
    *,
    terms: list[GeneratedCycleSolutionTerm],
) -> OfficialTimetablePayload:
    if not terms:
        raise ValueError("Cannot build combined cycle payload without term solutions")

    ordered_terms = sorted(terms, key=lambda item: item.term_number)
    base_program_id = ordered_terms[0].payload.program_id

    faculty_by_id: dict[str, object] = {}
    course_by_id: dict[str, object] = {}
    room_by_id: dict[str, object] = {}
    merged_slots: list[dict] = []

    for term in ordered_terms:
        payload = term.payload
        for faculty in payload.faculty_data:
            faculty_by_id.setdefault(faculty.id, faculty)
        for course in payload.course_data:
            course_by_id.setdefault(course.id, course)
        for room in payload.room_data:
            room_by_id.setdefault(room.id, room)

        for slot in payload.timetable_data:
            slot_dict = slot.model_dump(by_alias=True)
            # Keep slot IDs unique when aggregating multiple semesters into one cycle snapshot.
            slot_dict["id"] = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"cycle:{base_program_id}:{term.term_number}:{slot.id}",
                )
            )
            merged_slots.append(slot_dict)

    return OfficialTimetablePayload(
        programId=base_program_id,
        termNumber=None,
        facultyData=[item.model_dump(by_alias=True) for item in faculty_by_id.values()],
        courseData=[item.model_dump(by_alias=True) for item in course_by_id.values()],
        roomData=[item.model_dump(by_alias=True) for item in room_by_id.values()],
        timetableData=merged_slots,
    )


def _persist_cycle_combined_snapshot_version(
    *,
    db: Session,
    current_user: User,
    cycle: str,
    terms: list[GeneratedCycleSolutionTerm],
    selected_solution_rank: int,
    state_metrics: dict,
) -> str:
    combined_payload = _build_cycle_combined_payload(terms=terms)
    label = _timestamped_cycle_bundle_label(
        program_id=combined_payload.program_id or "program",
        cycle=cycle,
        term_numbers=[item.term_number for item in terms],
    )
    summary = {
        "program_id": combined_payload.program_id,
        "term_number": None,
        "term_numbers": [item.term_number for item in terms],
        "slots": len(combined_payload.timetable_data),
        "conflicts": sum(item.hard_conflicts for item in terms),
        "source": "generation-cycle-bundle",
        "cycle": cycle,
        "auto_saved": True,
        "combined_cycle_snapshot": True,
        "solution_rank": selected_solution_rank,
        "resource_penalty": state_metrics.get("resource_penalty"),
        "faculty_preference_penalty": state_metrics.get("faculty_preference_penalty"),
        "workload_gap_penalty": state_metrics.get("workload_gap_penalty"),
    }
    db.add(
        TimetableVersion(
            label=label,
            payload=combined_payload.model_dump(by_alias=True),
            summary=summary,
            created_by_id=current_user.id,
        )
    )
    return label


def _upsert_auto_resolved_conflict(
    *,
    db: Session,
    conflict: object,
    current_user: User,
    resolution_message: str,
    run_label: str,
) -> None:
    decision = db.execute(
        select(TimetableConflictDecision).where(TimetableConflictDecision.conflict_id == conflict.id)
    ).scalar_one_or_none()
    if decision is None:
        decision = TimetableConflictDecision(conflict_id=conflict.id, decision=ConflictDecision.yes, resolved=True)
        db.add(decision)
    snapshot = conflict.model_dump(by_alias=True)
    snapshot["resolved"] = True
    snapshot["resolution"] = resolution_message
    decision.decision = ConflictDecision.yes
    decision.resolved = True
    decision.note = f"[Auto-Resolved {run_label}] {resolution_message}"
    decision.decided_by_id = current_user.id
    decision.conflict_snapshot = snapshot


def _conflict_resolution_priority(conflict: object) -> tuple[int, int, int, str]:
    severity_rank = {
        "high": 0,
        "medium": 1,
        "low": 2,
    }
    type_rank = {
        "section-overlap": 0,
        "faculty-overlap": 1,
        "room-overlap": 2,
        "room-type": 3,
        "working-hours": 4,
        "workload-overflow": 5,
        "course-faculty-inconsistency": 6,
        "practical-block": 7,
        "course-duration": 8,
        "elective-overlap": 9,
        "capacity": 10,
        "availability": 11,
    }
    severity = str(getattr(conflict, "severity", "low") or "low").strip().lower()
    conflict_type = str(getattr(conflict, "type", "") or "").strip()
    affected_slots = list(getattr(conflict, "affected_slots", []) or [])
    return (
        severity_rank.get(severity, 3),
        type_rank.get(conflict_type, 99),
        -len(affected_slots),
        str(getattr(conflict, "id", "") or ""),
    )


def _payload_assignment_signature(payload: object) -> tuple[tuple[str, str, str, str, str, str, str], ...]:
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


def _auto_resolve_generation_conflicts(
    *,
    db: Session,
    generation: GenerateTimetableResponse,
    current_user: User,
    run_label: str,
) -> list[AutoResolvedConflictEntry]:
    if not generation.alternatives:
        return []

    from app.api.routes import timetable as timetable_routes
    from app.schemas.timetable import OfficialTimetablePayload

    resolution_log: list[AutoResolvedConflictEntry] = []
    for alternative in generation.alternatives:
        working_payload = OfficialTimetablePayload.model_validate(alternative.payload.model_dump(by_alias=True))
        max_rounds = min(120, max(20, len(working_payload.timetable_data) * 2))
        round_index = 0
        visited_signatures: set[tuple[tuple[str, str, str, str, str, str, str], ...]] = {
            _payload_assignment_signature(working_payload)
        }
        while round_index < max_rounds:
            round_index += 1
            conflicts = timetable_routes._build_conflicts(working_payload, db)
            if not conflicts:
                break

            ordered_conflicts = sorted(conflicts, key=_conflict_resolution_priority)
            best_candidate: dict | None = None
            for conflict in ordered_conflicts:
                candidate_payload = OfficialTimetablePayload.model_validate(working_payload.model_dump(by_alias=True))
                resolved_payload, resolution_message = timetable_routes._apply_best_effort_resolution(
                    payload=candidate_payload,
                    conflict=conflict,
                    db=db,
                )
                if resolved_payload is None:
                    continue

                post_conflicts = timetable_routes._build_conflicts(resolved_payload, db)
                if len(post_conflicts) > len(conflicts):
                    continue

                candidate_signature = _payload_assignment_signature(resolved_payload)
                if candidate_signature in visited_signatures:
                    continue

                conflict_resolved = not any(item.id == conflict.id for item in post_conflicts)
                improvement = len(conflicts) - len(post_conflicts)
                if improvement <= 0 and not conflict_resolved:
                    continue

                candidate_score = (
                    -improvement,
                    0 if conflict_resolved else 1,
                    len(post_conflicts),
                    _conflict_resolution_priority(conflict),
                )
                if best_candidate is None or candidate_score < best_candidate["score"]:
                    best_candidate = {
                        "score": candidate_score,
                        "payload": resolved_payload,
                        "signature": candidate_signature,
                        "conflict": conflict,
                        "message": resolution_message,
                        "conflict_resolved": conflict_resolved,
                    }

            if best_candidate is None:
                break

            working_payload = best_candidate["payload"]
            visited_signatures.add(best_candidate["signature"])
            if best_candidate["conflict_resolved"]:
                resolved_conflict = best_candidate["conflict"]
                resolution_message = best_candidate["message"]
                log_entry = AutoResolvedConflictEntry(
                    conflict_id=resolved_conflict.id,
                    conflict_type=resolved_conflict.type,
                    description=resolved_conflict.description,
                    resolution=resolution_message,
                    resolved=True,
                )
                resolution_log.append(log_entry)
                _upsert_auto_resolved_conflict(
                    db=db,
                    conflict=resolved_conflict,
                    current_user=current_user,
                    resolution_message=resolution_message,
                    run_label=run_label,
                )

        final_conflicts = timetable_routes._build_conflicts(working_payload, db)
        alternative.payload = working_payload
        detected_conflict_count = len(final_conflicts)
        alternative.hard_conflicts = detected_conflict_count

    unique_log = _dedupe_auto_resolved_conflicts(resolution_log, limit=200)
    generation.auto_resolved_conflicts = unique_log
    return unique_log


def _run_generation(
    *,
    db: Session,
    settings: GenerationSettingsBase,
    payload: GenerateTimetableRequest,
    reserved_resource_slots: list[dict] | None = None,
    progress_reporter: Callable[[dict], None] | None = None,
) -> GenerateTimetableResponse:
    tuned_settings = _runtime_tuned_settings(settings)
    cache_scope = "cycle" if reserved_resource_slots is not None else "single"
    scheduler = EvolutionaryScheduler(
        db=db,
        program_id=payload.program_id,
        term_number=payload.term_number,
        settings=tuned_settings,
        reserved_resource_slots=reserved_resource_slots,
        progress_reporter=progress_reporter,
        hyperparameter_cache_scope=cache_scope,
    )
    return scheduler.run(payload)


def _run_generation_with_optional_progress(
    *,
    db: Session,
    settings: GenerationSettingsBase,
    payload: GenerateTimetableRequest,
    reserved_resource_slots: list[dict] | None = None,
    progress_reporter: Callable[[dict], None] | None = None,
) -> GenerateTimetableResponse:
    if progress_reporter is None:
        return _run_generation(
            db=db,
            settings=settings,
            payload=payload,
            reserved_resource_slots=reserved_resource_slots,
        )
    return _run_generation(
        db=db,
        settings=settings,
        payload=payload,
        reserved_resource_slots=reserved_resource_slots,
        progress_reporter=progress_reporter,
    )


def _reset_db_session_after_long_compute(db: Session) -> None:
    """
    Timetable generation can run for minutes without touching the DB.
    Some Postgres setups (poolers / managed DBs) can drop idle connections, which then
    surface as psycopg OperationalError during post-processing queries.
    Closing the Session here forces a fresh checkout (engine pool_pre_ping) on next use.
    """
    try:
        db.rollback()
    except Exception:
        # Best-effort cleanup; rollback can fail if the connection is already broken.
        pass
    try:
        db.close()
    except Exception:
        pass


def _reload_generation_user(db: Session, user_id: str) -> User:
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User session no longer valid for generation",
        )
    return user


def _runtime_tuned_settings(settings: GenerationSettingsBase) -> GenerationSettingsBase:
    tuned = GenerationSettingsBase.model_validate(settings.model_dump())
    automatic_baseline = default_generation_settings()

    # Hyperparameters are fully automatic at runtime; user-provided knobs are ignored.
    tuned.population_size = automatic_baseline.population_size
    tuned.generations = automatic_baseline.generations
    tuned.mutation_rate = automatic_baseline.mutation_rate
    tuned.crossover_rate = automatic_baseline.crossover_rate
    tuned.elite_count = automatic_baseline.elite_count
    tuned.tournament_size = automatic_baseline.tournament_size
    tuned.stagnation_limit = automatic_baseline.stagnation_limit
    tuned.annealing_iterations = automatic_baseline.annealing_iterations
    tuned.annealing_initial_temperature = automatic_baseline.annealing_initial_temperature
    tuned.annealing_cooling_rate = automatic_baseline.annealing_cooling_rate

    # Keep generation responsive for interactive admin workflows.
    tuned.population_size = min(tuned.population_size, 90)
    tuned.generations = min(tuned.generations, 110)
    tuned.stagnation_limit = min(tuned.stagnation_limit, 40)
    tuned.annealing_iterations = min(tuned.annealing_iterations, 1400)

    if tuned.solver_strategy in {"auto", "hybrid", "genetic"}:
        max_budget = 7_500
        budget = tuned.population_size * tuned.generations
        if budget > max_budget:
            tuned.generations = max(40, max_budget // max(20, tuned.population_size))

    return GenerationSettingsBase.model_validate(tuned.model_dump())


def _persist_payload_as_official(
    *,
    db: Session,
    current_user: User,
    payload_dict: dict,
    summary: dict,
    hard_conflicts: int = 0,
    version_label: str | None = None,
) -> str:
    if hard_conflicts > 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Generated timetable still has hard conflicts; cannot persist as official",
        )

    record = db.get(OfficialTimetable, 1)
    if record is None:
        record = OfficialTimetable(id=1, payload=payload_dict, updated_by_id=current_user.id)
        db.add(record)
    else:
        record.payload = payload_dict
        record.updated_by_id = current_user.id

    resolved_label = version_label or _next_version_label(db)
    version = TimetableVersion(
        label=resolved_label,
        payload=payload_dict,
        summary=summary,
        created_by_id=current_user.id,
    )
    db.add(version)
    return resolved_label


def _persist_generated_official(
    *,
    db: Session,
    current_user: User,
    result: GenerateTimetableResponse,
    version_label: str | None = None,
) -> str:
    best = result.alternatives[0]
    payload_dict = best.payload.model_dump(by_alias=True)
    summary = {
        "program_id": best.payload.program_id,
        "term_number": best.payload.term_number,
        "slots": len(best.payload.timetable_data),
        "conflicts": best.hard_conflicts,
        "source": "generation",
    }
    return _persist_payload_as_official(
        db=db,
        current_user=current_user,
        payload_dict=payload_dict,
        summary=summary,
        hard_conflicts=best.hard_conflicts,
        version_label=version_label,
    )


def _ensure_conflict_free_result(result: GenerateTimetableResponse, *, context: str) -> None:
    best = result.alternatives[0]
    if best.hard_conflicts > 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"{context} could not produce a conflict-free timetable. "
                f"Best candidate still has hard conflicts ({best.hard_conflicts})."
            ),
        )


def _retain_conflict_free_alternatives(result: GenerateTimetableResponse, *, context: str) -> bool:
    conflict_free = [item for item in result.alternatives if item.hard_conflicts == 0]
    if not conflict_free:
        result.alternatives = sorted(
            result.alternatives,
            key=lambda item: (item.hard_conflicts, item.soft_penalty, -item.fitness),
        )
        for index, alternative in enumerate(result.alternatives, start=1):
            alternative.rank = index
        return False
    for index, alternative in enumerate(conflict_free, start=1):
        alternative.rank = index
    result.alternatives = conflict_free
    return True


def _resolve_cycle_term_numbers(
    *,
    db: Session,
    program_id: str,
    cycle: str,
    requested_terms: list[int] | None,
) -> list[int]:
    if cycle == "custom":
        if not requested_terms:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Custom cycle requires term_numbers")
        return sorted(requested_terms)

    configured_terms = sorted(
        db.execute(
            select(ProgramTerm.term_number).where(ProgramTerm.program_id == program_id)
        )
        .scalars()
        .all()
    )
    if not configured_terms:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No program terms configured for this program",
        )

    if cycle == "odd":
        filtered = [item for item in configured_terms if item % 2 == 1]
    elif cycle == "even":
        filtered = [item for item in configured_terms if item % 2 == 0]
    else:
        filtered = configured_terms

    if not filtered:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No configured terms found for cycle '{cycle}'",
        )
    return filtered


def _resolve_default_cycle(db: Session) -> str:
    record = db.get(InstitutionSettings, 1)
    if record is None:
        return DEFAULT_ACADEMIC_CYCLE.semester_cycle
    configured = (record.semester_cycle or "").strip().lower()
    if configured in {"odd", "even"}:
        return configured
    return DEFAULT_ACADEMIC_CYCLE.semester_cycle


def _build_reserved_slots_from_payload(payload: object) -> list[dict]:
    slots: list[dict] = []
    for slot in payload.timetable_data:
        slots.append(
            {
                "day": slot.day,
                "start_time": slot.startTime,
                "end_time": slot.endTime,
                "room_id": slot.roomId,
                "faculty_id": slot.facultyId,
            }
        )
        seen: set[str] = set()
        for assistant_id in _slot_assistant_faculty_ids(slot):
            if assistant_id == slot.facultyId or assistant_id in seen:
                continue
            seen.add(assistant_id)
            slots.append(
                {
                    "day": slot.day,
                    "start_time": slot.startTime,
                    "end_time": slot.endTime,
                    "room_id": None,
                    "faculty_id": assistant_id,
                }
            )
    return slots


def _is_no_feasible_placement_error(exc: HTTPException) -> bool:
    if exc.status_code != status.HTTP_400_BAD_REQUEST:
        return False
    detail = exc.detail
    if isinstance(detail, str):
        return "No feasible placement options" in detail
    if isinstance(detail, dict):
        return "No feasible placement options" in str(detail.get("detail", detail))
    return "No feasible placement options" in str(detail)


def _load_faculty_map(db: Session, *, program_id: str | None = None) -> dict[str, Faculty]:
    statement = select(Faculty)
    if program_id:
        statement = statement.where(Faculty.program_id == program_id)
    return {item.id: item for item in db.execute(statement).scalars().all()}


def _resolve_preferred_codes_for_term(faculty: Faculty, term_number: int | None = None) -> set[str]:
    preferred = {
        str(code).strip().upper()
        for code in (faculty.preferred_subject_codes or [])
        if str(code).strip()
    }
    semester_preferences = faculty.semester_preferences or {}
    if term_number is None:
        for codes in semester_preferences.values():
            preferred.update(str(code).strip().upper() for code in (codes or []) if str(code).strip())
        return preferred

    term_specific = semester_preferences.get(str(term_number), [])
    preferred.update(str(code).strip().upper() for code in (term_specific or []) if str(code).strip())
    return preferred


def _load_faculty_preference_map(
    db: Session,
    term_number: int | None = None,
    *,
    program_id: str | None = None,
) -> dict[str, set[str]]:
    preference_map: dict[str, set[str]] = {}
    statement = select(Faculty)
    if program_id:
        statement = statement.where(Faculty.program_id == program_id)
    faculty_rows = list(db.execute(statement).scalars())
    for faculty in faculty_rows:
        normalized = _resolve_preferred_codes_for_term(faculty, term_number)
        if normalized:
            preference_map[faculty.id] = normalized
    return preference_map


def _faculty_preference_penalty(payload: object, faculty_preference_map: dict[str, set[str]]) -> float:
    course_code_by_id = {
        course.id: course.code.strip().upper()
        for course in payload.course_data
        if course.code and course.code.strip()
    }
    penalty = 0.0
    for slot in payload.timetable_data:
        preferred_codes = faculty_preference_map.get(slot.facultyId)
        if not preferred_codes:
            continue
        course_code = course_code_by_id.get(slot.courseId, "")
        if not course_code or course_code not in preferred_codes:
            penalty += 1.0
    return penalty


def _faculty_target_hours(faculty: Faculty, minimum_hours_per_week: float = 0.0) -> float:
    if faculty.workload_hours > 0:
        return float(max(faculty.workload_hours, minimum_hours_per_week))
    if faculty.max_hours > 0:
        return float(max(faculty.max_hours, minimum_hours_per_week))
    return float(max(0.0, minimum_hours_per_week))


def _load_program_faculty_min_hours(db: Session, program_id: str) -> float:
    row = (
        db.execute(
            select(ProgramConstraint.faculty_min_hours_per_week).where(
                ProgramConstraint.program_id == program_id
            )
        )
        .scalar_one_or_none()
    )
    if row is None:
        return 0.0
    return float(max(0, int(row)))


def _time_ranges_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    return start_a < end_b and start_b < end_a


def _build_workload_gap_suggestions(
    *,
    term_payloads: list[tuple[int, object]],
    faculty_map: dict[str, Faculty],
    minimum_hours_per_week: float = 0.0,
    max_faculty: int = 10,
    max_bridges: int = 6,
) -> list[FacultyWorkloadGapSuggestion]:
    if not term_payloads or not faculty_map:
        return []

    preferred_codes_by_faculty_term = {
        (item.id, term_number): _resolve_preferred_codes_for_term(item, term_number)
        for item in faculty_map.values()
        for term_number, _ in term_payloads
    }

    assigned_minutes_by_faculty: dict[str, int] = defaultdict(int)
    preferred_minutes_by_faculty: dict[str, int] = defaultdict(int)
    occupancy_by_faculty: dict[str, dict[int, dict[str, list[tuple[int, int]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    opportunities: list[dict] = []

    for term_number, payload in term_payloads:
        course_meta = {
            course.id: (course.code.strip().upper(), course.code, course.name)
            for course in payload.course_data
            if course.code and course.code.strip()
        }

        grouped: dict[tuple[int, str, str, str, str], dict] = {}
        for slot in payload.timetable_data:
            start_min = parse_time_to_minutes(slot.startTime)
            end_min = parse_time_to_minutes(slot.endTime)
            minutes = max(0, end_min - start_min)
            if minutes <= 0:
                continue

            assigned_minutes_by_faculty[slot.facultyId] += minutes
            occupancy_by_faculty[slot.facultyId][term_number][slot.day].append((start_min, end_min))

            course_code_upper, course_code_display, course_name = course_meta.get(
                slot.courseId,
                ("", slot.courseId, slot.courseId),
            )
            preferred_for_term = preferred_codes_by_faculty_term.get((slot.facultyId, term_number), set())
            if course_code_upper and course_code_upper in preferred_for_term:
                preferred_minutes_by_faculty[slot.facultyId] += minutes

            key = (
                term_number,
                slot.courseId,
                slot.section,
                slot.batch or "",
                slot.facultyId,
            )
            entry = grouped.get(key)
            if entry is None:
                entry = {
                    "term_number": term_number,
                    "course_id": slot.courseId,
                    "course_code": course_code_display,
                    "course_code_upper": course_code_upper,
                    "course_name": course_name,
                    "section_name": slot.section,
                    "batch": slot.batch,
                    "faculty_id": slot.facultyId,
                    "minutes": 0,
                    "intervals": set(),
                }
                grouped[key] = entry

            entry["minutes"] += minutes
            entry["intervals"].add((slot.day, start_min, end_min))

        opportunities.extend(grouped.values())

    suggestions: list[FacultyWorkloadGapSuggestion] = []
    for faculty in faculty_map.values():
        target_hours = _faculty_target_hours(faculty, minimum_hours_per_week)
        if target_hours <= 0:
            continue

        assigned_minutes = assigned_minutes_by_faculty.get(faculty.id, 0)
        assigned_hours = assigned_minutes / 60.0
        gap_hours = target_hours - assigned_hours
        if gap_hours <= 0.01:
            continue

        preferred_codes = {
            code
            for (faculty_id, _term_number), codes in preferred_codes_by_faculty_term.items()
            if faculty_id == faculty.id
            for code in codes
        }
        if assigned_minutes == 0 and not preferred_codes:
            continue

        preferred_assigned_hours = preferred_minutes_by_faculty.get(faculty.id, 0) / 60.0

        candidate_bridges: list[dict] = []
        for option in opportunities:
            if option["faculty_id"] == faculty.id:
                continue

            intervals = list(option["intervals"])
            term_number = option["term_number"]
            existing_by_day = occupancy_by_faculty.get(faculty.id, {}).get(term_number, {})
            has_overlap = False
            for day, start_min, end_min in intervals:
                for occupied_start, occupied_end in existing_by_day.get(day, []):
                    if _time_ranges_overlap(start_min, end_min, occupied_start, occupied_end):
                        has_overlap = True
                        break
                if has_overlap:
                    break

            course_code_upper = option["course_code_upper"]
            preferred_for_bridge_term = preferred_codes_by_faculty_term.get((faculty.id, term_number), set())
            is_preferred_subject = bool(course_code_upper and course_code_upper in preferred_for_bridge_term)
            candidate_bridges.append(
                {
                    **option,
                    "feasible_without_conflict": not has_overlap,
                    "is_preferred_subject": is_preferred_subject,
                    "weekly_hours": option["minutes"] / 60.0,
                }
            )

        candidate_bridges.sort(
            key=lambda item: (
                not item["feasible_without_conflict"],
                not item["is_preferred_subject"],
                -item["weekly_hours"],
                item["term_number"],
                item["course_code"],
                item["section_name"],
                item["batch"] or "",
            )
        )

        bridge_rows: list[FacultyWorkloadBridgeSuggestion] = []
        covered_hours = 0.0
        for bridge in candidate_bridges:
            bridge_rows.append(
                FacultyWorkloadBridgeSuggestion(
                    term_number=bridge["term_number"],
                    course_id=bridge["course_id"],
                    course_code=bridge["course_code"],
                    course_name=bridge["course_name"],
                    section_name=bridge["section_name"],
                    batch=bridge["batch"],
                    weekly_hours=round(bridge["weekly_hours"], 2),
                    is_preferred_subject=bridge["is_preferred_subject"],
                    feasible_without_conflict=bridge["feasible_without_conflict"],
                )
            )
            if bridge["feasible_without_conflict"]:
                covered_hours += bridge["weekly_hours"]
            if len(bridge_rows) >= max_bridges or covered_hours >= gap_hours:
                break

        suggestions.append(
            FacultyWorkloadGapSuggestion(
                faculty_id=faculty.id,
                faculty_name=faculty.name,
                department=faculty.department,
                target_hours=round(target_hours, 2),
                assigned_hours=round(assigned_hours, 2),
                preferred_assigned_hours=round(preferred_assigned_hours, 2),
                gap_hours=round(max(0.0, gap_hours), 2),
                suggested_bridges=bridge_rows,
            )
        )

    suggestions.sort(
        key=lambda item: (
            -item.gap_hours,
            -(1 if item.suggested_bridges else 0),
            item.faculty_name.lower(),
        )
    )
    return suggestions[:max_faculty]


def _workload_gap_penalty(
    *,
    term_payloads: list[tuple[int, object]],
    faculty_map: dict[str, Faculty],
    minimum_hours_per_week: float = 0.0,
) -> float:
    if not term_payloads or not faculty_map:
        return 0.0

    assigned_minutes_by_faculty: dict[str, int] = defaultdict(int)
    for _term_number, payload in term_payloads:
        for slot in payload.timetable_data:
            start_min = parse_time_to_minutes(slot.startTime)
            end_min = parse_time_to_minutes(slot.endTime)
            if end_min <= start_min:
                continue
            assigned_minutes_by_faculty[slot.facultyId] += end_min - start_min

    total_gap_hours = 0.0
    for faculty in faculty_map.values():
        target_hours = _faculty_target_hours(faculty, minimum_hours_per_week)
        if target_hours <= 0:
            continue
        assigned_hours = assigned_minutes_by_faculty.get(faculty.id, 0) / 60.0
        total_gap_hours += max(0.0, target_hours - assigned_hours)

    return round(total_gap_hours, 3)


def _build_occupancy_matrix(payload: object) -> OccupancyMatrix:
    section_matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    faculty_matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    room_matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for slot in payload.timetable_data:
        slot_key = f"{slot.day}|{slot.startTime}-{slot.endTime}"
        section_matrix[slot.section][slot_key] += 1
        faculty_matrix[slot.facultyId][slot_key] += 1
        room_matrix[slot.roomId][slot_key] += 1

    faculty_labels = {item.id: item.name for item in payload.faculty_data}
    room_labels = {item.id: item.name for item in payload.room_data}

    return OccupancyMatrix(
        section_matrix={section: dict(values) for section, values in section_matrix.items()},
        faculty_matrix={faculty_id: dict(values) for faculty_id, values in faculty_matrix.items()},
        room_matrix={room_id: dict(values) for room_id, values in room_matrix.items()},
        faculty_labels=faculty_labels,
        room_labels=room_labels,
    )


def _attach_occupancy_matrices(generation: GenerateTimetableResponse) -> None:
    for alternative in generation.alternatives:
        alternative.occupancy_matrix = _build_occupancy_matrix(alternative.payload)


def _attach_workload_gap_suggestions(
    *,
    generation: GenerateTimetableResponse,
    term_number: int,
    faculty_map: dict[str, Faculty],
    minimum_hours_per_week: float = 0.0,
) -> None:
    for alternative in generation.alternatives:
        alternative.workload_gap_suggestions = _build_workload_gap_suggestions(
            term_payloads=[(term_number, alternative.payload)],
            faculty_map=faculty_map,
            minimum_hours_per_week=minimum_hours_per_week,
        )


def _cross_term_resource_overlap_count(terms: list[GeneratedCycleSolutionTerm]) -> int:
    room_usage: dict[tuple[str, str, str, str], int] = defaultdict(int)
    faculty_usage: dict[tuple[str, str, str, str], int] = defaultdict(int)

    for term in terms:
        for slot in term.payload.timetable_data:
            room_usage[(slot.day, slot.startTime, slot.endTime, slot.roomId)] += 1
            seen_faculty_ids: set[str] = set()
            primary_faculty_id = str(slot.facultyId or "").strip()
            if primary_faculty_id:
                seen_faculty_ids.add(primary_faculty_id)
                faculty_usage[(slot.day, slot.startTime, slot.endTime, primary_faculty_id)] += 1
            for assistant_id in _slot_assistant_faculty_ids(slot):
                if assistant_id in seen_faculty_ids:
                    continue
                seen_faculty_ids.add(assistant_id)
                faculty_usage[(slot.day, slot.startTime, slot.endTime, assistant_id)] += 1

    room_overlap = sum(max(0, count - 1) for count in room_usage.values())
    faculty_overlap = sum(max(0, count - 1) for count in faculty_usage.values())
    return room_overlap + faculty_overlap


def _dominates(left: dict, right: dict) -> bool:
    return (
        left["resource_penalty"] <= right["resource_penalty"]
        and left["hard_conflicts"] <= right["hard_conflicts"]
        and left["faculty_preference_penalty"] <= right["faculty_preference_penalty"]
        and left["workload_gap_penalty"] <= right["workload_gap_penalty"]
        and (
            left["resource_penalty"] < right["resource_penalty"]
            or left["hard_conflicts"] < right["hard_conflicts"]
            or left["faculty_preference_penalty"] < right["faculty_preference_penalty"]
            or left["workload_gap_penalty"] < right["workload_gap_penalty"]
        )
    )


def _pareto_prune(states: list[dict], *, limit: int) -> list[dict]:
    if not states:
        return []

    frontier: list[dict] = []
    for index, candidate in enumerate(states):
        dominated = False
        for other_index, other in enumerate(states):
            if index == other_index:
                continue
            if _dominates(other, candidate):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)

    frontier.sort(
        key=lambda item: (
            item["resource_penalty"],
            item["hard_conflicts"],
            item["faculty_preference_penalty"],
            item["workload_gap_penalty"],
            item["soft_penalty"],
            item["runtime_ms"],
        )
    )
    return frontier[:limit]


@router.get("/timetable/generation-settings", response_model=GenerationSettingsOut)
def get_generation_settings(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> GenerationSettingsOut:
    return load_generation_settings(db)


@router.put("/timetable/generation-settings", response_model=GenerationSettingsOut)
def update_generation_settings(
    payload: GenerationSettingsUpdate,
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> GenerationSettingsOut:
    record = db.get(TimetableGenerationSettings, 1)
    data = payload.model_dump()
    if record is None:
        record = TimetableGenerationSettings(id=1, **data)
        db.add(record)
    else:
        for key, value in data.items():
            setattr(record, key, value)
    db.commit()
    db.refresh(record)
    return load_generation_settings(db)


@router.get("/timetable/locks", response_model=list[SlotLockOut])
def list_slot_locks(
    program_id: str = Query(min_length=1, max_length=36),
    term_number: int = Query(ge=1, le=20),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[SlotLockOut]:
    return list(
        db.execute(
            select(TimetableSlotLock).where(
                TimetableSlotLock.program_id == program_id,
                TimetableSlotLock.term_number == term_number,
            )
        ).scalars()
    )


@router.post("/timetable/locks", response_model=SlotLockOut, status_code=status.HTTP_201_CREATED)
def create_slot_lock(
    payload: SlotLockCreate,
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> SlotLockOut:
    lock = TimetableSlotLock(**payload.model_dump(), created_by_id=current_user.id)
    db.add(lock)
    db.commit()
    db.refresh(lock)
    return lock


@router.delete("/timetable/locks/{lock_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_slot_lock(
    lock_id: str,
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> None:
    lock = db.get(TimetableSlotLock, lock_id)
    if lock is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Slot lock not found")
    db.delete(lock)
    db.commit()


def _generate_timetable_impl(
    *,
    payload: GenerateTimetableRequest,
    background_tasks: BackgroundTasks | None,
    current_user: User,
    db: Session,
    progress_reporter: Callable[[dict], None] | None = None,
) -> GenerateTimetableResponse:
    started = perf_counter()
    current_user_id = str(current_user.id)
    logger.info(
        "TIMETABLE GENERATION START | user_id=%s | program_id=%s | term=%s | alternatives=%s | persist=%s",
        current_user_id,
        payload.program_id,
        payload.term_number,
        payload.alternative_count,
        payload.persist_official,
    )
    try:
        did_commit = False
        if payload.settings_override is not None:
            settings = payload.settings_override
        else:
            settings = load_generation_settings(db)
        program_min_hours_per_week = _load_program_faculty_min_hours(db, payload.program_id)
        logger.info(
            "TIMETABLE GENERATION STRATEGY | user_id=%s | program_id=%s | term=%s | requested_strategy=%s | effective=moea_sa_auto",
            current_user_id,
            payload.program_id,
            payload.term_number,
            settings.solver_strategy,
        )

        result = _run_generation_with_optional_progress(
            db=db,
            settings=GenerationSettingsBase.model_validate(settings.model_dump()),
            payload=payload,
            progress_reporter=progress_reporter,
        )
        _reset_db_session_after_long_compute(db)
        current_user = _reload_generation_user(db, current_user_id)
        if progress_reporter is not None:
            progress_reporter(
                {
                    "stage": "postprocess",
                    "level": "info",
                    "progress_percent": 94.0,
                    "message": "Applying conflict resolution and validating alternatives.",
                }
            )
        resolved_conflicts = _auto_resolve_generation_conflicts(
            db=db,
            generation=result,
            current_user=current_user,
            run_label=_timestamped_generation_label(
                prefix="gen",
                program_id=payload.program_id,
                term_number=payload.term_number,
            ),
        )
        if progress_reporter is not None:
            progress_reporter(
                {
                    "stage": "postprocess",
                    "level": "info",
                    "progress_percent": 95.0,
                    "message": (
                        "Conflict auto-resolution pass complete: "
                        f"{len(resolved_conflicts)} conflict(s) resolved."
                    ),
                    "metrics": {
                        "resolved_conflicts": len(resolved_conflicts),
                        "best_remaining_hard_conflicts": (
                            result.alternatives[0].hard_conflicts
                            if result.alternatives
                            else None
                        ),
                    },
                }
            )

        has_conflict_free = _retain_conflict_free_alternatives(result, context="Generation")
        if not has_conflict_free:
            logger.warning("TIMETABLE GENERATION | No conflict-free alternatives produced")
            if not result.alternatives:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Scheduler could not produce a conflict-free timetable. Please adjust constraints or resources.",
                )
        if progress_reporter is not None:
            progress_reporter(
                {
                    "stage": "postprocess",
                    "level": "info",
                    "progress_percent": 97.0,
                    "message": "Persisting generated snapshot and enrichment metadata.",
                }
            )
        result.auto_saved_version_label = _persist_generated_snapshot_version(
            db=db,
            current_user=current_user,
            generation=result,
            label_prefix="gen",
            term_number=payload.term_number,
        )
        _attach_occupancy_matrices(result)
        faculty_map = _load_faculty_map(db, program_id=payload.program_id)
        _attach_workload_gap_suggestions(
            generation=result,
            term_number=payload.term_number,
            faculty_map=faculty_map,
            minimum_hours_per_week=program_min_hours_per_week,
        )

        if payload.persist_official:
            try:
                version_label = _persist_generated_official(
                    db=db,
                    current_user=current_user,
                    result=result,
                )
                result.published_version_label = version_label
                log_activity(
                    db,
                    user=current_user,
                    action="timetable.generate.publish",
                    entity_type="official_timetable",
                    entity_id="1",
                    details={
                        "program_id": payload.program_id,
                        "term_number": payload.term_number,
                        "version_label": version_label,
                    },
                )
                db.commit()
                did_commit = True
                try:
                    notify_all_users(
                        db,
                        title="Timetable Updated",
                        message=f"Official timetable updated from generated result ({version_label}).",
                        notification_type=NotificationType.timetable,
                        exclude_user_id=current_user_id,
                        deliver_email=False,
                        background_tasks=background_tasks,
                    )
                    db.commit()
                    did_commit = True
                except Exception:
                    db.rollback()
                    logger.exception(
                        "TIMETABLE GENERATION PUBLISH NOTIFICATION FAILED | user_id=%s | program_id=%s | term=%s | version=%s",
                        current_user_id,
                        payload.program_id,
                        payload.term_number,
                        version_label,
                    )
            except HTTPException as publish_exc:
                publish_detail = str(publish_exc.detail)
                if publish_exc.status_code == status.HTTP_400_BAD_REQUEST and "hard conflicts" in publish_detail.lower():
                    result.publish_warning = publish_detail
                    logger.warning(
                        "TIMETABLE GENERATION PUBLISH SKIPPED | user_id=%s | program_id=%s | term=%s | reason=%s",
                        current_user_id,
                        payload.program_id,
                        payload.term_number,
                        publish_detail,
                    )
                else:
                    raise

        db.commit()
        if progress_reporter is not None:
            progress_reporter(
                {
                    "stage": "finalization",
                    "level": "success",
                    "progress_percent": 100.0,
                    "message": "Generation completed successfully.",
                    "metrics": {
                        "program_id": payload.program_id,
                        "term_number": payload.term_number,
                        "alternatives": len(result.alternatives),
                        "best_hard_conflicts": result.alternatives[0].hard_conflicts if result.alternatives else None,
                        "runtime_ms": result.runtime_ms,
                    },
                }
            )

        elapsed_ms = int((perf_counter() - started) * 1000)
        logger.info(
            "TIMETABLE GENERATION COMPLETE | user_id=%s | program_id=%s | term=%s | alternatives=%s | best_hard_conflicts=%s | runtime_ms=%s | wall_ms=%s",
            current_user_id,
            payload.program_id,
            payload.term_number,
            len(result.alternatives),
            result.alternatives[0].hard_conflicts if result.alternatives else 0,
            result.runtime_ms,
            elapsed_ms,
        )
        return result
    except Exception:
        elapsed_ms = int((perf_counter() - started) * 1000)
        logger.exception(
            "TIMETABLE GENERATION FAILED | user_id=%s | program_id=%s | term=%s | wall_ms=%s",
            current_user_id,
            payload.program_id,
            payload.term_number,
            elapsed_ms,
        )
        raise


def _generate_timetable_cycle_impl(
    *,
    payload: GenerateTimetableCycleRequest,
    background_tasks: BackgroundTasks | None,
    current_user: User,
    db: Session,
    progress_reporter: Callable[[dict], None] | None = None,
) -> GenerateTimetableCycleResponse:
    started = perf_counter()
    current_user_id = str(current_user.id)
    logger.info(
        "TIMETABLE CYCLE GENERATION START | user_id=%s | program_id=%s | cycle=%s | alternatives=%s | pareto_limit=%s | persist=%s",
        current_user_id,
        payload.program_id,
        payload.cycle,
        payload.alternative_count,
        payload.pareto_limit,
        payload.persist_official,
    )
    try:
        did_commit = False
        if payload.settings_override is not None:
            settings = payload.settings_override
        else:
            settings = load_generation_settings(db)
        program_min_hours_per_week = _load_program_faculty_min_hours(db, payload.program_id)
        logger.info(
            "TIMETABLE CYCLE GENERATION STRATEGY | user_id=%s | program_id=%s | cycle=%s | requested_strategy=%s | effective=moea_sa_auto",
            current_user_id,
            payload.program_id,
            payload.cycle,
            settings.solver_strategy,
        )

        resolved_cycle = payload.cycle or _resolve_default_cycle(db)

        term_numbers = _resolve_cycle_term_numbers(
            db=db,
            program_id=payload.program_id,
            cycle=resolved_cycle,
            requested_terms=payload.term_numbers,
        )
        total_terms = max(1, len(term_numbers))
        effective_alternative_count = min(5, max(payload.alternative_count, 3))

        if progress_reporter is not None:
            progress_reporter(
                {
                    "stage": "cycle.initialization",
                    "level": "info",
                    "progress_percent": 0.0,
                    "message": f"Cycle run started for terms: {', '.join(str(term) for term in term_numbers)}.",
                    "metrics": {
                        "cycle": resolved_cycle,
                        "term_numbers": term_numbers,
                        "alternative_count": payload.alternative_count,
                        "effective_alternative_count": effective_alternative_count,
                        "pareto_limit": payload.pareto_limit,
                    },
                }
            )

        faculty_map = _load_faculty_map(db, program_id=payload.program_id)
        initial_state = {
            "terms": [],
            "reserved_slots": [],
            "resource_penalty": 0,
            "cross_term_overlap_penalty": 0,
            "faculty_preference_penalty": 0.0,
            "workload_gap_penalty": 0.0,
            "hard_conflicts": 0,
            "soft_penalty": 0.0,
            "runtime_ms": 0,
            "term_generation_map": {},
        }
        candidate_states = [initial_state]

        for term_position, term_number in enumerate(term_numbers):
            term_preference_map = _load_faculty_preference_map(
                db,
                term_number,
                program_id=payload.program_id,
            )
            expanded_states: list[dict] = []
            state_count = max(1, len(candidate_states))
            for state_index, state in enumerate(candidate_states):
                generation_request = GenerateTimetableRequest(
                    program_id=payload.program_id,
                    term_number=term_number,
                    alternative_count=effective_alternative_count,
                    persist_official=False,
                    settings_override=None,
                )

                def _term_progress(event: dict) -> None:
                    if progress_reporter is None:
                        return
                    event_payload = dict(event)
                    stage = str(event_payload.get("stage") or "search")
                    event_payload["stage"] = f"cycle.term_{term_number}.{stage}"
                    metrics = dict(event_payload.get("metrics") or {})
                    metrics["term_number"] = term_number
                    metrics["state_index"] = state_index
                    metrics["state_number"] = state_index + 1
                    metrics["state_count"] = state_count
                    event_payload["metrics"] = metrics
                    raw_progress = event_payload.get("progress_percent")
                    if isinstance(raw_progress, (int, float)):
                        term_progress = max(0.0, min(100.0, float(raw_progress))) / 100.0
                        state_progress = (state_index + term_progress) / state_count
                        global_progress = ((term_position + state_progress) / total_terms) * 100.0
                        event_payload["progress_percent"] = min(99.5, global_progress)
                    progress_reporter(event_payload)

                run_settings = GenerationSettingsBase.model_validate(settings.model_dump())
                if run_settings.random_seed is not None:
                    run_settings.random_seed = run_settings.random_seed + (term_number * 1000) + state_index

                relaxed_reserved_mode = False
                try:
                    generation_result = _run_generation_with_optional_progress(
                        db=db,
                        settings=run_settings,
                        payload=generation_request,
                        reserved_resource_slots=state["reserved_slots"],
                        progress_reporter=_term_progress if progress_reporter is not None else None,
                    )
                except HTTPException as exc:
                    if not state["reserved_slots"] or not _is_no_feasible_placement_error(exc):
                        raise
                    relaxed_reserved_mode = True
                    logger.warning(
                        "CYCLE TERM FALLBACK | user_id=%s | program_id=%s | term=%s | state_index=%s | reason=%s",
                        current_user_id,
                        payload.program_id,
                        term_number,
                        state_index,
                        exc.detail,
                    )
                    fallback_settings = GenerationSettingsBase.model_validate(run_settings.model_dump())
                    if fallback_settings.random_seed is not None:
                        fallback_settings.random_seed += 17
                    generation_result = _run_generation_with_optional_progress(
                        db=db,
                        settings=fallback_settings,
                        payload=generation_request,
                        reserved_resource_slots=[],
                        progress_reporter=_term_progress if progress_reporter is not None else None,
                    )
                _reset_db_session_after_long_compute(db)
                current_user = _reload_generation_user(db, current_user_id)
                faculty_map = _load_faculty_map(db, program_id=payload.program_id)
                run_label = _timestamped_generation_label(
                    prefix="cycle-run",
                    program_id=payload.program_id,
                    term_number=term_number,
                )
                resolved_conflicts = _auto_resolve_generation_conflicts(
                    db=db,
                    generation=generation_result,
                    current_user=current_user,
                    run_label=run_label,
                )
                if progress_reporter is not None:
                    _term_progress(
                        {
                            "stage": "postprocess",
                            "level": "info",
                            "progress_percent": 96.0,
                            "message": (
                                "Conflict auto-resolution pass complete: "
                                f"{len(resolved_conflicts)} conflict(s) resolved."
                            ),
                            "metrics": {
                                "resolved_conflicts": len(resolved_conflicts),
                                "best_remaining_hard_conflicts": (
                                    generation_result.alternatives[0].hard_conflicts
                                    if generation_result.alternatives
                                    else None
                                ),
                            },
                        }
                    )
                _retain_conflict_free_alternatives(
                    generation_result,
                    context=f"Cycle generation term {term_number}",
                )
                _attach_occupancy_matrices(generation_result)

                viable_alternatives = sorted(
                    generation_result.alternatives,
                    key=lambda item: (item.hard_conflicts, item.soft_penalty, -item.fitness),
                )
                if not viable_alternatives:
                    continue

                for alternative in viable_alternatives:
                    term_solution = GeneratedCycleSolutionTerm(
                        term_number=term_number,
                        alternative_rank=alternative.rank,
                        fitness=alternative.fitness,
                        hard_conflicts=alternative.hard_conflicts,
                        soft_penalty=alternative.soft_penalty,
                        payload=alternative.payload,
                        occupancy_matrix=alternative.occupancy_matrix,
                    )
                    next_terms = [*state["terms"], term_solution]
                    overlap_penalty = _cross_term_resource_overlap_count(next_terms)
                    next_reserved_slots = [
                        *state["reserved_slots"],
                        *_build_reserved_slots_from_payload(alternative.payload),
                    ]
                    next_preference_penalty = state["faculty_preference_penalty"] + _faculty_preference_penalty(
                        alternative.payload,
                        term_preference_map,
                    )
                    next_hard_conflicts = state["hard_conflicts"] + alternative.hard_conflicts
                    next_soft_penalty = state["soft_penalty"] + alternative.soft_penalty
                    next_cross_term_overlap_penalty = overlap_penalty
                    next_resource_penalty = next_cross_term_overlap_penalty
                    if relaxed_reserved_mode:
                        next_resource_penalty += 1
                    next_workload_gap_penalty = _workload_gap_penalty(
                        term_payloads=[(item.term_number, item.payload) for item in next_terms],
                        faculty_map=faculty_map,
                        minimum_hours_per_week=program_min_hours_per_week,
                    )
                    next_runtime_ms = state["runtime_ms"] + generation_result.runtime_ms
                    next_generation_map = {**state["term_generation_map"], term_number: generation_result}

                    expanded_states.append(
                        {
                            "terms": next_terms,
                            "reserved_slots": next_reserved_slots,
                            "resource_penalty": next_resource_penalty,
                            "cross_term_overlap_penalty": next_cross_term_overlap_penalty,
                            "faculty_preference_penalty": next_preference_penalty,
                            "workload_gap_penalty": next_workload_gap_penalty,
                            "hard_conflicts": next_hard_conflicts,
                            "soft_penalty": next_soft_penalty,
                            "runtime_ms": next_runtime_ms,
                            "term_generation_map": next_generation_map,
                        }
                    )

            if not expanded_states:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        f"Cycle generation failed for term {term_number}: "
                        "no conflict-free alternatives could satisfy cross-term resource constraints."
                    ),
                )
            candidate_states = _pareto_prune(expanded_states, limit=payload.pareto_limit)
            if progress_reporter is not None:
                progress_reporter(
                    {
                        "stage": f"cycle.term_{term_number}.frontier",
                        "level": "info",
                        "progress_percent": min(
                            99.7,
                            ((term_position + 1) / total_terms) * 100.0,
                        ),
                        "message": (
                            f"Term {term_number} complete. Pareto candidate states: {len(candidate_states)}."
                        ),
                        "metrics": {
                            "term_number": term_number,
                            "pareto_candidate_states": len(candidate_states),
                        },
                    }
                )

        if not candidate_states:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cycle generation could not produce any feasible Pareto-front alternatives",
            )

        candidate_states.sort(
            key=lambda item: (
                item["resource_penalty"],
                item["hard_conflicts"],
                item["faculty_preference_penalty"],
                item["workload_gap_penalty"],
                item["soft_penalty"],
                item["runtime_ms"],
            )
        )
        selected_state = candidate_states[0]
        pareto_front: list[GeneratedCycleSolution] = []
        for rank, state in enumerate(candidate_states, start=1):
            ordered_terms = sorted(state["terms"], key=lambda item: item.term_number)
            enriched_terms: list[GeneratedCycleSolutionTerm] = []
            for term in ordered_terms:
                term_suggestions = _build_workload_gap_suggestions(
                    term_payloads=[(term.term_number, term.payload)],
                    faculty_map=faculty_map,
                    minimum_hours_per_week=program_min_hours_per_week,
                )
                enriched_terms.append(
                    GeneratedCycleSolutionTerm(
                        term_number=term.term_number,
                        alternative_rank=term.alternative_rank,
                        fitness=term.fitness,
                        hard_conflicts=term.hard_conflicts,
                        soft_penalty=term.soft_penalty,
                        payload=term.payload,
                        workload_gap_suggestions=term_suggestions,
                        occupancy_matrix=term.occupancy_matrix,
                    )
                )
            cycle_suggestions = _build_workload_gap_suggestions(
                term_payloads=[(term.term_number, term.payload) for term in ordered_terms],
                faculty_map=faculty_map,
                minimum_hours_per_week=program_min_hours_per_week,
            )
            pareto_front.append(
                GeneratedCycleSolution(
                    rank=rank,
                    resource_penalty=state["resource_penalty"],
                    faculty_preference_penalty=round(state["faculty_preference_penalty"], 3),
                    workload_gap_penalty=round(state["workload_gap_penalty"], 3),
                    hard_conflicts=state["hard_conflicts"],
                    soft_penalty=round(state["soft_penalty"], 3),
                    runtime_ms=state["runtime_ms"],
                    terms=enriched_terms,
                    workload_gap_suggestions=cycle_suggestions,
                )
            )

        selected_terms = sorted(selected_state["terms"], key=lambda item: item.term_number)

        published_labels_by_term: dict[int, str] = {}
        if payload.persist_official and selected_terms:
            published_labels: list[str] = []
            for term in selected_terms:
                payload_dict = term.payload.model_dump(by_alias=True)
                summary = {
                    "program_id": term.payload.program_id,
                    "term_number": term.term_number,
                    "slots": len(term.payload.timetable_data),
                    "conflicts": term.hard_conflicts,
                    "source": "generation-cycle",
                    "cycle": resolved_cycle,
                    "solution_rank": 1,
                    "term_alternative_rank": term.alternative_rank,
                    "resource_penalty": selected_state["resource_penalty"],
                    "faculty_preference_penalty": round(selected_state["faculty_preference_penalty"], 3),
                    "workload_gap_penalty": round(selected_state["workload_gap_penalty"], 3),
                }
                published_version_label = _persist_payload_as_official(
                    db=db,
                    current_user=current_user,
                    payload_dict=payload_dict,
                    summary=summary,
                    hard_conflicts=term.hard_conflicts,
                )
                published_labels_by_term[term.term_number] = published_version_label
                published_labels.append(published_version_label)

            latest_label = published_labels[-1] if published_labels else "latest"
            log_activity(
                db,
                user=current_user,
                action="timetable.generate.cycle.publish",
                entity_type="official_timetable",
                entity_id="1",
                details={
                    "program_id": payload.program_id,
                    "cycle": resolved_cycle,
                    "term_numbers": term_numbers,
                    "published_versions": published_labels,
                    "selected_solution_rank": 1,
                    "pareto_front_size": len(candidate_states),
                    "workload_gap_penalty": round(selected_state["workload_gap_penalty"], 3),
                },
            )
            db.commit()
            did_commit = True
            try:
                notify_all_users(
                    db,
                    title="Timetable Cycle Updated",
                    message=(
                        f"{resolved_cycle.capitalize()} cycle generated for terms {', '.join(str(item) for item in term_numbers)}. "
                        f"Published Pareto solution #1. Official timetable currently points to term {term_numbers[-1]} ({latest_label})."
                    ),
                    notification_type=NotificationType.timetable,
                    exclude_user_id=current_user_id,
                    deliver_email=False,
                    background_tasks=background_tasks,
                )
                db.commit()
                did_commit = True
            except Exception:
                db.rollback()
                logger.exception(
                    "TIMETABLE CYCLE PUBLISH NOTIFICATION FAILED | user_id=%s | program_id=%s | cycle=%s | terms=%s",
                    current_user_id,
                    payload.program_id,
                    resolved_cycle,
                    ",".join(str(term) for term in term_numbers),
                )

        results: list[GeneratedCycleTermResult] = []
        selected_generation_map = selected_state["term_generation_map"]
        for term_number in term_numbers:
            generation_result = selected_generation_map.get(term_number)
            if generation_result is None:
                continue
            _attach_workload_gap_suggestions(
                generation=generation_result,
                term_number=term_number,
                faculty_map=faculty_map,
                minimum_hours_per_week=program_min_hours_per_week,
            )
            auto_saved_label = _persist_generated_snapshot_version(
                db=db,
                current_user=current_user,
                generation=generation_result,
                label_prefix="cyclegen",
                cycle=resolved_cycle,
                term_number=term_number,
            )
            results.append(
                GeneratedCycleTermResult(
                    term_number=term_number,
                    generation=generation_result,
                    published_version_label=published_labels_by_term.get(term_number),
                    auto_saved_version_label=auto_saved_label,
                )
            )

        combined_cycle_label: str | None = None
        if selected_terms:
            combined_cycle_label = _persist_cycle_combined_snapshot_version(
                db=db,
                current_user=current_user,
                cycle=resolved_cycle,
                terms=selected_terms,
                selected_solution_rank=pareto_front[0].rank if pareto_front else 1,
                state_metrics=selected_state,
            )

        db.commit()

        elapsed_ms = int((perf_counter() - started) * 1000)
        logger.info(
            "TIMETABLE CYCLE GENERATION COMPLETE | user_id=%s | program_id=%s | cycle=%s | terms=%s | pareto=%s | selected_rank=%s | combined_snapshot=%s | wall_ms=%s",
            current_user_id,
            payload.program_id,
            resolved_cycle,
            ",".join(str(term) for term in term_numbers),
            len(pareto_front),
            pareto_front[0].rank if pareto_front else None,
            combined_cycle_label,
            elapsed_ms,
        )
        response = GenerateTimetableCycleResponse(
            program_id=payload.program_id,
            cycle=resolved_cycle,
            term_numbers=term_numbers,
            results=results,
            pareto_front=pareto_front,
            selected_solution_rank=pareto_front[0].rank if pareto_front else None,
        )
        if progress_reporter is not None:
            progress_reporter(
                {
                    "stage": "cycle.finalization",
                    "level": "success",
                    "progress_percent": 100.0,
                    "message": "Cycle MOEA-SA run completed.",
                    "metrics": {
                        "cycle": resolved_cycle,
                        "terms": term_numbers,
                        "pareto_solutions": len(pareto_front),
                        "selected_solution_rank": response.selected_solution_rank,
                    },
                }
            )
        return response
    except Exception:
        elapsed_ms = int((perf_counter() - started) * 1000)
        logger.exception(
            "TIMETABLE CYCLE GENERATION FAILED | user_id=%s | program_id=%s | cycle=%s | wall_ms=%s",
            current_user_id,
            payload.program_id,
            payload.cycle,
            elapsed_ms,
        )
        raise


def _normalize_job_level(raw: str | None) -> str:
    if raw in {"info", "success", "warn", "error"}:
        return raw
    return "info"


def _build_generation_job_status(snapshot: dict) -> GenerationJobStatusOut:
    events = []
    for item in snapshot.get("events", []):
        latest_generation = None
        raw_latest_generation = item.get("latest_generation")
        if isinstance(raw_latest_generation, dict):
            try:
                latest_generation = GenerateTimetableResponse.model_validate(raw_latest_generation)
            except Exception:
                latest_generation = None
        try:
            events.append(
                {
                    "id": int(item.get("id", 0)),
                    "at": item.get("at"),
                    "stage": str(item.get("stage") or "search"),
                    "level": _normalize_job_level(str(item.get("level") or "info")),
                    "message": str(item.get("message") or "Update"),
                    "progress_percent": item.get("progress_percent"),
                    "metrics": item.get("metrics") if isinstance(item.get("metrics"), dict) else {},
                    "latest_generation": latest_generation,
                }
            )
        except Exception:
            continue

    latest_generation = None
    raw_latest_generation = snapshot.get("latest_generation")
    if isinstance(raw_latest_generation, dict):
        try:
            latest_generation = GenerateTimetableResponse.model_validate(raw_latest_generation)
        except Exception:
            latest_generation = None

    result = None
    raw_result = snapshot.get("result")
    if isinstance(raw_result, dict):
        try:
            if snapshot.get("kind") == "cycle":
                result = GenerateTimetableCycleResponse.model_validate(raw_result)
            else:
                result = GenerateTimetableResponse.model_validate(raw_result)
        except Exception:
            result = None

    return GenerationJobStatusOut(
        job_id=snapshot["job_id"],
        kind=snapshot["kind"],
        status=snapshot["status"],
        created_at=snapshot["created_at"],
        started_at=snapshot.get("started_at"),
        finished_at=snapshot.get("finished_at"),
        updated_at=snapshot.get("updated_at") or snapshot["created_at"],
        progress_percent=snapshot.get("progress_percent"),
        stage=snapshot.get("stage"),
        message=snapshot.get("message"),
        events=events,
        last_event_id=int(snapshot.get("last_event_id", 0)),
        latest_generation=latest_generation,
        result=result,
        error_message=snapshot.get("error_message"),
        next_poll_after_ms=1200 if snapshot.get("status") in {"queued", "running"} else 2500,
    )


def _start_background_generation_job(
    *,
    kind: str,
    owner_user: User,
    worker: Callable[[Session, User, Callable[[dict], None]], GenerateTimetableResponse | GenerateTimetableCycleResponse],
) -> GenerationJobAccepted:
    owner_user_id = str(owner_user.id)
    created = generation_job_store.create_job(kind=kind, owner_user_id=owner_user_id)
    job_id = created["job_id"]

    def _runner() -> None:
        db_session = SessionLocal()
        generation_job_store.mark_running(job_id)
        try:
            refreshed_user = db_session.get(User, owner_user_id)
            if refreshed_user is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User session no longer valid for live generation",
                )
            result = worker(db_session, refreshed_user, lambda event: generation_job_store.append_event(job_id, event))
            generation_job_store.append_event(
                job_id,
                {
                    "stage": "job",
                    "level": "success",
                    "message": "Generation job finished and stopped cleanly.",
                    "progress_percent": 100.0,
                    "metrics": {
                        "status": "succeeded",
                    },
                },
            )
            generation_job_store.mark_succeeded(job_id, result.model_dump(by_alias=True))
        except Exception as exc:
            if isinstance(exc, HTTPException):
                error_message = str(exc.detail)
            else:
                error_message = str(exc) or "Live generation job failed"
            generation_job_store.append_event(
                job_id,
                {
                    "stage": "job",
                    "level": "error",
                    "message": error_message,
                    "progress_percent": 100.0,
                },
            )
            generation_job_store.mark_failed(job_id, error_message)
            logger.exception("Generation job failed | job_id=%s | kind=%s", job_id, kind)
        finally:
            db_session.close()

    threading.Thread(target=_runner, daemon=True, name=f"generation-job-{job_id[:8]}").start()

    return GenerationJobAccepted(
        job_id=job_id,
        kind=kind,
        status=created["status"],
        created_at=created["created_at"],
    )


@router.post("/timetable/generate", response_model=GenerateTimetableResponse)
def generate_timetable(
    payload: GenerateTimetableRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> GenerateTimetableResponse:
    return _generate_timetable_impl(
        payload=payload,
        background_tasks=background_tasks,
        current_user=current_user,
        db=db,
    )


@router.post(
    "/timetable/generate/live",
    response_model=GenerationJobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
def generate_timetable_live(
    payload: GenerateTimetableRequest,
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
) -> GenerationJobAccepted:
    return _start_background_generation_job(
        kind="single",
        owner_user=current_user,
        worker=lambda db, user, reporter: _generate_timetable_impl(
            payload=payload,
            background_tasks=None,
            current_user=user,
            db=db,
            progress_reporter=reporter,
        ),
    )


@router.post("/timetable/generate-cycle", response_model=GenerateTimetableCycleResponse)
def generate_timetable_cycle(
    payload: GenerateTimetableCycleRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> GenerateTimetableCycleResponse:
    return _generate_timetable_cycle_impl(
        payload=payload,
        background_tasks=background_tasks,
        current_user=current_user,
        db=db,
    )


@router.post(
    "/timetable/generate-cycle/live",
    response_model=GenerationJobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
def generate_timetable_cycle_live(
    payload: GenerateTimetableCycleRequest,
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
) -> GenerationJobAccepted:
    return _start_background_generation_job(
        kind="cycle",
        owner_user=current_user,
        worker=lambda db, user, reporter: _generate_timetable_cycle_impl(
            payload=payload,
            background_tasks=None,
            current_user=user,
            db=db,
            progress_reporter=reporter,
        ),
    )


@router.get("/timetable/generation-jobs/{job_id}", response_model=GenerationJobStatusOut)
def get_generation_job_status(
    job_id: str,
    since_event_id: int = Query(default=0, ge=0),
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
) -> GenerationJobStatusOut:
    snapshot = generation_job_store.snapshot(job_id, since_event_id=since_event_id)
    if snapshot is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Generation job not found")

    owner_user_id = snapshot.get("owner_user_id")
    if current_user.role != UserRole.admin and owner_user_id != current_user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied for this generation job")

    return _build_generation_job_status(snapshot)


@router.get("/timetable/reevaluation/events", response_model=list[ReevaluationEventOut])
def get_reevaluation_events(
    program_id: str | None = Query(default=None, min_length=1, max_length=36),
    term_number: int | None = Query(default=None, ge=1, le=20),
    event_status: ReevaluationStatus | None = Query(default=None, alias="status"),
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> list[ReevaluationEventOut]:
    rows = list_reevaluation_events(
        db,
        program_id=program_id,
        term_number=term_number,
        status=event_status,
    )
    output: list[ReevaluationEventOut] = []
    for row in rows:
        output.append(
            ReevaluationEventOut(
                id=row.id,
                program_id=row.program_id,
                term_number=row.term_number,
                change_type=row.change_type,
                entity_type=row.entity_type,
                entity_id=row.entity_id,
                description=row.description,
                details=row.details or {},
                status=row.status,
                triggered_by_id=row.triggered_by_id,
                triggered_at=row.triggered_at,
                resolved_by_id=row.resolved_by_id,
                resolved_at=row.resolved_at,
                resolution_note=row.resolution_note,
                has_official_impact=official_scope_impacted(
                    db,
                    program_id=row.program_id,
                    term_number=row.term_number,
                ),
            )
        )
    return output


@router.post("/timetable/reevaluation/run", response_model=ReevaluateTimetableResponse)
def run_curriculum_reevaluation(
    payload: ReevaluateTimetableRequest,
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> ReevaluateTimetableResponse:
    if payload.settings_override is not None:
        settings = payload.settings_override
    else:
        settings = load_generation_settings(db)

    generation_request = GenerateTimetableRequest(
        program_id=payload.program_id,
        term_number=payload.term_number,
        alternative_count=payload.alternative_count,
        persist_official=False,
        settings_override=None,
    )
    generation = _run_generation(
        db=db,
        settings=GenerationSettingsBase.model_validate(settings.model_dump()),
        payload=generation_request,
    )
    _attach_occupancy_matrices(generation)
    faculty_map = _load_faculty_map(db, program_id=payload.program_id)
    program_min_hours_per_week = _load_program_faculty_min_hours(db, payload.program_id)
    _attach_workload_gap_suggestions(
        generation=generation,
        term_number=payload.term_number,
        faculty_map=faculty_map,
        minimum_hours_per_week=program_min_hours_per_week,
    )

    version_label: str | None = None
    if payload.persist_official:
        version_label = _persist_generated_official(
            db=db,
            current_user=current_user,
            result=generation,
        )

    resolved = []
    if payload.mark_resolved:
        resolved = resolve_reevaluation_events(
            db,
            program_id=payload.program_id,
            term_number=payload.term_number,
            resolved_by=current_user,
            resolution_note=payload.resolution_note
            or ("Re-evaluated and published" if payload.persist_official else "Re-evaluated"),
        )
        db.flush()

    pending_count = len(
        list_reevaluation_events(
            db,
            program_id=payload.program_id,
            term_number=payload.term_number,
            status=ReevaluationStatus.pending,
        )
    )

    log_activity(
        db,
        user=current_user,
        action="timetable.reevaluation.run",
        entity_type="program_term",
        entity_id=f"{payload.program_id}:{payload.term_number}",
        details={
            "persist_official": payload.persist_official,
            "resolved_events": len(resolved),
            "pending_events": pending_count,
            "version_label": version_label,
        },
    )

    db.commit()
    if payload.persist_official and version_label:
        try:
            notify_all_users(
                db,
                title="Timetable Re-evaluated",
                message=f"Curriculum-driven re-evaluation published ({version_label}).",
                notification_type=NotificationType.timetable,
                exclude_user_id=current_user.id,
                deliver_email=False,
            )
            db.commit()
        except Exception:
            db.rollback()
            logger.exception(
                "TIMETABLE REEVALUATION PUBLISH NOTIFICATION FAILED | user_id=%s | program_id=%s | term=%s | version=%s",
                current_user.id,
                payload.program_id,
                payload.term_number,
                version_label,
            )
    return ReevaluateTimetableResponse(
        generation=generation,
        resolved_events=len(resolved),
        pending_events=pending_count,
    )
