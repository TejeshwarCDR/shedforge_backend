import dataclasses
import random
from types import SimpleNamespace
from app.models.room import RoomType
from app.schemas.generator import GenerateTimetableRequest, GenerationSettingsBase, ObjectiveWeights
from app.schemas.timetable import (
    OfficialTimetablePayload, TimeSlotPayload, FacultyPayload, CoursePayload, RoomPayload
)
from app.services.evolution_scheduler import BlockRequest, EvolutionaryScheduler, PlacementOption, SlotSegment

def _build_minimal_scheduler() -> EvolutionaryScheduler:
    scheduler = object.__new__(EvolutionaryScheduler)
    scheduler.settings = GenerationSettingsBase(
        solver_strategy="fast",
        random_seed=42,
        objective_weights=ObjectiveWeights()
    )
    scheduler.eval_cache = {}
    scheduler.schedule_policy = SimpleNamespace(period_minutes=50)
    scheduler.day_slots = {
        "Monday": [
            SlotSegment(start=530, end=580),  # 08:50-09:40
            SlotSegment(start=580, end=630),  # 09:40-10:30
            SlotSegment(start=645, end=695),  # 10:45-11:35
        ]
    }
    scheduler.semester_constraint = None
    scheduler.expected_section_minutes = 0
    scheduler.faculty_windows = {}
    scheduler.room_windows = {}
    scheduler.reserved_resource_slots_by_day = {}
    scheduler.elective_overlap_pairs = set()
    scheduler.shared_lecture_sections_by_course = {}
    scheduler.fixed_genes = {}
    scheduler.courses = {}
    scheduler.random = random.Random(42)

    scheduler.rooms = {
        "r1": SimpleNamespace(id="r1", capacity=100, type=RoomType.lecture, name="R1", availability_windows=[]),
    }
    scheduler.faculty = {
        "f1": SimpleNamespace(id="f1", availability=[], max_hours=20, workload_hours=0, name="F1", email="f1@example.com"),
    }

    scheduler.block_requests = [
        BlockRequest(
            request_id=0,
            course_id="c1",
            course_code="C1",
            section="A",
            batch=None,
            student_count=60,
            primary_faculty_id="f1",
            preferred_faculty_ids=("f1",),
            block_size=1,
            is_lab=False,
            session_type="theory",
            allow_parallel_batches=False,
            room_candidate_ids=("r1",),
            options=(
                PlacementOption(day="Monday", start_index=0, room_id="r1", faculty_id="f1"),
                PlacementOption(day="Monday", start_index=1, room_id="r1", faculty_id="f1"),
            ),
        )
    ]
    scheduler.request_indices_by_course = {"c1": [0]}
    scheduler.option_priority_indices = {0: [0, 1]}

    # Mock more methods needed by _evaluate and others
    scheduler._option_bounds = lambda opt, size: (530, 580) if opt.start_index == 0 else (580, 630)
    scheduler._within_semester_time_window = lambda s, e: True
    scheduler._reserved_conflict_flags = lambda **kwargs: (False, False)
    scheduler._faculty_allows_day = lambda f, d: True
    scheduler._is_elective_request = lambda r: False
    scheduler._is_faculty_back_to_back = lambda *args: False
    scheduler._is_allowed_shared_overlap = lambda *args: False
    scheduler._parallel_lab_overlap_allowed = lambda *args: False
    scheduler._parallel_lab_sync_required = lambda *args: False
    scheduler._intensive_repair_step_cap = lambda: 10
    
    # Needs to mock _decode_payload if we call run()
    def fake_decode(genes):
        slots = []
        course_ids = set()
        faculty_ids = set()
        room_ids = set()
        for i, opt_idx in enumerate(genes):
            req = scheduler.block_requests[i]
            opt = req.options[opt_idx]
            slots.append(TimeSlotPayload(
                id=f"s{i}", day=opt.day, startTime="08:50", endTime="09:40",
                courseId=req.course_id, roomId=opt.room_id, facultyId=opt.faculty_id,
                section=req.section, batch=req.batch, studentCount=req.student_count,
                sessionType=req.session_type
            ))
            course_ids.add(req.course_id)
            faculty_ids.add(opt.faculty_id)
            room_ids.add(opt.room_id)
        
        return OfficialTimetablePayload(
            programId="p1", termNumber=1,
            facultyData=[FacultyPayload(id=fid, name=fid, department="D1", workloadHours=0, maxHours=20, email=f"{fid}@example.com") for fid in faculty_ids],
            courseData=[CoursePayload(id=cid, code=cid, name=cid, type="theory", credits=3, facultyId="f1", duration=1, hoursPerWeek=3) for cid in course_ids],
            roomData=[RoomPayload(id=rid, name=rid, capacity=100, type="lecture", building="Main") for rid in room_ids],
            timetableData=slots
        )
    
    scheduler._decode_payload = fake_decode
    scheduler._payload_fingerprint = lambda p: tuple(p.timetable_data)
    
    # We also need to mock _evaluate since it's used in _run_fast_solver
    from app.services.evolution_scheduler import EvaluationResult
    def fake_evaluate(genes):
        # Very simple evaluation for the mock
        hard = 0
        used_spots = set()
        for i, opt_idx in enumerate(genes):
            req = scheduler.block_requests[i]
            opt = req.options[opt_idx]
            spot = (opt.day, opt.start_index)
            if spot in used_spots:
                hard += 1
            used_spots.add(spot)
        return EvaluationResult(fitness=-float(hard), hard_conflicts=hard, soft_penalty=0.0)
    
    scheduler._evaluate = fake_evaluate
    scheduler._is_better_eval = lambda l, r: l.hard_conflicts < r.hard_conflicts
    
    return scheduler

def test_fast_solver_basic_generation():
    scheduler = _build_minimal_scheduler()
    request = GenerateTimetableRequest(program_id="p1", term_number=1)
    
    scheduler.program_id = "p1"
    scheduler.term_number = 1
    
    response = scheduler._run_fast_solver(request)
    
    assert len(response.alternatives) > 0
    assert response.alternatives[0].hard_conflicts == 0
    assert response.runtime_ms < 500

def test_fast_solver_handles_conflicts():
    scheduler = _build_minimal_scheduler()
    scheduler.block_requests.append(
        BlockRequest(
            request_id=1,
            course_id="c2",
            course_code="C2",
            section="B",
            batch=None,
            student_count=60,
            primary_faculty_id="f1",
            preferred_faculty_ids=("f1",),
            block_size=1,
            is_lab=False,
            session_type="theory",
            allow_parallel_batches=False,
            room_candidate_ids=("r1",),
            options=(
                PlacementOption(day="Monday", start_index=0, room_id="r1", faculty_id="f1"),
            ),
        )
    )
    scheduler.block_requests[0] = dataclasses.replace(
        scheduler.block_requests[0],
        options=(PlacementOption(day="Monday", start_index=0, room_id="r1", faculty_id="f1"),)
    )
    scheduler.option_priority_indices[1] = [0]
    scheduler.option_priority_indices[0] = [0]

    request = GenerateTimetableRequest(program_id="p1", term_number=1)
    response = scheduler._run_fast_solver(request)
    
    assert response.alternatives[0].hard_conflicts > 0

