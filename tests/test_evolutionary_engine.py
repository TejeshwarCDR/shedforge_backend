from types import SimpleNamespace

from app.models.course import CourseType
from app.models.room import RoomType
from app.schemas.generator import (
    GenerateTimetableRequest,
    GenerateTimetableResponse,
    GeneratedAlternative,
    GenerationSettingsBase,
)
from app.schemas.timetable import OfficialTimetablePayload
from app.services.evolution_scheduler import BlockRequest, EvolutionaryScheduler, PlacementOption, SlotSegment


def register_user(client, payload):
    response = client.post("/api/auth/register", json=payload)
    assert response.status_code == 201
    return response.json()


def login_user(client, email, password, role):
    response = client.post(
        "/api/auth/login",
        json={"email": email, "password": password, "role": role},
    )
    assert response.status_code == 200
    return response.json()["access_token"]


def create_program_with_term_and_sections(client, token, *, code: str, term_number: int, sections: list[tuple[str, int]]):
    program = client.post(
        "/api/programs",
        json={
            "name": f"B.Tech {code}",
            "code": code,
            "department": "CSE",
            "degree": "BS",
            "duration_years": 4,
            "sections": len(sections),
            "total_students": sum(cap for _, cap in sections),
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert program.status_code == 201
    program_id = program.json()["id"]

    term = client.post(
        f"/api/programs/{program_id}/terms",
        json={"term_number": term_number, "name": f"Semester {term_number}", "credits_required": 4},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert term.status_code == 201

    for section_name, capacity in sections:
        section = client.post(
            f"/api/programs/{program_id}/sections",
            json={"term_number": term_number, "name": section_name, "capacity": capacity},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert section.status_code == 201

    return program_id


def test_generation_respects_slot_locks_with_zero_hard_conflicts(client):
    admin_payload = {
        "name": "Engine Admin",
        "email": "engine-admin@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    register_user(client, admin_payload)
    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")

    faculty = client.post(
        "/api/faculty",
        json={
            "name": "Prof Locked",
            "email": "prof.locked@example.com",
            "department": "CSE",
            "workload_hours": 0,
            "max_hours": 20,
            "availability": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
            "availability_windows": [],
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert faculty.status_code == 201
    faculty_id = faculty.json()["id"]

    lecture_room = client.post(
        "/api/rooms",
        json={
            "name": "LH-201",
            "building": "Main",
            "capacity": 70,
            "type": "lecture",
            "has_lab_equipment": False,
            "has_projector": True,
            "availability_windows": [],
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert lecture_room.status_code == 201

    course = client.post(
        "/api/courses",
        json={
            "code": "CS210",
            "name": "Algorithms",
            "type": "theory",
            "credits": 4,
            "duration_hours": 1,
            "sections": 1,
            "hours_per_week": 2,
            "faculty_id": faculty_id,
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert course.status_code == 201
    course_id = course.json()["id"]

    program_id = create_program_with_term_and_sections(
        client,
        admin_token,
        code="CSE-LOCK",
        term_number=1,
        sections=[("A", 60)],
    )

    map_course = client.post(
        f"/api/programs/{program_id}/courses",
        json={
            "term_number": 1,
            "course_id": course_id,
            "is_required": True,
            "lab_batch_count": 1,
            "allow_parallel_batches": True,
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert map_course.status_code == 201

    lock = client.post(
        "/api/timetable/locks",
        json={
            "program_id": program_id,
            "term_number": 1,
            "day": "Monday",
            "start_time": "08:50",
            "end_time": "09:40",
            "section_name": "A",
            "course_id": course_id,
            "batch": None,
            "room_id": None,
            "faculty_id": None,
            "notes": "Must start with locked slot",
            "is_active": True,
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert lock.status_code == 201

    generated = client.post(
        "/api/timetable/generate",
        json={
            "program_id": program_id,
            "term_number": 1,
            "alternative_count": 1,
            "persist_official": False,
            "settings_override": {
                "population_size": 50,
                "generations": 80,
                "mutation_rate": 0.12,
                "crossover_rate": 0.8,
                "elite_count": 4,
                "tournament_size": 3,
                "stagnation_limit": 20,
                "random_seed": 17,
                "objective_weights": {
                    "room_conflict": 400,
                    "faculty_conflict": 400,
                    "section_conflict": 500,
                    "room_capacity": 200,
                    "room_type": 150,
                    "faculty_availability": 180,
                    "locked_slot": 1000,
                    "semester_limit": 200,
                    "workload_overflow": 90,
                    "spread_balance": 20,
                },
            },
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert generated.status_code == 200
    best = generated.json()["alternatives"][0]
    assert best["hard_conflicts"] == 0
    assert any(
        slot["courseId"] == course_id
        and slot["section"] == "A"
        and slot["day"] == "Monday"
        and slot["startTime"] == "08:50"
        and slot["endTime"] == "09:40"
        for slot in best["payload"]["timetableData"]
    )


def test_generation_aligns_shared_lecture_sections(client):
    admin_payload = {
        "name": "Shared Admin",
        "email": "shared-admin@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    register_user(client, admin_payload)
    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")

    faculty = client.post(
        "/api/faculty",
        json={
            "name": "Prof Shared",
            "email": "prof.shared@example.com",
            "department": "CSE",
            "workload_hours": 0,
            "max_hours": 20,
            "availability": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
            "availability_windows": [],
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert faculty.status_code == 201
    faculty_id = faculty.json()["id"]

    room = client.post(
        "/api/rooms",
        json={
            "name": "LH-301",
            "building": "Main",
            "capacity": 80,
            "type": "lecture",
            "has_lab_equipment": False,
            "has_projector": True,
            "availability_windows": [],
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert room.status_code == 201

    course = client.post(
        "/api/courses",
        json={
            "code": "CS301",
            "name": "Systems Design",
            "type": "theory",
            "credits": 3,
            "duration_hours": 1,
            "sections": 2,
            "hours_per_week": 1,
            "faculty_id": faculty_id,
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert course.status_code == 201
    course_id = course.json()["id"]

    program_id = create_program_with_term_and_sections(
        client,
        admin_token,
        code="CSE-SHARE",
        term_number=3,
        sections=[("A", 30), ("B", 30)],
    )

    map_course = client.post(
        f"/api/programs/{program_id}/courses",
        json={
            "term_number": 3,
            "course_id": course_id,
            "is_required": True,
            "lab_batch_count": 1,
            "allow_parallel_batches": True,
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert map_course.status_code == 201

    shared_group = client.post(
        f"/api/programs/{program_id}/shared-lecture-groups",
        json={
            "term_number": 3,
            "name": "CS301 Combined A+B",
            "course_id": course_id,
            "section_names": ["A", "B"],
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert shared_group.status_code == 201

    generated = client.post(
        "/api/timetable/generate",
        json={
            "program_id": program_id,
            "term_number": 3,
            "alternative_count": 1,
            "persist_official": False,
            "settings_override": {
                "population_size": 40,
                "generations": 60,
                "mutation_rate": 0.15,
                "crossover_rate": 0.8,
                "elite_count": 4,
                "tournament_size": 3,
                "stagnation_limit": 20,
                "random_seed": 19,
                "objective_weights": {
                    "room_conflict": 400,
                    "faculty_conflict": 400,
                    "section_conflict": 500,
                    "room_capacity": 200,
                    "room_type": 150,
                    "faculty_availability": 180,
                    "locked_slot": 1000,
                    "semester_limit": 200,
                    "workload_overflow": 90,
                    "spread_balance": 20,
                },
            },
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert generated.status_code == 200
    best = generated.json()["alternatives"][0]
    assert best["hard_conflicts"] == 0

    shared_slots = [
        slot
        for slot in best["payload"]["timetableData"]
        if slot["courseId"] == course_id and slot["section"] in {"A", "B"}
    ]
    assert len(shared_slots) == 2
    signatures = {
        (slot["day"], slot["startTime"], slot["endTime"], slot["roomId"], slot["facultyId"])
        for slot in shared_slots
    }
    assert len(signatures) == 1


def test_generation_fails_fast_when_section_credit_load_exceeds_available_weekly_slots(client):
    admin_payload = {
        "name": "Load Validation Admin",
        "email": "load-validation-admin@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    register_user(client, admin_payload)
    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")

    faculty = client.post(
        "/api/faculty",
        json={
            "name": "Prof Heavy",
            "email": "prof.heavy@example.com",
            "department": "CSE",
            "workload_hours": 0,
            "max_hours": 40,
            "availability": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
            "availability_windows": [],
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert faculty.status_code == 201
    faculty_id = faculty.json()["id"]

    support_faculty_one = client.post(
        "/api/faculty",
        json={
            "name": "Prof Support One",
            "email": "prof.support.one@example.com",
            "department": "CSE",
            "workload_hours": 0,
            "max_hours": 40,
            "availability": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
            "availability_windows": [],
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert support_faculty_one.status_code == 201

    support_faculty_two = client.post(
        "/api/faculty",
        json={
            "name": "Prof Support Two",
            "email": "prof.support.two@example.com",
            "department": "CSE",
            "workload_hours": 0,
            "max_hours": 40,
            "availability": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
            "availability_windows": [],
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert support_faculty_two.status_code == 201

    support_faculty_three = client.post(
        "/api/faculty",
        json={
            "name": "Prof Support Three",
            "email": "prof.support.three@example.com",
            "department": "CSE",
            "workload_hours": 0,
            "max_hours": 40,
            "availability": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
            "availability_windows": [],
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert support_faculty_three.status_code == 201
    support_faculty_three_id = support_faculty_three.json()["id"]

    room = client.post(
        "/api/rooms",
        json={
            "name": "LH-LOAD-101",
            "building": "Main",
            "capacity": 70,
            "type": "lecture",
            "has_lab_equipment": False,
            "has_projector": True,
            "availability_windows": [],
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert room.status_code == 201

    course = client.post(
        "/api/courses",
        json={
            "code": "LOAD401",
            "name": "Impossible Weekly Load",
            "type": "theory",
            "credits": 20,
            "duration_hours": 1,
            "sections": 1,
            "hours_per_week": 40,
            "faculty_id": faculty_id,
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert course.status_code == 201
    course_id = course.json()["id"]

    overflow_course = client.post(
        "/api/courses",
        json={
            "code": "LOAD402",
            "name": "Overflow Weekly Load",
            "type": "theory",
            "credits": 10,
            "duration_hours": 1,
            "sections": 1,
            "hours_per_week": 10,
            "faculty_id": support_faculty_three_id,
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert overflow_course.status_code == 201
    overflow_course_id = overflow_course.json()["id"]

    program = client.post(
        "/api/programs",
        json={
            "name": "B.Tech CSE-LOAD",
            "code": "CSE-LOAD",
            "department": "CSE",
            "degree": "BS",
            "duration_years": 4,
            "sections": 1,
            "total_students": 60,
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert program.status_code == 201
    program_id = program.json()["id"]

    term = client.post(
        f"/api/programs/{program_id}/terms",
        json={"term_number": 1, "name": "Semester 1", "credits_required": 50},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert term.status_code == 201

    section = client.post(
        f"/api/programs/{program_id}/sections",
        json={"term_number": 1, "name": "A", "capacity": 60},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert section.status_code == 201

    map_course = client.post(
        f"/api/programs/{program_id}/courses",
        json={
            "term_number": 1,
            "course_id": course_id,
            "is_required": True,
            "lab_batch_count": 1,
            "allow_parallel_batches": True,
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert map_course.status_code == 201

    map_overflow_course = client.post(
        f"/api/programs/{program_id}/courses",
        json={
            "term_number": 1,
            "course_id": overflow_course_id,
            "is_required": True,
            "lab_batch_count": 1,
            "allow_parallel_batches": True,
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert map_overflow_course.status_code == 201

    generated = client.post(
        "/api/timetable/generate",
        json={
            "program_id": program_id,
            "term_number": 1,
            "alternative_count": 1,
            "persist_official": False,
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert generated.status_code == 400
    assert "weekly credit load exceeds available timetable capacity" in generated.json()["detail"]


def test_auto_strategy_continues_after_conflicted_hybrid_if_annealing_is_conflict_free(monkeypatch):
    scheduler = object.__new__(EvolutionaryScheduler)
    scheduler.settings = GenerationSettingsBase(solver_strategy="auto")
    scheduler.program_id = "program-auto"
    scheduler.term_number = 1

    def make_result(hard_conflicts: int, fitness: float) -> GenerateTimetableResponse:
        return GenerateTimetableResponse(
            alternatives=[
                GeneratedAlternative(
                    rank=1,
                    fitness=fitness,
                    hard_conflicts=hard_conflicts,
                    soft_penalty=float(hard_conflicts),
                    payload=OfficialTimetablePayload(
                        programId="program-auto",
                        termNumber=1,
                        facultyData=[],
                        courseData=[],
                        roomData=[],
                        timetableData=[],
                    ),
                )
            ],
            settings_used=GenerationSettingsBase(),
            runtime_ms=1,
        )

    conflicted_hybrid = make_result(hard_conflicts=3, fitness=-300.0)
    annealed_conflict_free = make_result(hard_conflicts=0, fitness=-10.0)

    call_order: list[str] = []

    def fake_hybrid(_request):
        call_order.append("hybrid")
        return conflicted_hybrid

    def fake_annealing(_request):
        call_order.append("simulated_annealing")
        return annealed_conflict_free

    def fake_genetic(_request):
        call_order.append("genetic")
        return conflicted_hybrid

    monkeypatch.setattr(scheduler, "_run_hybrid_search", fake_hybrid)
    monkeypatch.setattr(scheduler, "_run_simulated_annealing", fake_annealing)
    monkeypatch.setattr(scheduler, "_run_classic_ga", fake_genetic)

    request = GenerateTimetableRequest(
        program_id="program-auto",
        term_number=1,
        alternative_count=1,
        persist_official=False,
    )
    result = EvolutionaryScheduler.run(scheduler, request)

    assert call_order == ["hybrid", "simulated_annealing"]
    assert result.alternatives
    assert result.alternatives[0].hard_conflicts == 0


def _build_scheduler_for_teacher_back_to_back(
    *,
    first_faculty_id: str,
    second_faculty_id: str,
    first_room_id: str,
    second_room_id: str,
    first_section: str,
    second_section: str,
) -> EvolutionaryScheduler:
    scheduler = object.__new__(EvolutionaryScheduler)
    scheduler.settings = GenerationSettingsBase()
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

    scheduler.rooms = {
        "r1": SimpleNamespace(id="r1", capacity=100, type=RoomType.lecture, availability_windows=[]),
        "r2": SimpleNamespace(id="r2", capacity=100, type=RoomType.lecture, availability_windows=[]),
    }
    scheduler.faculty = {
        "f1": SimpleNamespace(id="f1", availability=[], max_hours=20, workload_hours=0),
        "f2": SimpleNamespace(id="f2", availability=[], max_hours=20, workload_hours=0),
    }

    scheduler.block_requests = [
        BlockRequest(
            request_id=0,
            course_id="c1",
            course_code="C1",
            section=first_section,
            batch=None,
            student_count=60,
            primary_faculty_id=first_faculty_id,
            preferred_faculty_ids=(first_faculty_id,),
            block_size=1,
            is_lab=False,
            session_type="theory",
            allow_parallel_batches=False,
            room_candidate_ids=(first_room_id,),
            options=(
                PlacementOption(day="Monday", start_index=0, room_id=first_room_id, faculty_id=first_faculty_id),
            ),
        ),
        BlockRequest(
            request_id=1,
            course_id="c2",
            course_code="C2",
            section=second_section,
            batch=None,
            student_count=60,
            primary_faculty_id=second_faculty_id,
            preferred_faculty_ids=(second_faculty_id,),
            block_size=1,
            is_lab=False,
            session_type="theory",
            allow_parallel_batches=False,
            room_candidate_ids=(second_room_id,),
            options=(
                PlacementOption(day="Monday", start_index=1, room_id=second_room_id, faculty_id=second_faculty_id),
            ),
        ),
    ]
    scheduler.request_indices_by_course = {"c1": [0], "c2": [1]}
    return scheduler


def test_teacher_back_to_back_slots_are_soft_penalized():
    scheduler = _build_scheduler_for_teacher_back_to_back(
        first_faculty_id="f1",
        second_faculty_id="f1",
        first_room_id="r1",
        second_room_id="r2",
        first_section="A",
        second_section="B",
    )

    evaluation = scheduler._evaluate([0, 0])
    conflicted = scheduler._conflicted_request_ids([0, 0])

    assert evaluation.hard_conflicts == 0
    assert evaluation.soft_penalty > 0
    assert conflicted == set()


def test_back_to_back_is_not_applied_to_room_or_section():
    scheduler = _build_scheduler_for_teacher_back_to_back(
        first_faculty_id="f1",
        second_faculty_id="f2",
        first_room_id="r1",
        second_room_id="r1",
        first_section="A",
        second_section="A",
    )

    evaluation = scheduler._evaluate([0, 0])
    conflicted = scheduler._conflicted_request_ids([0, 0])

    assert evaluation.hard_conflicts == 0
    assert conflicted == set()


def _build_scheduler_for_elective_alignment(
    *,
    second_start_index: int,
) -> EvolutionaryScheduler:
    scheduler = object.__new__(EvolutionaryScheduler)
    scheduler.settings = GenerationSettingsBase()
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
    scheduler.courses = {
        "el-1": SimpleNamespace(type=CourseType.elective),
        "el-2": SimpleNamespace(type=CourseType.elective),
    }

    scheduler.rooms = {
        "r1": SimpleNamespace(id="r1", capacity=100, type=RoomType.lecture, availability_windows=[]),
        "r2": SimpleNamespace(id="r2", capacity=100, type=RoomType.lecture, availability_windows=[]),
    }
    scheduler.faculty = {
        "f1": SimpleNamespace(id="f1", availability=[], max_hours=20, workload_hours=0),
        "f2": SimpleNamespace(id="f2", availability=[], max_hours=20, workload_hours=0),
    }

    scheduler.block_requests = [
        BlockRequest(
            request_id=0,
            course_id="el-1",
            course_code="EL1",
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
            ),
        ),
        BlockRequest(
            request_id=1,
            course_id="el-2",
            course_code="EL2",
            section="B",
            batch=None,
            student_count=60,
            primary_faculty_id="f2",
            preferred_faculty_ids=("f2",),
            block_size=1,
            is_lab=False,
            session_type="theory",
            allow_parallel_batches=False,
            room_candidate_ids=("r2",),
            options=(
                PlacementOption(day="Monday", start_index=second_start_index, room_id="r2", faculty_id="f2"),
            ),
        ),
    ]
    scheduler.request_indices_by_course = {"el-1": [0], "el-2": [1]}
    return scheduler


def test_elective_course_must_align_time_slots_across_sections():
    scheduler = _build_scheduler_for_elective_alignment(second_start_index=1)

    evaluation = scheduler._evaluate([0, 0])
    conflicted = scheduler._conflicted_request_ids([0, 0])

    assert evaluation.hard_conflicts >= scheduler.settings.objective_weights.section_conflict
    assert conflicted == {0, 1}


def test_elective_course_same_slot_across_sections_is_valid():
    scheduler = _build_scheduler_for_elective_alignment(second_start_index=0)

    evaluation = scheduler._evaluate([0, 0])
    conflicted = scheduler._conflicted_request_ids([0, 0])

    assert evaluation.hard_conflicts == 0
    assert conflicted == set()


def test_single_faculty_enforcement_relaxes_when_dedicated_capacity_is_insufficient():
    scheduler = object.__new__(EvolutionaryScheduler)
    scheduler.schedule_policy = SimpleNamespace(period_minutes=50)
    scheduler.request_indices_by_course = {"c-1": list(range(10))}
    scheduler.block_requests = [
        BlockRequest(
            request_id=index,
            course_id="c-1",
            course_code="C1",
            section=f"S{index}",
            batch=None,
            student_count=60,
            primary_faculty_id="f-1",
            preferred_faculty_ids=("f-1",),
            block_size=1,
            is_lab=False,
            session_type="theory",
            allow_parallel_batches=False,
            room_candidate_ids=("r-1",),
            options=(PlacementOption(day="Monday", start_index=0, room_id="r-1", faculty_id="f-1"),),
        )
        for index in range(10)
    ]
    scheduler.courses = {"c-1": SimpleNamespace(code="C1", faculty_id="f-1")}
    scheduler.faculty = {"f-1": SimpleNamespace(name="Prof C1", max_hours=2)}

    requirements = scheduler._build_single_faculty_requirements_by_course()

    assert requirements["c-1"] is False
