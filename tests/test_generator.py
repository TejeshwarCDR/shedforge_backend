from fastapi import HTTPException

from app.schemas.generator import GenerateTimetableResponse, GeneratedAlternative, GenerationSettingsBase
from app.schemas.timetable import OfficialTimetablePayload


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


def create_foundation(client, admin_token):
    faculty_one = client.post(
        "/api/faculty",
        json={
            "name": "Prof Theory",
            "email": "theory@example.com",
            "department": "CSE",
            "workload_hours": 0,
            "max_hours": 20,
            "availability": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
            "availability_windows": [],
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert faculty_one.status_code == 201
    faculty_one_id = faculty_one.json()["id"]

    faculty_two = client.post(
        "/api/faculty",
        json={
            "name": "Prof Lab",
            "email": "lab@example.com",
            "department": "CSE",
            "workload_hours": 0,
            "max_hours": 20,
            "availability": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
            "availability_windows": [],
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert faculty_two.status_code == 201
    faculty_two_id = faculty_two.json()["id"]

    lecture_room = client.post(
        "/api/rooms",
        json={
            "name": "LH-101",
            "building": "Main",
            "capacity": 80,
            "type": "lecture",
            "has_lab_equipment": False,
            "has_projector": True,
            "availability_windows": [],
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert lecture_room.status_code == 201

    lab_room = client.post(
        "/api/rooms",
        json={
            "name": "LAB-101",
            "building": "Main",
            "capacity": 40,
            "type": "lab",
            "has_lab_equipment": True,
            "has_projector": True,
            "availability_windows": [],
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert lab_room.status_code == 201
    lab_room_2 = client.post(
        "/api/rooms",
        json={
            "name": "LAB-102",
            "building": "Main",
            "capacity": 40,
            "type": "lab",
            "has_lab_equipment": True,
            "has_projector": True,
            "availability_windows": [],
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert lab_room_2.status_code == 201

    theory_course = client.post(
        "/api/courses",
        json={
            "code": "CS101",
            "name": "Programming Fundamentals",
            "type": "theory",
            "credits": 2,
            "duration_hours": 1,
            "sections": 1,
            "hours_per_week": 2,
            "faculty_id": faculty_one_id,
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert theory_course.status_code == 201
    theory_course_id = theory_course.json()["id"]

    lab_course = client.post(
        "/api/courses",
        json={
            "code": "CSL101",
            "name": "Programming Lab",
            "type": "lab",
            "credits": 2,
            "duration_hours": 2,
            "sections": 1,
            "hours_per_week": 2,
            "faculty_id": faculty_two_id,
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert lab_course.status_code == 201
    lab_course_id = lab_course.json()["id"]

    program = client.post(
        "/api/programs",
        json={
            "name": "B.Tech CSE",
            "code": "CSE",
            "department": "CSE",
            "degree": "BS",
            "duration_years": 4,
            "sections": 1,
            "total_students": 40,
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert program.status_code == 201
    program_id = program.json()["id"]

    term = client.post(
        f"/api/programs/{program_id}/terms",
        json={"term_number": 1, "name": "Semester 1", "credits_required": 5},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert term.status_code == 201

    section = client.post(
        f"/api/programs/{program_id}/sections",
        json={"term_number": 1, "name": "A", "capacity": 40},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert section.status_code == 201

    program_course_theory = client.post(
        f"/api/programs/{program_id}/courses",
        json={
            "term_number": 1,
            "course_id": theory_course_id,
            "is_required": True,
            "lab_batch_count": 1,
            "allow_parallel_batches": True,
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert program_course_theory.status_code == 201

    program_course_lab = client.post(
        f"/api/programs/{program_id}/courses",
        json={
            "term_number": 1,
            "course_id": lab_course_id,
            "is_required": True,
            "lab_batch_count": 2,
            "allow_parallel_batches": True,
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert program_course_lab.status_code == 201

    return {
        "program_id": program_id,
        "theory_course_id": theory_course_id,
    }


def test_generation_settings_and_slot_locks(client):
    admin_payload = {
        "name": "Admin User",
        "email": "admin@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    register_user(client, admin_payload)
    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")

    foundation = create_foundation(client, admin_token)

    settings_response = client.get(
        "/api/timetable/generation-settings",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert settings_response.status_code == 200

    update_response = client.put(
        "/api/timetable/generation-settings",
        json={
            "population_size": 80,
            "generations": 120,
            "mutation_rate": 0.2,
            "crossover_rate": 0.75,
            "elite_count": 6,
            "tournament_size": 4,
            "stagnation_limit": 40,
            "random_seed": 7,
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
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert update_response.status_code == 200
    assert update_response.json()["population_size"] == 80

    create_lock_response = client.post(
        "/api/timetable/locks",
        json={
            "program_id": foundation["program_id"],
            "term_number": 1,
            "day": "Monday",
            "start_time": "08:50",
            "end_time": "09:40",
            "section_name": "A",
            "course_id": foundation["theory_course_id"],
            "batch": None,
            "room_id": None,
            "faculty_id": None,
            "notes": "Pinned opening slot",
            "is_active": True,
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert create_lock_response.status_code == 201
    lock_id = create_lock_response.json()["id"]

    list_lock_response = client.get(
        f"/api/timetable/locks?program_id={foundation['program_id']}&term_number=1",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert list_lock_response.status_code == 200
    assert any(item["id"] == lock_id for item in list_lock_response.json())


def test_timetable_generation_and_publish(client):
    admin_payload = {
        "name": "Admin User",
        "email": "admin2@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    register_user(client, admin_payload)
    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")
    foundation = create_foundation(client, admin_token)

    generate_response = client.post(
        "/api/timetable/generate",
        json={
            "program_id": foundation["program_id"],
            "term_number": 1,
            "alternative_count": 2,
            "persist_official": True,
            "settings_override": {
                "population_size": 60,
                "generations": 80,
                "mutation_rate": 0.15,
                "crossover_rate": 0.8,
                "elite_count": 4,
                "tournament_size": 3,
                "stagnation_limit": 20,
                "random_seed": 21,
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
    assert generate_response.status_code == 200
    data = generate_response.json()
    assert data["alternatives"]
    assert data["alternatives"][0]["payload"]["timetableData"]
    assert isinstance(data["alternatives"][0]["workload_gap_suggestions"], list)

    official_response = client.get(
        "/api/timetable/official",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert official_response.status_code == 200
    assert official_response.json()["programId"] == foundation["program_id"]


def test_timetable_generation_with_simulated_annealing_strategy(client):
    admin_payload = {
        "name": "Admin Anneal",
        "email": "admin-anneal@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    register_user(client, admin_payload)
    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")
    foundation = create_foundation(client, admin_token)

    settings_response = client.put(
        "/api/timetable/generation-settings",
        json={
            "solver_strategy": "simulated_annealing",
            "population_size": 60,
            "generations": 90,
            "mutation_rate": 0.16,
            "crossover_rate": 0.8,
            "elite_count": 4,
            "tournament_size": 3,
            "stagnation_limit": 20,
            "annealing_iterations": 240,
            "annealing_initial_temperature": 5.0,
            "annealing_cooling_rate": 0.992,
            "random_seed": 33,
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
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert settings_response.status_code == 200
    assert settings_response.json()["solver_strategy"] == "simulated_annealing"

    generate_response = client.post(
        "/api/timetable/generate",
        json={
            "program_id": foundation["program_id"],
            "term_number": 1,
            "alternative_count": 1,
            "persist_official": False,
            "settings_override": {
                "solver_strategy": "simulated_annealing",
                "population_size": 60,
                "generations": 90,
                "mutation_rate": 0.16,
                "crossover_rate": 0.8,
                "elite_count": 4,
                "tournament_size": 3,
                "stagnation_limit": 20,
                "annealing_iterations": 240,
                "annealing_initial_temperature": 5.0,
                "annealing_cooling_rate": 0.992,
                "random_seed": 33,
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
    assert generate_response.status_code == 200
    data = generate_response.json()
    assert data["alternatives"]
    assert data["settings_used"]["solver_strategy"] == "simulated_annealing"


def test_timetable_generation_with_stale_assigned_faculty(client):
    admin_payload = {
        "name": "Admin User",
        "email": "admin-stale-faculty@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    register_user(client, admin_payload)
    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")
    foundation = create_foundation(client, admin_token)

    stale_assignment_response = client.put(
        f"/api/courses/{foundation['theory_course_id']}",
        json={"faculty_id": "7130d5e2-f56a-406f-b42f-5b62252240ba"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert stale_assignment_response.status_code == 200

    generate_response = client.post(
        "/api/timetable/generate",
        json={
            "program_id": foundation["program_id"],
            "term_number": 1,
            "alternative_count": 1,
            "persist_official": False,
            "settings_override": {
                "population_size": 40,
                "generations": 20,
                "mutation_rate": 0.15,
                "crossover_rate": 0.8,
                "elite_count": 4,
                "tournament_size": 3,
                "stagnation_limit": 10,
                "random_seed": 55,
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
    assert generate_response.status_code == 200
    body = generate_response.json()
    assert body["alternatives"]
    assert body["alternatives"][0]["payload"]["timetableData"]


def test_generation_uses_feasibility_fallback_when_room_windows_block_all_slots(client):
    admin_payload = {
        "name": "Admin Fallback",
        "email": "admin-fallback-room@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    register_user(client, admin_payload)
    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")
    foundation = create_foundation(client, admin_token)

    rooms_response = client.get(
        "/api/rooms/",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert rooms_response.status_code == 200
    room_rows = rooms_response.json()
    assert room_rows

    for room in room_rows:
        update_response = client.put(
            f"/api/rooms/{room['id']}",
            json={
                "availability_windows": [
                    {"day": "Sunday", "start_time": "08:50", "end_time": "16:35"},
                ]
            },
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert update_response.status_code == 200

    generate_response = client.post(
        "/api/timetable/generate",
        json={
            "program_id": foundation["program_id"],
            "term_number": 1,
            "alternative_count": 1,
            "persist_official": False,
            "settings_override": {
                "population_size": 30,
                "generations": 20,
                "mutation_rate": 0.15,
                "crossover_rate": 0.8,
                "elite_count": 4,
                "tournament_size": 3,
                "stagnation_limit": 10,
                "random_seed": 78,
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
    assert generate_response.status_code == 200
    payload = generate_response.json()
    assert payload["alternatives"]
    assert payload["alternatives"][0]["payload"]["timetableData"]


def test_generate_returns_ranked_candidates_with_warning_when_conflicts_remain(client, monkeypatch):
    admin_payload = {
        "name": "Admin User",
        "email": "admin-conflict@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    register_user(client, admin_payload)
    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")

    def fake_run_generation(*, db, settings, payload, reserved_resource_slots=None):
        return GenerateTimetableResponse(
            alternatives=[
                GeneratedAlternative(
                    rank=1,
                    fitness=-1000.0,
                    hard_conflicts=3,
                    soft_penalty=0.0,
                    payload=OfficialTimetablePayload(
                        programId=payload.program_id,
                        termNumber=payload.term_number,
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

    monkeypatch.setattr("app.api.routes.generator._run_generation", fake_run_generation)

    response = client.post(
        "/api/timetable/generate",
        json={
            "program_id": "program-test",
            "term_number": 1,
            "alternative_count": 1,
            "persist_official": True,
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["alternatives"]
    assert payload["alternatives"][0]["hard_conflicts"] == 3
    assert "publish_warning" in payload
    assert "hard conflicts" in payload["publish_warning"].lower()


def test_generate_filters_out_hard_conflict_alternatives(client, monkeypatch):
    admin_payload = {
        "name": "Admin User",
        "email": "admin-conflict-filter@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    register_user(client, admin_payload)
    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")

    def fake_run_generation(*, db, settings, payload, reserved_resource_slots=None):
        return GenerateTimetableResponse(
            alternatives=[
                GeneratedAlternative(
                    rank=1,
                    fitness=100.0,
                    hard_conflicts=0,
                    soft_penalty=5.0,
                    payload=OfficialTimetablePayload(
                        programId=payload.program_id,
                        termNumber=payload.term_number,
                        facultyData=[],
                        courseData=[],
                        roomData=[],
                        timetableData=[],
                    ),
                ),
                GeneratedAlternative(
                    rank=2,
                    fitness=120.0,
                    hard_conflicts=2,
                    soft_penalty=1.0,
                    payload=OfficialTimetablePayload(
                        programId=payload.program_id,
                        termNumber=payload.term_number,
                        facultyData=[],
                        courseData=[],
                        roomData=[],
                        timetableData=[],
                    ),
                ),
            ],
            settings_used=GenerationSettingsBase(),
            runtime_ms=1,
        )

    monkeypatch.setattr("app.api.routes.generator._run_generation", fake_run_generation)

    response = client.post(
        "/api/timetable/generate",
        json={
            "program_id": "program-test",
            "term_number": 1,
            "alternative_count": 2,
            "persist_official": False,
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["alternatives"]) == 1
    assert payload["alternatives"][0]["hard_conflicts"] == 0
    assert payload["alternatives"][0]["rank"] == 1


def test_cycle_generation_retries_without_reserved_slots_when_strict_mode_is_infeasible(client, monkeypatch):
    admin_payload = {
        "name": "Admin User",
        "email": "admin-cycle-fallback@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    register_user(client, admin_payload)
    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")

    monkeypatch.setattr(
        "app.api.routes.generator._resolve_cycle_term_numbers",
        lambda **kwargs: [1, 3],
    )

    calls: list[tuple[int, bool]] = []

    def fake_run_generation(*, db, settings, payload, reserved_resource_slots=None):
        has_reserved = bool(reserved_resource_slots)
        calls.append((payload.term_number, has_reserved))
        if payload.term_number == 3 and has_reserved:
            raise HTTPException(
                status_code=400,
                detail="No feasible placement options for course DEMO, section A batch B1",
            )

        slot_suffix = "09:40" if payload.term_number == 1 else "10:30"
        return GenerateTimetableResponse(
            alternatives=[
                GeneratedAlternative(
                    rank=1,
                    fitness=-10.0,
                    hard_conflicts=0,
                    soft_penalty=0.0,
                    payload=OfficialTimetablePayload(
                        programId=payload.program_id,
                        termNumber=payload.term_number,
                        facultyData=[
                            {
                                "id": "f1",
                                "name": "Mock Faculty",
                                "department": "CSE",
                                "workloadHours": 0,
                                "maxHours": 20,
                                "availability": ["Monday"],
                                "email": "mock-faculty@example.com",
                            }
                        ],
                        courseData=[
                            {
                                "id": "c1",
                                "code": "MOCK101",
                                "name": "Mock Course",
                                "type": "theory",
                                "credits": 1,
                                "facultyId": "f1",
                                "duration": 1,
                                "hoursPerWeek": 1,
                            }
                        ],
                        roomData=[
                            {
                                "id": "r1",
                                "name": "R1",
                                "capacity": 60,
                                "type": "lecture",
                                "building": "Main",
                            }
                        ],
                        timetableData=[
                            {
                                "id": f"s-{payload.term_number}",
                                "day": "Monday",
                                "startTime": "08:50",
                                "endTime": slot_suffix,
                                "courseId": "c1",
                                "facultyId": "f1",
                                "roomId": "r1",
                                "section": "A",
                                "batch": None,
                                "studentCount": 50,
                            }
                        ],
                    ),
                )
            ],
            settings_used=GenerationSettingsBase(),
            runtime_ms=1,
        )

    monkeypatch.setattr("app.api.routes.generator._run_generation", fake_run_generation)

    response = client.post(
        "/api/timetable/generate-cycle",
        json={
            "program_id": "program-test",
            "cycle": "odd",
            "alternative_count": 1,
            "persist_official": False,
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["term_numbers"] == [1, 3]
    assert len(payload["results"]) == 2
    assert (3, True) in calls
    assert (3, False) in calls


def test_generate_skips_publish_and_returns_warning_when_conflicts_remain(client, monkeypatch):
    admin_payload = {
        "name": "Admin User",
        "email": "admin-publish-warning@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    register_user(client, admin_payload)
    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")

    def fake_run_generation(*, db, settings, payload, reserved_resource_slots=None):
        return GenerateTimetableResponse(
            alternatives=[
                GeneratedAlternative(
                    rank=1,
                    fitness=-1000.0,
                    hard_conflicts=2,
                    soft_penalty=0.0,
                    payload=OfficialTimetablePayload(
                        programId=payload.program_id,
                        termNumber=payload.term_number,
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

    monkeypatch.setattr("app.api.routes.generator._run_generation", fake_run_generation)

    response = client.post(
        "/api/timetable/generate",
        json={
            "program_id": "program-test",
            "term_number": 1,
            "alternative_count": 1,
            "persist_official": True,
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["alternatives"]
    assert payload["alternatives"][0]["hard_conflicts"] == 2
    assert "publish_warning" in payload
    assert "hard conflicts" in payload["publish_warning"].lower()
    assert payload.get("published_version_label") is None


def test_generate_publish_persists_even_if_notification_dispatch_fails(client, monkeypatch):
    admin_payload = {
        "name": "Admin Notify Fail",
        "email": "admin-notify-fail@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    register_user(client, admin_payload)
    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")

    def fake_run_generation(*, db, settings, payload, reserved_resource_slots=None):
        return GenerateTimetableResponse(
            alternatives=[
                GeneratedAlternative(
                    rank=1,
                    fitness=100.0,
                    hard_conflicts=0,
                    soft_penalty=0.0,
                    payload=OfficialTimetablePayload(
                        programId=payload.program_id,
                        termNumber=payload.term_number,
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

    def failing_notify_all_users(*args, **kwargs):
        raise RuntimeError("notification transport unavailable")

    monkeypatch.setattr("app.api.routes.generator._run_generation", fake_run_generation)
    monkeypatch.setattr("app.api.routes.generator.notify_all_users", failing_notify_all_users)

    response = client.post(
        "/api/timetable/generate",
        json={
            "program_id": "program-test",
            "term_number": 1,
            "alternative_count": 1,
            "persist_official": True,
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["published_version_label"]

    official = client.get(
        "/api/timetable/official",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert official.status_code == 200
