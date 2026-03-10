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


def test_curriculum_change_events_and_selective_reevaluation(client):
    admin_payload = {
        "name": "Admin Reeval",
        "email": "admin-reeval@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    register_user(client, admin_payload)
    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")

    create_program = client.post(
        "/api/programs",
        json={
            "name": "B.Tech CSE",
            "code": "CSE-REEVAL",
            "department": "CSE",
            "degree": "BS",
            "duration_years": 4,
            "sections": 2,
            "total_students": 120,
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert create_program.status_code == 201
    program_id = create_program.json()["id"]

    create_term = client.post(
        f"/api/programs/{program_id}/terms",
        json={
            "term_number": 1,
            "name": "Semester 1",
            "credits_required": 0,
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert create_term.status_code == 201

    section_a = client.post(
        f"/api/programs/{program_id}/sections",
        json={"term_number": 1, "name": "A", "capacity": 60},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert section_a.status_code == 201

    faculty = client.post(
        "/api/faculty",
        json={
            "name": "Faculty Reeval",
            "designation": "Assistant Professor",
            "email": "faculty-reeval@example.com",
            "department": "CSE",
            "workload_hours": 0,
            "max_hours": 20,
            "availability": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
            "availability_windows": [],
            "avoid_back_to_back": False,
            "preferred_min_break_minutes": 0,
            "preference_notes": None,
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert faculty.status_code == 201
    faculty_id = faculty.json()["id"]

    room = client.post(
        "/api/rooms",
        json={
            "name": "A201",
            "building": "Main Block",
            "capacity": 70,
            "type": "lecture",
            "has_lab_equipment": False,
            "has_projector": True,
            "availability_windows": [],
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert room.status_code == 201
    room_id = room.json()["id"]

    course = client.post(
        "/api/courses",
        json={
            "code": "CS101R",
            "name": "Intro Computing",
            "type": "theory",
            "credits": 3,
            "duration_hours": 1,
            "sections": 1,
            "hours_per_week": 1,
            "faculty_id": faculty_id,
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert course.status_code == 201
    course_id = course.json()["id"]

    map_course = client.post(
        f"/api/programs/{program_id}/courses",
        json={
            "term_number": 1,
            "course_id": course_id,
            "is_required": True,
            "prerequisite_course_ids": [],
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert map_course.status_code == 201

    publish_initial = client.put(
        "/api/timetable/official?versionLabel=v-reeval-initial",
        json={
            "programId": program_id,
            "termNumber": 1,
            "facultyData": [
                {
                    "id": faculty_id,
                    "name": "Faculty Reeval",
                    "department": "CSE",
                    "workloadHours": 0,
                    "maxHours": 20,
                    "availability": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                    "email": "faculty-reeval@example.com",
                    "currentWorkload": 0,
                }
            ],
            "courseData": [
                {
                    "id": course_id,
                    "code": "CS101R",
                    "name": "Intro Computing",
                    "type": "theory",
                    "credits": 3,
                    "facultyId": faculty_id,
                    "duration": 1,
                    "hoursPerWeek": 1,
                }
            ],
            "roomData": [
                {
                    "id": room_id,
                    "name": "A201",
                    "capacity": 70,
                    "type": "lecture",
                    "building": "Main Block",
                    "hasLabEquipment": False,
                    "hasProjector": True,
                    "utilization": 0,
                }
            ],
            "timetableData": [
                {
                    "id": "slot-1",
                    "day": "Monday",
                    "startTime": "08:50",
                    "endTime": "09:40",
                    "courseId": course_id,
                    "roomId": room_id,
                    "facultyId": faculty_id,
                    "section": "A",
                    "studentCount": 60,
                }
            ],
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert publish_initial.status_code == 200

    section_b = client.post(
        f"/api/programs/{program_id}/sections",
        json={"term_number": 1, "name": "B", "capacity": 60},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert section_b.status_code == 201

    list_events = client.get(
        f"/api/timetable/reevaluation/events?program_id={program_id}&term_number=1&status=pending",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert list_events.status_code == 200
    pending_events = list_events.json()
    assert pending_events
    assert any(item["has_official_impact"] is True for item in pending_events)

    run_reevaluation = client.post(
        "/api/timetable/reevaluation/run",
        json={
            "program_id": program_id,
            "term_number": 1,
            "alternative_count": 1,
            "persist_official": True,
            "mark_resolved": True,
            "settings_override": {
                "population_size": 20,
                "generations": 20,
                "mutation_rate": 0.12,
                "crossover_rate": 0.8,
                "elite_count": 2,
                "tournament_size": 2,
                "stagnation_limit": 10,
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
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert run_reevaluation.status_code == 200
    reeval_payload = run_reevaluation.json()
    assert reeval_payload["resolved_events"] >= 1
    assert reeval_payload["pending_events"] == 0
    assert reeval_payload["generation"]["alternatives"]

    list_events_after = client.get(
        f"/api/timetable/reevaluation/events?program_id={program_id}&term_number=1&status=pending",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert list_events_after.status_code == 200
    assert list_events_after.json() == []

    versions = client.get(
        "/api/timetable/versions",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert versions.status_code == 200
    assert len(versions.json()) >= 2
