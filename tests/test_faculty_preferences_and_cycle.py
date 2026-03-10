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


def test_faculty_registration_auto_creates_profile_and_allows_preference_updates(client):
    faculty_payload = {
        "name": "Dr. Faculty User",
        "email": "faculty-user@example.com",
        "password": "password123",
        "role": "faculty",
        "department": "CSE",
        "preferred_subject_codes": ["23cse211", "23CSE302", "23cse211"],
    }
    register_user(client, faculty_payload)
    faculty_token = login_user(client, faculty_payload["email"], faculty_payload["password"], "faculty")

    profile_response = client.get(
        "/api/faculty/me",
        headers={"Authorization": f"Bearer {faculty_token}"},
    )
    assert profile_response.status_code == 200
    profile = profile_response.json()
    assert profile["email"] == faculty_payload["email"]
    assert profile["preferred_subject_codes"] == ["23CSE211", "23CSE302"]

    update_response = client.put(
        f"/api/faculty/{profile['id']}",
        json={
            "max_hours": 16,
            "availability": ["Monday", "Tuesday", "Wednesday"],
            "availability_windows": [
                {"day": "Monday", "start_time": "08:50", "end_time": "12:25"},
                {"day": "Tuesday", "start_time": "08:50", "end_time": "16:35"},
                {"day": "Wednesday", "start_time": "10:45", "end_time": "16:35"},
            ],
            "preferred_subject_codes": ["23cse211", "23CSE302", "23cse211"],
            "preference_notes": "Prefer advanced theory subjects.",
        },
        headers={"Authorization": f"Bearer {faculty_token}"},
    )
    assert update_response.status_code == 200
    updated = update_response.json()
    assert updated["max_hours"] == 16
    assert updated["preferred_subject_codes"] == ["23CSE211", "23CSE302"]

    forbidden_response = client.put(
        f"/api/faculty/{profile['id']}",
        json={"department": "EEE"},
        headers={"Authorization": f"Bearer {faculty_token}"},
    )
    assert forbidden_response.status_code == 403


def test_cycle_generation_avoids_cross_term_faculty_and_room_overlap(client):
    admin_payload = {
        "name": "Cycle Admin",
        "email": "cycle-admin@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    register_user(client, admin_payload)
    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")

    faculty_response = client.post(
        "/api/faculty",
        json={
            "name": "Dr. Shared Faculty",
            "designation": "Associate Professor",
            "email": "shared-faculty@example.com",
            "department": "CSE",
            "workload_hours": 0,
            "max_hours": 20,
            "availability": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
            "availability_windows": [],
            "avoid_back_to_back": False,
            "preferred_min_break_minutes": 0,
            "preference_notes": None,
            "preferred_subject_codes": ["CSE101A", "CSE301A"],
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert faculty_response.status_code == 201
    faculty_id = faculty_response.json()["id"]

    room_response = client.post(
        "/api/rooms",
        json={
            "name": "A101",
            "building": "Main",
            "capacity": 70,
            "type": "lecture",
            "has_lab_equipment": False,
            "has_projector": True,
            "availability_windows": [],
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert room_response.status_code == 201

    program_response = client.post(
        "/api/programs",
        json={
            "name": "B.Tech CSE",
            "code": "CSE",
            "department": "CSE",
            "degree": "BS",
            "duration_years": 4,
            "sections": 2,
            "total_students": 120,
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert program_response.status_code == 201
    program_id = program_response.json()["id"]

    for term_number in (1, 3):
        create_term = client.post(
            f"/api/programs/{program_id}/terms",
            json={"term_number": term_number, "name": f"Semester {term_number}", "credits_required": 3},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert create_term.status_code == 201

        create_section = client.post(
            f"/api/programs/{program_id}/sections",
            json={"term_number": term_number, "name": "A", "capacity": 60},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert create_section.status_code == 201

    course_term_1 = client.post(
        "/api/courses",
        json={
            "code": "CSE101A",
            "name": "Foundations I",
            "type": "theory",
            "credits": 3,
            "duration_hours": 1,
            "sections": 1,
            "hours_per_week": 1,
            "faculty_id": faculty_id,
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert course_term_1.status_code == 201
    course_term_1_id = course_term_1.json()["id"]

    course_term_3 = client.post(
        "/api/courses",
        json={
            "code": "CSE301A",
            "name": "Advanced I",
            "type": "theory",
            "credits": 3,
            "duration_hours": 1,
            "sections": 1,
            "hours_per_week": 1,
            "faculty_id": faculty_id,
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert course_term_3.status_code == 201
    course_term_3_id = course_term_3.json()["id"]

    add_program_course_1 = client.post(
        f"/api/programs/{program_id}/courses",
        json={
            "term_number": 1,
            "course_id": course_term_1_id,
            "is_required": True,
            "lab_batch_count": 1,
            "allow_parallel_batches": True,
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert add_program_course_1.status_code == 201

    add_program_course_3 = client.post(
        f"/api/programs/{program_id}/courses",
        json={
            "term_number": 3,
            "course_id": course_term_3_id,
            "is_required": True,
            "lab_batch_count": 1,
            "allow_parallel_batches": True,
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert add_program_course_3.status_code == 201

    cycle_response = client.post(
        "/api/timetable/generate-cycle",
        json={
            "program_id": program_id,
            "cycle": "odd",
            "alternative_count": 1,
            "persist_official": False,
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert cycle_response.status_code == 200
    payload = cycle_response.json()

    assert payload["term_numbers"] == [1, 3]
    assert len(payload["results"]) == 2
    assert payload["selected_solution_rank"] == 1
    assert payload["pareto_front"]
    assert payload["pareto_front"][0]["rank"] == 1
    assert payload["pareto_front"][0]["resource_penalty"] >= 0
    assert payload["pareto_front"][0]["faculty_preference_penalty"] >= 0
    assert payload["pareto_front"][0]["workload_gap_penalty"] >= 0
    assert isinstance(payload["pareto_front"][0]["workload_gap_suggestions"], list)

    first_term_slot = payload["results"][0]["generation"]["alternatives"][0]["payload"]["timetableData"][0]
    second_term_slot = payload["results"][1]["generation"]["alternatives"][0]["payload"]["timetableData"][0]

    assert (
        first_term_slot["day"],
        first_term_slot["startTime"],
        first_term_slot["endTime"],
        first_term_slot["roomId"],
        first_term_slot["facultyId"],
    ) != (
        second_term_slot["day"],
        second_term_slot["startTime"],
        second_term_slot["endTime"],
        second_term_slot["roomId"],
        second_term_slot["facultyId"],
    )

    selected_solution = payload["pareto_front"][0]
    first_selected_slot = selected_solution["terms"][0]["payload"]["timetableData"][0]
    second_selected_slot = selected_solution["terms"][1]["payload"]["timetableData"][0]
    assert (
        first_selected_slot["day"],
        first_selected_slot["startTime"],
        first_selected_slot["endTime"],
        first_selected_slot["roomId"],
        first_selected_slot["facultyId"],
    ) != (
        second_selected_slot["day"],
        second_selected_slot["startTime"],
        second_selected_slot["endTime"],
        second_selected_slot["roomId"],
        second_selected_slot["facultyId"],
    )
