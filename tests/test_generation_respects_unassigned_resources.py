from app.schemas.generator import ObjectiveWeights


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


def test_generation_skips_real_room_and_faculty_when_not_required(client):
    admin_payload = {
        "name": "Resource Flag Admin",
        "email": "resource-flag-admin@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    register_user(client, admin_payload)
    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")
    auth_headers = {"Authorization": f"Bearer {admin_token}"}

    program_response = client.post(
        "/api/programs",
        json={
            "name": "B.Tech CSE Flags",
            "code": "BTCSE-FLAG",
            "department": "CSE",
            "degree": "BS",
            "duration_years": 4,
            "sections": 1,
            "total_students": 60,
        },
        headers=auth_headers,
    )
    assert program_response.status_code == 201
    program_id = program_response.json()["id"]

    term_response = client.post(
        f"/api/programs/{program_id}/terms",
        json={"term_number": 8, "name": "Semester 8", "credits_required": 3},
        headers=auth_headers,
    )
    assert term_response.status_code == 201

    section_response = client.post(
        f"/api/programs/{program_id}/sections",
        json={"term_number": 8, "name": "A", "capacity": 60},
        headers=auth_headers,
    )
    assert section_response.status_code == 201

    faculty_response = client.post(
        "/api/faculty",
        json={
            "program_id": program_id,
            "name": "Prof Real",
            "email": "prof.real@example.com",
            "designation": "Professor",
            "department": "CSE",
            "workload_hours": 0,
            "max_hours": 20,
            "availability": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
            "availability_windows": [],
        },
        headers=auth_headers,
    )
    assert faculty_response.status_code == 201
    real_faculty_id = faculty_response.json()["id"]

    room_response = client.post(
        "/api/rooms",
        json={
            "program_id": program_id,
            "name": "Real Lab 1",
            "building": "AB2",
            "capacity": 60,
            "type": "lab",
            "has_lab_equipment": True,
            "has_projector": True,
            "availability_windows": [],
        },
        headers=auth_headers,
    )
    assert room_response.status_code == 201
    real_room_id = room_response.json()["id"]

    course_response = client.post(
        "/api/courses",
        json={
            "program_id": program_id,
            "code": "23CSE499",
            "name": "Project - Phase III",
            "type": "lab",
            "credits": 6,
            "duration_hours": 1,
            "sections": 1,
            "hours_per_week": 12,
            "semester_number": 8,
            "batch_year": 4,
            "theory_hours": 0,
            "tutorial_hours": 0,
            "lab_hours": 12,
            "batch_segregation": False,
            "practical_contiguous_slots": 3,
            "assign_faculty": False,
            "assign_classroom": False,
            "faculty_id": real_faculty_id,
            "default_room_id": real_room_id,
        },
        headers=auth_headers,
    )
    assert course_response.status_code == 201
    course_id = course_response.json()["id"]

    mapping_response = client.post(
        f"/api/programs/{program_id}/courses",
        json={
            "term_number": 8,
            "course_id": course_id,
            "is_required": True,
            "lab_batch_count": 1,
            "allow_parallel_batches": True,
        },
        headers=auth_headers,
    )
    assert mapping_response.status_code == 201

    generate_response = client.post(
        "/api/timetable/generate",
        json={
            "program_id": program_id,
            "term_number": 8,
            "alternative_count": 1,
                "settings_override": {
                    "population_size": 20,
                    "generations": 10,
                    "random_seed": 77,
                    "objective_weights": ObjectiveWeights().model_dump(),
                },
            },
        headers=auth_headers,
    )
    assert generate_response.status_code == 200
    body = generate_response.json()
    alternative = body["alternatives"][0]
    payload = alternative["payload"]
    slots = payload["timetableData"]
    assert slots

    assert all(slot["courseId"] == course_id for slot in slots)
    assert all(slot["facultyId"] != real_faculty_id for slot in slots)
    assert all(slot["roomId"] != real_room_id for slot in slots)

    faculty_by_id = {item["id"]: item for item in payload["facultyData"]}
    room_by_id = {item["id"]: item for item in payload["roomData"]}
    for slot in slots:
        faculty_row = faculty_by_id[slot["facultyId"]]
        room_row = room_by_id[slot["roomId"]]
        assert faculty_row["name"] == "No Faculty Required"
        assert room_row["name"] == "No Classroom Required"
