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


def test_semester_constraints_crud_and_enforcement(client):
    admin_payload = {
        "name": "Admin User",
        "email": "admin@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    student_payload = {
        "name": "Student User",
        "email": "student@example.com",
        "password": "password123",
        "role": "student",
        "department": "CS",
        "section_name": "A",
    }

    register_user(client, admin_payload)
    register_user(client, student_payload)

    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")
    student_token = login_user(client, student_payload["email"], student_payload["password"], "student")

    constraint_payload = {
        "term_number": 4,
        "earliest_start_time": "08:50",
        "latest_end_time": "12:25",
        "max_hours_per_day": 3,
        "max_hours_per_week": 3,
        "min_break_minutes": 0,
        "max_consecutive_hours": 1,
    }

    forbidden_response = client.put(
        "/api/constraints/semesters/4",
        json=constraint_payload,
        headers={"Authorization": f"Bearer {student_token}"},
    )
    assert forbidden_response.status_code == 403

    response = client.put(
        "/api/constraints/semesters/4",
        json=constraint_payload,
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert response.status_code == 200

    list_response = client.get(
        "/api/constraints/semesters",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert list_response.status_code == 200
    assert list_response.json()

    base_payload = {
        "facultyData": [
            {
                "id": "f-1",
                "name": "Prof A",
                "department": "CS",
                "workloadHours": 0,
                "maxHours": 10,
                "availability": [],
                "email": "prof@example.com",
            }
        ],
        "courseData": [
            {
                "id": "c-1",
                "code": "CS101",
                "name": "Intro to CS",
                "type": "theory",
                "credits": 3,
                "facultyId": "f-1",
                "duration": 1,
                "hoursPerWeek": 1,
            }
        ],
        "roomData": [
            {
                "id": "r-1",
                "name": "Room 101",
                "capacity": 30,
                "type": "lecture",
                "building": "Main",
            }
        ],
        "timetableData": [
            {
                "id": "ts-1",
                "day": "Monday",
                "startTime": "08:50",
                "endTime": "09:40",
                "courseId": "c-1",
                "roomId": "r-1",
                "facultyId": "f-1",
                "section": "A",
                "studentCount": 20,
            },
            {
                "id": "ts-2",
                "day": "Monday",
                "startTime": "09:40",
                "endTime": "10:30",
                "courseId": "c-1",
                "roomId": "r-1",
                "facultyId": "f-1",
                "section": "A",
                "studentCount": 20,
            },
        ],
    }

    missing_term_response = client.put(
        "/api/timetable/official",
        json=base_payload,
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert missing_term_response.status_code == 400

    invalid_payload = {**base_payload, "termNumber": 4}
    response = client.put(
        "/api/timetable/official",
        json=invalid_payload,
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert response.status_code == 400

    valid_payload = {
        **base_payload,
        "termNumber": 4,
        "timetableData": [
            {
                "id": "ts-3",
                "day": "Monday",
                "startTime": "08:50",
                "endTime": "09:40",
                "courseId": "c-1",
                "roomId": "r-1",
                "facultyId": "f-1",
                "section": "A",
                "studentCount": 20,
            }
        ],
    }

    response = client.put(
        "/api/timetable/official",
        json=valid_payload,
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert response.status_code == 200


def test_semester_weekly_limit_is_validated_per_section_not_global(client):
    admin_payload = {
        "name": "Admin User",
        "email": "admin-weekly-per-section@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    register_user(client, admin_payload)
    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")

    constraint_payload = {
        "term_number": 5,
        "earliest_start_time": "08:50",
        "latest_end_time": "12:25",
        "max_hours_per_day": 2,
        "max_hours_per_week": 1,
        "min_break_minutes": 0,
        "max_consecutive_hours": 2,
    }
    upsert_response = client.put(
        "/api/constraints/semesters/5",
        json=constraint_payload,
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert upsert_response.status_code == 200

    payload = {
        "termNumber": 5,
        "facultyData": [
            {
                "id": "f-1",
                "name": "Prof A",
                "department": "CS",
                "workloadHours": 0,
                "maxHours": 10,
                "availability": [],
                "email": "prof-a@example.com",
            },
            {
                "id": "f-2",
                "name": "Prof B",
                "department": "CS",
                "workloadHours": 0,
                "maxHours": 10,
                "availability": [],
                "email": "prof-b@example.com",
            },
        ],
        "courseData": [
            {
                "id": "c-1",
                "code": "CS101",
                "name": "Intro to CS",
                "type": "theory",
                "credits": 3,
                "facultyId": "f-1",
                "duration": 1,
                "hoursPerWeek": 1,
            },
            {
                "id": "c-2",
                "code": "CS102",
                "name": "Discrete Math",
                "type": "theory",
                "credits": 3,
                "facultyId": "f-2",
                "duration": 1,
                "hoursPerWeek": 1,
            },
        ],
        "roomData": [
            {
                "id": "r-1",
                "name": "Room 101",
                "capacity": 30,
                "type": "lecture",
                "building": "Main",
            },
            {
                "id": "r-2",
                "name": "Room 102",
                "capacity": 30,
                "type": "lecture",
                "building": "Main",
            },
        ],
        "timetableData": [
            {
                "id": "ts-a",
                "day": "Monday",
                "startTime": "08:50",
                "endTime": "09:40",
                "courseId": "c-1",
                "roomId": "r-1",
                "facultyId": "f-1",
                "section": "A",
                "studentCount": 20,
            },
            {
                "id": "ts-b",
                "day": "Monday",
                "startTime": "09:40",
                "endTime": "10:30",
                "courseId": "c-2",
                "roomId": "r-2",
                "facultyId": "f-2",
                "section": "B",
                "studentCount": 20,
            },
        ],
    }

    response = client.put(
        "/api/timetable/official",
        json=payload,
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert response.status_code == 200
