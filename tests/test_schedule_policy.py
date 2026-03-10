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


def test_schedule_policy_permissions_and_persistence(client):
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
        "department": "CSE",
        "section_name": "A",
    }

    register_user(client, admin_payload)
    register_user(client, student_payload)

    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")
    student_token = login_user(client, student_payload["email"], student_payload["password"], "student")

    payload = {
        "period_minutes": 45,
        "lab_contiguous_slots": 3,
        "breaks": [
            {"name": "Short Break", "start_time": "10:15", "end_time": "10:30"},
            {"name": "Lunch", "start_time": "12:15", "end_time": "13:00"},
        ],
    }

    forbidden_response = client.put(
        "/api/settings/schedule-policy",
        json=payload,
        headers={"Authorization": f"Bearer {student_token}"},
    )
    assert forbidden_response.status_code == 403

    update_response = client.put(
        "/api/settings/schedule-policy",
        json=payload,
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert update_response.status_code == 200
    assert update_response.json()["period_minutes"] == 45
    assert update_response.json()["lab_contiguous_slots"] == 3

    get_response = client.get(
        "/api/settings/schedule-policy",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert get_response.status_code == 200
    assert get_response.json() == update_response.json()


def test_schedule_policy_enforces_lab_slot_blocks(client):
    admin_payload = {
        "name": "Admin User",
        "email": "admin2@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    register_user(client, admin_payload)
    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")

    update_response = client.put(
        "/api/settings/schedule-policy",
        json={
            "period_minutes": 50,
            "lab_contiguous_slots": 2,
            "breaks": [
                {"name": "Short Break", "start_time": "10:30", "end_time": "10:45"},
                {"name": "Lunch Break", "start_time": "12:25", "end_time": "13:15"},
            ],
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert update_response.status_code == 200

    base_payload = {
        "termNumber": 1,
        "facultyData": [
            {
                "id": "f-1",
                "name": "Prof Lab",
                "department": "CSE",
                "workloadHours": 0,
                "maxHours": 10,
                "availability": [],
                "email": "prof@example.com",
            }
        ],
        "courseData": [
            {
                "id": "lab-1",
                "code": "CSL101",
                "name": "Systems Lab",
                "type": "lab",
                "credits": 1,
                "facultyId": "f-1",
                "duration": 2,
                "hoursPerWeek": 2,
            }
        ],
        "roomData": [
            {
                "id": "r-1",
                "name": "LAB-1",
                "capacity": 40,
                "type": "lab",
                "building": "Main",
            }
        ],
    }

    invalid_payload = {
        **base_payload,
        "timetableData": [
            {
                "id": "ts-1",
                "day": "Monday",
                "startTime": "08:50",
                "endTime": "09:40",
                "courseId": "lab-1",
                "roomId": "r-1",
                "facultyId": "f-1",
                "section": "A",
                "batch": "A1",
                "studentCount": 20,
            }
        ],
    }

    invalid_response = client.put(
        "/api/timetable/official",
        json=invalid_payload,
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert invalid_response.status_code == 400

    valid_payload = {
        **base_payload,
        "timetableData": [
            {
                "id": "ts-1",
                "day": "Monday",
                "startTime": "08:50",
                "endTime": "09:40",
                "courseId": "lab-1",
                "roomId": "r-1",
                "facultyId": "f-1",
                "section": "A",
                "batch": "A1",
                "studentCount": 20,
            },
            {
                "id": "ts-2",
                "day": "Monday",
                "startTime": "09:40",
                "endTime": "10:30",
                "courseId": "lab-1",
                "roomId": "r-1",
                "facultyId": "f-1",
                "section": "A",
                "batch": "A1",
                "studentCount": 20,
            },
        ],
    }

    valid_response = client.put(
        "/api/timetable/official",
        json=valid_payload,
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert valid_response.status_code == 200
