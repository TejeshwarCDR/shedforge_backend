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


def test_working_hours_enforced_for_official_timetable(client):
    admin_payload = {
        "name": "Admin User",
        "email": "admin@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    register_user(client, admin_payload)
    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")

    hours_payload = {
        "hours": [
            {"day": "Monday", "start_time": "09:00", "end_time": "10:00", "enabled": True},
            {"day": "Tuesday", "start_time": "09:00", "end_time": "17:00", "enabled": False},
            {"day": "Wednesday", "start_time": "09:00", "end_time": "17:00", "enabled": False},
            {"day": "Thursday", "start_time": "09:00", "end_time": "17:00", "enabled": False},
            {"day": "Friday", "start_time": "09:00", "end_time": "17:00", "enabled": False},
            {"day": "Saturday", "start_time": "09:00", "end_time": "13:00", "enabled": False},
            {"day": "Sunday", "start_time": "09:00", "end_time": "13:00", "enabled": False},
        ]
    }

    response = client.put(
        "/api/settings/working-hours",
        json=hours_payload,
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert response.status_code == 200

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
    }

    invalid_payload = {
        **base_payload,
        "timetableData": [
            {
                "id": "ts-1",
                "day": "Monday",
                "startTime": "08:00",
                "endTime": "09:00",
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
        json=invalid_payload,
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert response.status_code == 400

    valid_payload = {
        **base_payload,
        "timetableData": [
            {
                "id": "ts-2",
                "day": "Monday",
                "startTime": "09:00",
                "endTime": "09:50",
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
