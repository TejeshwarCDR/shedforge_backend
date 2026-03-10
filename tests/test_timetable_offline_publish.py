from app.api.routes import timetable as timetable_routes


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


def _publish_payload(*, faculty_one_id: str, faculty_two_id: str, faculty_one_email: str, faculty_two_email: str):
    return {
        "facultyData": [
            {
                "id": faculty_one_id,
                "name": "Faculty One",
                "department": "CSE",
                "workloadHours": 0,
                "maxHours": 20,
                "availability": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                "email": faculty_one_email,
            },
            {
                "id": faculty_two_id,
                "name": "Faculty Two",
                "department": "CSE",
                "workloadHours": 0,
                "maxHours": 20,
                "availability": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                "email": faculty_two_email,
            },
        ],
        "courseData": [
            {
                "id": "course-a",
                "code": "CS-A",
                "name": "Course A",
                "type": "theory",
                "credits": 3,
                "facultyId": faculty_one_id,
                "duration": 1,
                "hoursPerWeek": 1,
            },
            {
                "id": "course-b",
                "code": "CS-B",
                "name": "Course B",
                "type": "theory",
                "credits": 3,
                "facultyId": faculty_two_id,
                "duration": 1,
                "hoursPerWeek": 1,
            },
        ],
        "roomData": [
            {
                "id": "room-a",
                "name": "A101",
                "capacity": 70,
                "type": "lecture",
                "building": "Main",
            },
            {
                "id": "room-b",
                "name": "A102",
                "capacity": 70,
                "type": "lecture",
                "building": "Main",
            },
        ],
        "timetableData": [
            {
                "id": "slot-a",
                "day": "Monday",
                "startTime": "08:50",
                "endTime": "09:40",
                "courseId": "course-a",
                "roomId": "room-a",
                "facultyId": faculty_one_id,
                "section": "A",
                "studentCount": 60,
            },
            {
                "id": "slot-b",
                "day": "Monday",
                "startTime": "09:40",
                "endTime": "10:30",
                "courseId": "course-b",
                "roomId": "room-b",
                "facultyId": faculty_two_id,
                "section": "B",
                "studentCount": 60,
            },
        ],
    }


def test_publish_offline_filtered_and_all(client, monkeypatch):
    admin_payload = {
        "name": "Admin Publish",
        "email": "offline-admin@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    faculty_one_payload = {
        "name": "Faculty One",
        "email": "offline-faculty-one@example.com",
        "password": "password123",
        "role": "faculty",
        "department": "CSE",
    }
    faculty_two_payload = {
        "name": "Faculty Two",
        "email": "offline-faculty-two@example.com",
        "password": "password123",
        "role": "faculty",
        "department": "CSE",
    }
    student_a_payload = {
        "name": "Student A",
        "email": "offline-student-a@example.com",
        "password": "password123",
        "role": "student",
        "department": "CSE",
        "section_name": "A",
    }
    student_b_payload = {
        "name": "Student B",
        "email": "offline-student-b@example.com",
        "password": "password123",
        "role": "student",
        "department": "CSE",
        "section_name": "B",
    }

    register_user(client, admin_payload)
    register_user(client, faculty_one_payload)
    register_user(client, faculty_two_payload)
    register_user(client, student_a_payload)
    register_user(client, student_b_payload)

    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")
    faculty_one_token = login_user(client, faculty_one_payload["email"], faculty_one_payload["password"], "faculty")
    faculty_two_token = login_user(client, faculty_two_payload["email"], faculty_two_payload["password"], "faculty")

    faculty_one_profile = client.get(
        "/api/faculty/me",
        headers={"Authorization": f"Bearer {faculty_one_token}"},
    )
    faculty_two_profile = client.get(
        "/api/faculty/me",
        headers={"Authorization": f"Bearer {faculty_two_token}"},
    )
    assert faculty_one_profile.status_code == 200
    assert faculty_two_profile.status_code == 200
    faculty_one_id = faculty_one_profile.json()["id"]
    faculty_two_id = faculty_two_profile.json()["id"]

    publish_response = client.put(
        "/api/timetable/official",
        json=_publish_payload(
            faculty_one_id=faculty_one_id,
            faculty_two_id=faculty_two_id,
            faculty_one_email=faculty_one_payload["email"],
            faculty_two_email=faculty_two_payload["email"],
        ),
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert publish_response.status_code == 200

    sent_to: list[str] = []

    def fake_send_email(*, to_email: str, subject: str, text_content: str, html_content=None):
        sent_to.append(to_email)

    monkeypatch.setattr(timetable_routes, "send_email", fake_send_email)

    filtered_publish_response = client.post(
        "/api/timetable/publish-offline",
        json={"filters": {"sectionName": "A"}},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert filtered_publish_response.status_code == 200
    filtered_payload = filtered_publish_response.json()
    assert filtered_payload["sent"] == 2
    assert filtered_payload["attempted"] == 2
    assert set(filtered_payload["recipients"]) == {
        faculty_one_payload["email"],
        student_a_payload["email"],
    }

    all_publish_response = client.post(
        "/api/timetable/publish-offline/all",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert all_publish_response.status_code == 200
    all_payload = all_publish_response.json()
    assert all_payload["sent"] == 4
    assert all_payload["attempted"] == 4
    assert set(all_payload["recipients"]) == {
        faculty_one_payload["email"],
        faculty_two_payload["email"],
        student_a_payload["email"],
        student_b_payload["email"],
    }
    assert len(sent_to) == 6


def test_publish_offline_requires_admin_or_scheduler(client):
    student_payload = {
        "name": "Student Unauthorized",
        "email": "offline-unauthorized@example.com",
        "password": "password123",
        "role": "student",
        "department": "CSE",
        "section_name": "A",
    }
    register_user(client, student_payload)
    student_token = login_user(client, student_payload["email"], student_payload["password"], "student")

    response = client.post(
        "/api/timetable/publish-offline/all",
        headers={"Authorization": f"Bearer {student_token}"},
    )
    assert response.status_code == 403


def test_official_publish_persists_when_notification_dispatch_fails(client, monkeypatch):
    admin_payload = {
        "name": "Admin Publish Robust",
        "email": "official-publish-robust@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    faculty_payload = {
        "name": "Faculty Robust",
        "email": "official-faculty@example.com",
        "password": "password123",
        "role": "faculty",
        "department": "CSE",
    }
    faculty_two_payload = {
        "name": "Faculty Robust Two",
        "email": "official-faculty-two@example.com",
        "password": "password123",
        "role": "faculty",
        "department": "CSE",
    }
    student_payload = {
        "name": "Student Robust",
        "email": "official-student@example.com",
        "password": "password123",
        "role": "student",
        "department": "CSE",
        "section_name": "A",
    }

    register_user(client, admin_payload)
    register_user(client, faculty_payload)
    register_user(client, faculty_two_payload)
    register_user(client, student_payload)

    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")
    faculty_token = login_user(client, faculty_payload["email"], faculty_payload["password"], "faculty")
    faculty_two_token = login_user(client, faculty_two_payload["email"], faculty_two_payload["password"], "faculty")

    faculty_profile = client.get(
        "/api/faculty/me",
        headers={"Authorization": f"Bearer {faculty_token}"},
    )
    faculty_two_profile = client.get(
        "/api/faculty/me",
        headers={"Authorization": f"Bearer {faculty_two_token}"},
    )
    assert faculty_profile.status_code == 200
    assert faculty_two_profile.status_code == 200
    faculty_id = faculty_profile.json()["id"]
    faculty_two_id = faculty_two_profile.json()["id"]

    def failing_notify_users(*args, **kwargs):
        raise RuntimeError("notification service unavailable")

    def failing_notify_all_users(*args, **kwargs):
        raise RuntimeError("broadcast notification unavailable")

    monkeypatch.setattr(timetable_routes, "notify_users", failing_notify_users)
    monkeypatch.setattr(timetable_routes, "notify_all_users", failing_notify_all_users)

    publish_response = client.put(
        "/api/timetable/official",
        json=_publish_payload(
            faculty_one_id=faculty_id,
            faculty_two_id=faculty_two_id,
            faculty_one_email=faculty_payload["email"],
            faculty_two_email=faculty_two_payload["email"],
        ),
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert publish_response.status_code == 200

    official_response = client.get(
        "/api/timetable/official",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert official_response.status_code == 200
