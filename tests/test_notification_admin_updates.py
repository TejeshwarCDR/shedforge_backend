from __future__ import annotations


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


def _system_titles(client, token: str) -> list[str]:
    response = client.get(
        "/api/notifications?notification_type=system",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    return [item["title"] for item in response.json()]


def test_settings_updates_notify_all_non_actor_users_with_email(client, monkeypatch):
    admin_payload = {
        "name": "Admin Settings",
        "email": "admin-settings-notify@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    scheduler_payload = {
        "name": "Scheduler Settings",
        "email": "scheduler-settings-notify@example.com",
        "password": "password123",
        "role": "scheduler",
        "department": "Administration",
    }
    faculty_payload = {
        "name": "Faculty Settings",
        "email": "faculty-settings-notify@example.com",
        "password": "password123",
        "role": "faculty",
        "department": "CSE",
    }
    student_payload = {
        "name": "Student Settings",
        "email": "student-settings-notify@example.com",
        "password": "password123",
        "role": "student",
        "department": "CSE",
        "section_name": "A",
    }
    register_user(client, admin_payload)
    register_user(client, scheduler_payload)
    register_user(client, faculty_payload)
    register_user(client, student_payload)

    sent_to: list[str] = []

    def fake_send_email(*, to_email: str, subject: str, text_content: str, html_content=None):
        sent_to.append(to_email)

    monkeypatch.setattr("app.services.notifications.send_email", fake_send_email)

    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")
    scheduler_token = login_user(client, scheduler_payload["email"], scheduler_payload["password"], "scheduler")
    faculty_token = login_user(client, faculty_payload["email"], faculty_payload["password"], "faculty")
    student_token = login_user(client, student_payload["email"], student_payload["password"], "student")

    update_cycle = client.put(
        "/api/settings/academic-cycle",
        json={"academic_year": "2026-2027", "semester_cycle": "odd"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert update_cycle.status_code == 200

    assert "Academic Cycle Updated" in _system_titles(client, scheduler_token)
    assert "Academic Cycle Updated" in _system_titles(client, faculty_token)
    assert "Academic Cycle Updated" in _system_titles(client, student_token)

    assert set(sent_to) == {
        scheduler_payload["email"],
        faculty_payload["email"],
        student_payload["email"],
    }


def test_academic_data_updates_notify_admin_scheduler_faculty_only(client):
    admin_payload = {
        "name": "Admin Academic",
        "email": "admin-academic-notify@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    scheduler_payload = {
        "name": "Scheduler Academic",
        "email": "scheduler-academic-notify@example.com",
        "password": "password123",
        "role": "scheduler",
        "department": "Administration",
    }
    faculty_payload = {
        "name": "Faculty Academic",
        "email": "faculty-academic-notify@example.com",
        "password": "password123",
        "role": "faculty",
        "department": "CSE",
    }
    student_payload = {
        "name": "Student Academic",
        "email": "student-academic-notify@example.com",
        "password": "password123",
        "role": "student",
        "department": "CSE",
        "section_name": "A",
    }
    register_user(client, admin_payload)
    register_user(client, scheduler_payload)
    register_user(client, faculty_payload)
    register_user(client, student_payload)

    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")
    scheduler_token = login_user(client, scheduler_payload["email"], scheduler_payload["password"], "scheduler")
    faculty_token = login_user(client, faculty_payload["email"], faculty_payload["password"], "faculty")
    student_token = login_user(client, student_payload["email"], student_payload["password"], "student")

    create_program = client.post(
        "/api/programs/",
        json={
            "name": "B.Tech CSE",
            "code": "BTCSE",
            "department": "CSE",
            "degree": "BS",
            "duration_years": 4,
            "sections": 8,
            "total_students": 480,
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert create_program.status_code == 201

    create_room = client.post(
        "/api/rooms/",
        json={
            "name": "A101",
            "building": "Main Block",
            "capacity": 70,
            "type": "lecture",
            "has_lab_equipment": False,
            "has_projector": True,
            "availability_windows": [],
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert create_room.status_code == 201

    create_course = client.post(
        "/api/courses/",
        json={
            "code": "CS200",
            "name": "Algorithms",
            "type": "theory",
            "credits": 4,
            "duration_hours": 1,
            "sections": 1,
            "hours_per_week": 4,
            "semester_number": 4,
            "batch_year": 2,
            "theory_hours": 3,
            "lab_hours": 0,
            "tutorial_hours": 1,
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert create_course.status_code == 201

    upsert_constraint = client.put(
        "/api/constraints/semesters/4",
        json={
            "term_number": 4,
            "earliest_start_time": "08:50",
            "latest_end_time": "16:35",
            "max_hours_per_day": 6,
            "max_hours_per_week": 24,
            "min_break_minutes": 15,
            "max_consecutive_hours": 3,
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert upsert_constraint.status_code == 200

    scheduler_titles = _system_titles(client, scheduler_token)
    faculty_titles = _system_titles(client, faculty_token)
    student_titles = _system_titles(client, student_token)

    expected_titles = {
        "Program Created",
        "Room Added",
        "Course Created",
        "Semester Constraint Added",
    }
    assert expected_titles.issubset(set(scheduler_titles))
    assert expected_titles.issubset(set(faculty_titles))
    assert expected_titles.isdisjoint(set(student_titles))
