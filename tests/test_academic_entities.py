
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


def test_program_crud_permissions(client):
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
        "department": "Computer Science",
        "section_name": "A",
    }

    register_user(client, admin_payload)
    register_user(client, student_payload)

    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")
    student_token = login_user(client, student_payload["email"], student_payload["password"], "student")

    program_payload = {
        "name": "Computer Science",
        "code": "CS",
        "department": "Computer Science",
        "degree": "BS",
        "duration_years": 4,
        "sections": 4,
        "total_students": 200,
    }

    student_create = client.post(
        "/api/programs",
        json=program_payload,
        headers={"Authorization": f"Bearer {student_token}"},
    )
    assert student_create.status_code == 403

    admin_create = client.post(
        "/api/programs",
        json=program_payload,
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert admin_create.status_code == 201

    list_response = client.get(
        "/api/programs",
        headers={"Authorization": f"Bearer {student_token}"},
    )
    assert list_response.status_code == 200
    assert len(list_response.json()) == 1


def test_students_endpoint_returns_only_students_for_admin_scheduler(client):
    admin_payload = {
        "name": "Admin User",
        "email": "admin-students@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    scheduler_payload = {
        "name": "Scheduler User",
        "email": "scheduler-students@example.com",
        "password": "password123",
        "role": "scheduler",
        "department": "Administration",
    }
    faculty_payload = {
        "name": "Faculty User",
        "email": "faculty-students@example.com",
        "password": "password123",
        "role": "faculty",
        "department": "CSE",
    }
    student_a_payload = {
        "name": "Student Alpha",
        "email": "student-alpha@example.com",
        "password": "password123",
        "role": "student",
        "department": "CSE",
        "section_name": "A",
    }
    student_b_payload = {
        "name": "Student Beta",
        "email": "student-beta@example.com",
        "password": "password123",
        "role": "student",
        "department": "CSE",
        "section_name": "B",
    }

    register_user(client, admin_payload)
    register_user(client, scheduler_payload)
    register_user(client, faculty_payload)
    register_user(client, student_a_payload)
    register_user(client, student_b_payload)

    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")
    scheduler_token = login_user(client, scheduler_payload["email"], scheduler_payload["password"], "scheduler")
    faculty_token = login_user(client, faculty_payload["email"], faculty_payload["password"], "faculty")
    student_token = login_user(client, student_a_payload["email"], student_a_payload["password"], "student")

    admin_response = client.get(
        "/api/students",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert admin_response.status_code == 200
    admin_rows = admin_response.json()
    assert len(admin_rows) == 2
    assert {row["email"] for row in admin_rows} == {
        student_a_payload["email"],
        student_b_payload["email"],
    }

    scheduler_response = client.get(
        "/api/students",
        headers={"Authorization": f"Bearer {scheduler_token}"},
    )
    assert scheduler_response.status_code == 200
    assert len(scheduler_response.json()) == 2

    faculty_response = client.get(
        "/api/students",
        headers={"Authorization": f"Bearer {faculty_token}"},
    )
    assert faculty_response.status_code == 403

    student_response = client.get(
        "/api/students",
        headers={"Authorization": f"Bearer {student_token}"},
    )
    assert student_response.status_code == 403
