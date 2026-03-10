
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


def test_program_structure_permissions(client):
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

    create_program = client.post(
        "/api/programs",
        json=program_payload,
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert create_program.status_code == 201
    program_id = create_program.json()["id"]

    term_payload = {"term_number": 1, "name": "Semester 1", "credits_required": 20}

    student_term = client.post(
        f"/api/programs/{program_id}/terms",
        json=term_payload,
        headers={"Authorization": f"Bearer {student_token}"},
    )
    assert student_term.status_code == 403

    admin_term = client.post(
        f"/api/programs/{program_id}/terms",
        json=term_payload,
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert admin_term.status_code == 201
