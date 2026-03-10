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


def create_program(client, admin_token):
    response = client.post(
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
    assert response.status_code == 201
    return response.json()["id"]


def test_issue_thread_supports_admin_replies_and_status_flow(client):
    admin_payload = {
        "name": "Admin Issues",
        "email": "admin-issues@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    student_payload = {
        "name": "Student Issue Reporter",
        "email": "student-issue@example.com",
        "password": "password123",
        "role": "student",
        "department": "CSE",
        "section_name": "A",
        "semester_number": 5,
    }
    register_user(client, admin_payload)
    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")
    program_id = create_program(client, admin_token)
    student_payload["program_id"] = program_id
    register_user(client, student_payload)
    student_token = login_user(client, student_payload["email"], student_payload["password"], "student")

    create_response = client.post(
        "/api/issues",
        json={
            "category": "conflict",
            "description": "Two classes are overlapping in Section A Tuesday morning.",
            "affected_slot_id": "slot-a-2",
        },
        headers={"Authorization": f"Bearer {student_token}"},
    )
    assert create_response.status_code == 201
    created_issue = create_response.json()
    issue_id = created_issue["id"]
    assert created_issue["message_count"] == 1
    assert created_issue["latest_message_preview"]

    detail_response = client.get(
        f"/api/issues/{issue_id}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert detail_response.status_code == 200
    detail_body = detail_response.json()
    assert len(detail_body["messages"]) == 1
    assert detail_body["messages"][0]["author_role"] == "student"

    admin_reply = client.post(
        f"/api/issues/{issue_id}/messages",
        json={"message": "Acknowledged. We are checking room and faculty overlap now."},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert admin_reply.status_code == 201
    assert admin_reply.json()["author_role"] == "admin"

    updated_to_resolved = client.put(
        f"/api/issues/{issue_id}",
        json={"status": "resolved", "resolution_notes": "Section A classes shifted to slot-a-3."},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert updated_to_resolved.status_code == 200
    assert updated_to_resolved.json()["status"] == "resolved"
    assert updated_to_resolved.json()["resolution_notes"] == "Section A classes shifted to slot-a-3."

    student_follow_up = client.post(
        f"/api/issues/{issue_id}/messages",
        json={"message": "Still seeing overlap on mobile view; please re-check."},
        headers={"Authorization": f"Bearer {student_token}"},
    )
    assert student_follow_up.status_code == 201

    refreshed = client.get(
        f"/api/issues/{issue_id}",
        headers={"Authorization": f"Bearer {student_token}"},
    )
    assert refreshed.status_code == 200
    refreshed_body = refreshed.json()
    assert refreshed_body["status"] == "in_progress"
    assert len(refreshed_body["messages"]) == 3


def test_issue_thread_permissions_are_enforced(client):
    admin_payload = {
        "name": "Admin Issues Guard",
        "email": "admin-issues-guard@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    reporter_payload = {
        "name": "Student Reporter A",
        "email": "student-reporter-a@example.com",
        "password": "password123",
        "role": "student",
        "department": "CSE",
        "section_name": "A",
        "semester_number": 5,
    }
    other_student_payload = {
        "name": "Student Reporter B",
        "email": "student-reporter-b@example.com",
        "password": "password123",
        "role": "student",
        "department": "CSE",
        "section_name": "B",
        "semester_number": 5,
    }
    register_user(client, admin_payload)
    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")
    program_id = create_program(client, admin_token)
    reporter_payload["program_id"] = program_id
    other_student_payload["program_id"] = program_id
    register_user(client, reporter_payload)
    register_user(client, other_student_payload)
    reporter_token = login_user(client, reporter_payload["email"], reporter_payload["password"], "student")
    other_token = login_user(client, other_student_payload["email"], other_student_payload["password"], "student")

    create_response = client.post(
        "/api/issues",
        json={
            "category": "capacity",
            "description": "Classroom does not have enough seats for Section A.",
        },
        headers={"Authorization": f"Bearer {reporter_token}"},
    )
    assert create_response.status_code == 201
    issue_id = create_response.json()["id"]

    forbidden_detail = client.get(
        f"/api/issues/{issue_id}",
        headers={"Authorization": f"Bearer {other_token}"},
    )
    assert forbidden_detail.status_code == 403

    forbidden_reply = client.post(
        f"/api/issues/{issue_id}/messages",
        json={"message": "I should not be able to reply here."},
        headers={"Authorization": f"Bearer {other_token}"},
    )
    assert forbidden_reply.status_code == 403

    admin_detail = client.get(
        f"/api/issues/{issue_id}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert admin_detail.status_code == 200
