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


def test_feedback_submission_routes_to_admin_and_tracks_scope(client):
    admin_payload = {
        "name": "Admin Feedback",
        "email": "admin-feedback@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    student_payload = {
        "name": "Student Reporter",
        "email": "student-feedback@example.com",
        "password": "password123",
        "role": "student",
        "department": "CSE",
        "section_name": "A",
    }
    faculty_payload = {
        "name": "Faculty Viewer",
        "email": "faculty-feedback@example.com",
        "password": "password123",
        "role": "faculty",
        "department": "CSE",
    }
    register_user(client, admin_payload)
    register_user(client, student_payload)
    register_user(client, faculty_payload)

    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")
    student_token = login_user(client, student_payload["email"], student_payload["password"], "student")
    faculty_token = login_user(client, faculty_payload["email"], faculty_payload["password"], "faculty")

    create_response = client.post(
        "/api/feedback",
        json={
            "subject": "Unable to view updated schedule",
            "category": "technical",
            "priority": "high",
            "message": "After login, my timetable card remains blank on mobile.",
        },
        headers={"Authorization": f"Bearer {student_token}"},
    )
    assert create_response.status_code == 201
    feedback_id = create_response.json()["id"]

    admin_list = client.get(
        "/api/feedback",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert admin_list.status_code == 200
    assert any(item["id"] == feedback_id for item in admin_list.json())

    student_list = client.get(
        "/api/feedback",
        headers={"Authorization": f"Bearer {student_token}"},
    )
    assert student_list.status_code == 200
    assert len(student_list.json()) == 1

    faculty_list = client.get(
        "/api/feedback",
        headers={"Authorization": f"Bearer {faculty_token}"},
    )
    assert faculty_list.status_code == 200
    assert faculty_list.json() == []

    admin_feedback_notifications = client.get(
        "/api/notifications?notification_type=feedback",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert admin_feedback_notifications.status_code == 200
    assert admin_feedback_notifications.json()
    assert any(item["notification_type"] == "feedback" for item in admin_feedback_notifications.json())


def test_feedback_creation_emits_realtime_notification_for_admin(client):
    admin_payload = {
        "name": "Admin WS Feedback",
        "email": "admin-feedback-ws@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    student_payload = {
        "name": "Student WS Feedback",
        "email": "student-feedback-ws@example.com",
        "password": "password123",
        "role": "student",
        "department": "CSE",
        "section_name": "A",
    }
    register_user(client, admin_payload)
    register_user(client, student_payload)

    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")
    student_token = login_user(client, student_payload["email"], student_payload["password"], "student")

    with client.websocket_connect(f"/api/notifications/ws?token={admin_token}") as websocket:
        connected = websocket.receive_json()
        assert connected["event"] == "connected"

        create_response = client.post(
            "/api/feedback",
            json={
                "subject": "Websocket feedback alert test",
                "category": "other",
                "priority": "medium",
                "message": "Trigger realtime feedback notification for admin.",
            },
            headers={"Authorization": f"Bearer {student_token}"},
        )
        assert create_response.status_code == 201

        event = websocket.receive_json()
        assert event["event"] == "notification.created"
        assert event["notification"]["notification_type"] == "feedback"


def test_feedback_thread_updates_and_notifies_reporter(client):
    admin_payload = {
        "name": "Admin Responder",
        "email": "admin-responder@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    faculty_payload = {
        "name": "Faculty Reporter",
        "email": "faculty-reporter@example.com",
        "password": "password123",
        "role": "faculty",
        "department": "CSE",
    }
    register_user(client, admin_payload)
    register_user(client, faculty_payload)

    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")
    faculty_token = login_user(client, faculty_payload["email"], faculty_payload["password"], "faculty")

    create_response = client.post(
        "/api/feedback",
        json={
            "subject": "Lab room projector issue",
            "category": "timetable",
            "priority": "medium",
            "message": "Please avoid assigning lab B204 for projection-heavy classes.",
        },
        headers={"Authorization": f"Bearer {faculty_token}"},
    )
    assert create_response.status_code == 201
    feedback_id = create_response.json()["id"]

    admin_reply = client.post(
        f"/api/feedback/{feedback_id}/messages",
        json={"message": "Noted. We will mark that room as projector-constrained this cycle."},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert admin_reply.status_code == 201

    admin_status = client.put(
        f"/api/feedback/{feedback_id}",
        json={"status": "resolved"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert admin_status.status_code == 200
    assert admin_status.json()["status"] == "resolved"

    detail = client.get(
        f"/api/feedback/{feedback_id}",
        headers={"Authorization": f"Bearer {faculty_token}"},
    )
    assert detail.status_code == 200
    detail_body = detail.json()
    assert detail_body["status"] == "resolved"
    assert len(detail_body["messages"]) == 2

    faculty_reply = client.post(
        f"/api/feedback/{feedback_id}/messages",
        json={"message": "Thank you. Please update me once reflected in the next timetable release."},
        headers={"Authorization": f"Bearer {faculty_token}"},
    )
    assert faculty_reply.status_code == 201

    refreshed = client.get(
        f"/api/feedback/{feedback_id}",
        headers={"Authorization": f"Bearer {faculty_token}"},
    )
    assert refreshed.status_code == 200
    assert refreshed.json()["status"] == "under_review"

    reporter_notifications = client.get(
        "/api/notifications?notification_type=feedback",
        headers={"Authorization": f"Bearer {faculty_token}"},
    )
    assert reporter_notifications.status_code == 200
    assert reporter_notifications.json()


def test_feedback_permissions_enforced_for_non_admin_users(client):
    admin_payload = {
        "name": "Admin Gatekeeper",
        "email": "admin-gate@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    student_a_payload = {
        "name": "Student A",
        "email": "student-a-feedback@example.com",
        "password": "password123",
        "role": "student",
        "department": "CSE",
        "section_name": "A",
    }
    student_b_payload = {
        "name": "Student B",
        "email": "student-b-feedback@example.com",
        "password": "password123",
        "role": "student",
        "department": "CSE",
        "section_name": "B",
    }
    scheduler_payload = {
        "name": "Scheduler Staff",
        "email": "scheduler-feedback@example.com",
        "password": "password123",
        "role": "scheduler",
        "department": "Office",
    }
    register_user(client, admin_payload)
    register_user(client, student_a_payload)
    register_user(client, student_b_payload)
    register_user(client, scheduler_payload)

    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")
    student_a_token = login_user(client, student_a_payload["email"], student_a_payload["password"], "student")
    student_b_token = login_user(client, student_b_payload["email"], student_b_payload["password"], "student")
    scheduler_token = login_user(client, scheduler_payload["email"], scheduler_payload["password"], "scheduler")

    create_response = client.post(
        "/api/feedback",
        json={
            "subject": "Section specific request",
            "category": "suggestion",
            "priority": "low",
            "message": "Please keep one free hour on Wednesday afternoon for club activity.",
        },
        headers={"Authorization": f"Bearer {student_a_token}"},
    )
    assert create_response.status_code == 201
    feedback_id = create_response.json()["id"]

    unauthorized_view = client.get(
        f"/api/feedback/{feedback_id}",
        headers={"Authorization": f"Bearer {student_b_token}"},
    )
    assert unauthorized_view.status_code == 403

    unauthorized_reply = client.post(
        f"/api/feedback/{feedback_id}/messages",
        json={"message": "I should not be able to reply here."},
        headers={"Authorization": f"Bearer {student_b_token}"},
    )
    assert unauthorized_reply.status_code == 403

    scheduler_update = client.put(
        f"/api/feedback/{feedback_id}",
        json={"status": "resolved"},
        headers={"Authorization": f"Bearer {scheduler_token}"},
    )
    assert scheduler_update.status_code == 403

    admin_update = client.put(
        f"/api/feedback/{feedback_id}",
        json={"status": "under_review"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert admin_update.status_code == 200
