from datetime import date, timedelta


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


def next_weekday(start: date, weekday: int) -> date:
    candidate = start
    while candidate.weekday() != weekday:
        candidate += timedelta(days=1)
    return candidate


def _publish_payload(*, slot_a_start: str, slot_a_end: str) -> dict:
    return {
        "facultyData": [
            {
                "id": "f-1",
                "name": "Faculty One",
                "department": "CSE",
                "workloadHours": 0,
                "maxHours": 20,
                "availability": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                "email": "faculty-notify@example.com",
            },
            {
                "id": "f-2",
                "name": "Faculty Two",
                "department": "CSE",
                "workloadHours": 0,
                "maxHours": 20,
                "availability": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                "email": "faculty-notify-2@example.com",
            },
        ],
        "courseData": [
            {
                "id": "course-a",
                "code": "CS-A",
                "name": "Course A",
                "type": "theory",
                "credits": 3,
                "facultyId": "f-1",
                "duration": 1,
                "hoursPerWeek": 1,
            },
            {
                "id": "course-b",
                "code": "CS-B",
                "name": "Course B",
                "type": "theory",
                "credits": 3,
                "facultyId": "f-2",
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
                "startTime": slot_a_start,
                "endTime": slot_a_end,
                "courseId": "course-a",
                "roomId": "room-a",
                "facultyId": "f-1",
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
                "facultyId": "f-2",
                "section": "B",
                "studentCount": 60,
            },
        ],
    }


def test_notification_filters_and_mark_all_read(client):
    admin_payload = {
        "name": "Admin User",
        "email": "notify-admin@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    student_payload = {
        "name": "Student User",
        "email": "notify-student@example.com",
        "password": "password123",
        "role": "student",
        "department": "CSE",
        "section_name": "A",
    }
    register_user(client, admin_payload)
    register_user(client, student_payload)

    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")
    student_token = login_user(client, student_payload["email"], student_payload["password"], "student")

    issue_response = client.post(
        "/api/issues",
        json={
            "category": "other",
            "description": "Notification filter test issue",
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert issue_response.status_code == 201

    backup_response = client.post(
        "/api/system/backup",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert backup_response.status_code == 200

    all_notifications = client.get(
        "/api/notifications",
        headers={"Authorization": f"Bearer {student_token}"},
    )
    assert all_notifications.status_code == 200
    assert len(all_notifications.json()) >= 2

    issue_notifications = client.get(
        "/api/notifications?notification_type=issue",
        headers={"Authorization": f"Bearer {student_token}"},
    )
    assert issue_notifications.status_code == 200
    assert issue_notifications.json()
    assert all(item["notification_type"] == "issue" for item in issue_notifications.json())

    mark_all_response = client.post(
        "/api/notifications/read-all",
        headers={"Authorization": f"Bearer {student_token}"},
    )
    assert mark_all_response.status_code == 200
    assert mark_all_response.json()["updated"] >= 2

    unread_notifications = client.get(
        "/api/notifications?is_read=false",
        headers={"Authorization": f"Bearer {student_token}"},
    )
    assert unread_notifications.status_code == 200
    assert unread_notifications.json() == []


def test_substitute_assignment_notifies_leave_owner_and_substitute(client, monkeypatch):
    admin_payload = {
        "name": "Admin Leave",
        "email": "admin-substitute@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    leave_faculty_payload = {
        "name": "Faculty Leave",
        "email": "faculty-leave@example.com",
        "password": "password123",
        "role": "faculty",
        "department": "CSE",
    }
    substitute_faculty_payload = {
        "name": "Faculty Substitute",
        "email": "faculty-substitute@example.com",
        "password": "password123",
        "role": "faculty",
        "department": "CSE",
    }
    register_user(client, admin_payload)
    register_user(client, leave_faculty_payload)
    register_user(client, substitute_faculty_payload)

    sent_to: list[str] = []

    def fake_send_email(*, to_email: str, subject: str, text_content: str, html_content=None):
        sent_to.append(to_email)

    monkeypatch.setattr("app.services.notifications.send_email", fake_send_email)

    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")
    leave_faculty_token = login_user(client, leave_faculty_payload["email"], leave_faculty_payload["password"], "faculty")
    substitute_faculty_token = login_user(
        client,
        substitute_faculty_payload["email"],
        substitute_faculty_payload["password"],
        "faculty",
    )

    leave_create = client.post(
        "/api/leaves",
        json={
            "leave_date": (date.today() + timedelta(days=2)).isoformat(),
            "leave_type": "casual",
            "reason": "Conference travel",
        },
        headers={"Authorization": f"Bearer {leave_faculty_token}"},
    )
    assert leave_create.status_code == 201
    leave_id = leave_create.json()["id"]

    leave_approve = client.put(
        f"/api/leaves/{leave_id}/status",
        json={"status": "approved", "admin_comment": "Approved"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert leave_approve.status_code == 200

    faculty_list = client.get("/api/faculty", headers={"Authorization": f"Bearer {admin_token}"})
    assert faculty_list.status_code == 200
    substitute_id = next(
        item["id"]
        for item in faculty_list.json()
        if item["email"] == substitute_faculty_payload["email"]
    )

    assign_response = client.post(
        f"/api/leaves/{leave_id}/substitute",
        json={"substitute_faculty_id": substitute_id, "notes": "Handle first two periods"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert assign_response.status_code == 200
    assert assign_response.json()["substitute_faculty_id"] == substitute_id

    leave_owner_notifications = client.get(
        "/api/notifications?notification_type=workflow",
        headers={"Authorization": f"Bearer {leave_faculty_token}"},
    )
    assert leave_owner_notifications.status_code == 200
    assert any("Substitute Assigned For Your Leave" == item["title"] for item in leave_owner_notifications.json())

    substitute_notifications = client.get(
        "/api/notifications?notification_type=workflow",
        headers={"Authorization": f"Bearer {substitute_faculty_token}"},
    )
    assert substitute_notifications.status_code == 200
    assert any("Substitute Class Assignment" == item["title"] for item in substitute_notifications.json())

    assert leave_faculty_payload["email"] in sent_to
    assert substitute_faculty_payload["email"] in sent_to


def test_leave_approval_auto_reassigns_slots_by_preference_and_notifies_users(client, monkeypatch):
    admin_payload = {
        "name": "Admin Auto Substitute",
        "email": "admin-auto-substitute@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    leave_faculty_payload = {
        "name": "Faculty On Leave",
        "email": "faculty-auto-leave@example.com",
        "password": "password123",
        "role": "faculty",
        "department": "CSE",
        "preferred_subject_codes": [],
    }
    substitute_faculty_payload = {
        "name": "Faculty Cover",
        "email": "faculty-auto-cover@example.com",
        "password": "password123",
        "role": "faculty",
        "department": "CSE",
        "preferred_subject_codes": ["CSAUTO101"],
    }
    student_payload = {
        "name": "Student Auto",
        "email": "student-auto-section-a@example.com",
        "password": "password123",
        "role": "student",
        "department": "CSE",
        "section_name": "A",
    }
    register_user(client, admin_payload)
    register_user(client, leave_faculty_payload)
    register_user(client, substitute_faculty_payload)
    register_user(client, student_payload)

    sent_to: list[str] = []

    def fake_send_email(*, to_email: str, subject: str, text_content: str, html_content=None):
        sent_to.append(to_email)

    monkeypatch.setattr("app.services.notifications.send_email", fake_send_email)

    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")
    leave_token = login_user(client, leave_faculty_payload["email"], leave_faculty_payload["password"], "faculty")
    substitute_token = login_user(
        client,
        substitute_faculty_payload["email"],
        substitute_faculty_payload["password"],
        "faculty",
    )
    student_token = login_user(client, student_payload["email"], student_payload["password"], "student")

    leave_profile = client.get("/api/faculty/me", headers={"Authorization": f"Bearer {leave_token}"})
    substitute_profile = client.get("/api/faculty/me", headers={"Authorization": f"Bearer {substitute_token}"})
    assert leave_profile.status_code == 200
    assert substitute_profile.status_code == 200
    leave_faculty_id = leave_profile.json()["id"]
    substitute_faculty_id = substitute_profile.json()["id"]

    publish_response = client.put(
        "/api/timetable/official?versionLabel=v-auto-substitute-base",
        json={
            "facultyData": [
                {
                    "id": leave_faculty_id,
                    "name": leave_faculty_payload["name"],
                    "department": "CSE",
                    "workloadHours": 0,
                    "maxHours": 20,
                    "availability": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                    "email": leave_faculty_payload["email"],
                },
                {
                    "id": substitute_faculty_id,
                    "name": substitute_faculty_payload["name"],
                    "department": "CSE",
                    "workloadHours": 0,
                    "maxHours": 20,
                    "availability": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                    "email": substitute_faculty_payload["email"],
                },
            ],
            "courseData": [
                {
                    "id": "course-auto",
                    "code": "CSAUTO101",
                    "name": "Auto Sub Course",
                    "type": "theory",
                    "credits": 3,
                    "facultyId": leave_faculty_id,
                    "duration": 1,
                    "hoursPerWeek": 1,
                }
            ],
            "roomData": [
                {
                    "id": "room-auto",
                    "name": "A101",
                    "capacity": 70,
                    "type": "lecture",
                    "building": "Main",
                }
            ],
            "timetableData": [
                {
                    "id": "slot-auto",
                    "day": "Monday",
                    "startTime": "08:50",
                    "endTime": "09:40",
                    "courseId": "course-auto",
                    "roomId": "room-auto",
                    "facultyId": leave_faculty_id,
                    "section": "A",
                    "studentCount": 60,
                }
            ],
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert publish_response.status_code == 200

    leave_date = next_weekday(date.today() + timedelta(days=1), 0).isoformat()  # Monday
    create_leave = client.post(
        "/api/leaves",
        json={
            "leave_date": leave_date,
            "leave_type": "casual",
            "reason": "Auto substitute verification",
        },
        headers={"Authorization": f"Bearer {leave_token}"},
    )
    assert create_leave.status_code == 201
    leave_id = create_leave.json()["id"]

    approve_leave = client.put(
        f"/api/leaves/{leave_id}/status",
        json={"status": "approved", "admin_comment": "Approved"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert approve_leave.status_code == 200

    pending_offers = client.get(
        "/api/leaves/substitute-offers?status=pending",
        headers={"Authorization": f"Bearer {substitute_token}"},
    )
    assert pending_offers.status_code == 200
    assert len(pending_offers.json()) == 1

    accept_offer = client.post(
        f"/api/leaves/substitute-offers/{pending_offers.json()[0]['id']}/respond",
        json={"decision": "accept"},
        headers={"Authorization": f"Bearer {substitute_token}"},
    )
    assert accept_offer.status_code == 200
    assert accept_offer.json()["status"] == "accepted"

    official_after = client.get(
        "/api/timetable/official",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert official_after.status_code == 200
    slot = official_after.json()["timetableData"][0]
    assert slot["facultyId"] == substitute_faculty_id

    substitute_notifications = client.get(
        "/api/notifications?notification_type=workflow",
        headers={"Authorization": f"Bearer {substitute_token}"},
    )
    assert substitute_notifications.status_code == 200
    assert any(
        item["title"] in {"Substitute Request Pending", "Substitute Acceptance Recorded"}
        for item in substitute_notifications.json()
    )

    student_notifications = client.get(
        "/api/notifications?notification_type=timetable",
        headers={"Authorization": f"Bearer {student_token}"},
    )
    assert student_notifications.status_code == 200
    assert any(item["title"] == "Class Schedule Updated" for item in student_notifications.json())

    assert substitute_faculty_payload["email"] in sent_to
    assert student_payload["email"] in sent_to


def test_leave_auto_substitute_skips_preferred_faculty_when_not_free_in_slot_window(client):
    admin_payload = {
        "name": "Admin Busy Substitute",
        "email": "admin-busy-substitute@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    leave_faculty_payload = {
        "name": "Faculty Busy Leave",
        "email": "faculty-busy-leave@example.com",
        "password": "password123",
        "role": "faculty",
        "department": "CSE",
    }
    substitute_faculty_payload = {
        "name": "Faculty Busy Cover",
        "email": "faculty-busy-cover@example.com",
        "password": "password123",
        "role": "faculty",
        "department": "CSE",
        "preferred_subject_codes": ["CSBUSY101"],
    }
    register_user(client, admin_payload)
    register_user(client, leave_faculty_payload)
    register_user(client, substitute_faculty_payload)

    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")
    leave_token = login_user(client, leave_faculty_payload["email"], leave_faculty_payload["password"], "faculty")
    substitute_token = login_user(
        client,
        substitute_faculty_payload["email"],
        substitute_faculty_payload["password"],
        "faculty",
    )

    leave_profile = client.get("/api/faculty/me", headers={"Authorization": f"Bearer {leave_token}"})
    substitute_profile = client.get("/api/faculty/me", headers={"Authorization": f"Bearer {substitute_token}"})
    assert leave_profile.status_code == 200
    assert substitute_profile.status_code == 200
    leave_faculty_id = leave_profile.json()["id"]
    substitute_faculty_id = substitute_profile.json()["id"]

    update_substitute = client.put(
        f"/api/faculty/{substitute_faculty_id}",
        json={
            "availability_windows": [
                {
                    "day": "Monday",
                    "start_time": "10:30",
                    "end_time": "11:20",
                }
            ]
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert update_substitute.status_code == 200

    publish_response = client.put(
        "/api/timetable/official?versionLabel=v-auto-substitute-busy",
        json={
            "facultyData": [
                {
                    "id": leave_faculty_id,
                    "name": leave_faculty_payload["name"],
                    "department": "CSE",
                    "workloadHours": 0,
                    "maxHours": 20,
                    "availability": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                    "email": leave_faculty_payload["email"],
                },
                {
                    "id": substitute_faculty_id,
                    "name": substitute_faculty_payload["name"],
                    "department": "CSE",
                    "workloadHours": 0,
                    "maxHours": 20,
                    "availability": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                    "email": substitute_faculty_payload["email"],
                },
            ],
            "courseData": [
                {
                    "id": "course-busy-leave",
                    "code": "CSBUSY101",
                    "name": "Busy Leave Course",
                    "type": "theory",
                    "credits": 3,
                    "facultyId": leave_faculty_id,
                    "duration": 1,
                    "hoursPerWeek": 1,
                },
            ],
            "roomData": [
                {
                    "id": "room-busy-1",
                    "name": "A101",
                    "capacity": 70,
                    "type": "lecture",
                    "building": "Main",
                },
            ],
            "timetableData": [
                {
                    "id": "slot-busy-leave",
                    "day": "Monday",
                    "startTime": "08:50",
                    "endTime": "09:40",
                    "courseId": "course-busy-leave",
                    "roomId": "room-busy-1",
                    "facultyId": leave_faculty_id,
                    "section": "A",
                    "studentCount": 60,
                },
            ],
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert publish_response.status_code == 200

    leave_date = next_weekday(date.today() + timedelta(days=1), 0).isoformat()  # Monday
    create_leave = client.post(
        "/api/leaves",
        json={
            "leave_date": leave_date,
            "leave_type": "casual",
            "reason": "Overlap check",
        },
        headers={"Authorization": f"Bearer {leave_token}"},
    )
    assert create_leave.status_code == 201
    leave_id = create_leave.json()["id"]

    approve_leave = client.put(
        f"/api/leaves/{leave_id}/status",
        json={"status": "approved"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert approve_leave.status_code == 200

    official_after = client.get(
        "/api/timetable/official",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert official_after.status_code == 200
    leave_slot = next(item for item in official_after.json()["timetableData"] if item["id"] == "slot-busy-leave")
    assert leave_slot["facultyId"] == leave_faculty_id


def test_timetable_update_targets_impacted_students(client):
    admin_payload = {
        "name": "Admin Publish",
        "email": "admin-notify-publish@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    student_a_payload = {
        "name": "Student A",
        "email": "student-a-notify@example.com",
        "password": "password123",
        "role": "student",
        "department": "CSE",
        "section_name": "A",
    }
    student_b_payload = {
        "name": "Student B",
        "email": "student-b-notify@example.com",
        "password": "password123",
        "role": "student",
        "department": "CSE",
        "section_name": "B",
    }
    register_user(client, admin_payload)
    register_user(client, student_a_payload)
    register_user(client, student_b_payload)

    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")
    student_a_token = login_user(client, student_a_payload["email"], student_a_payload["password"], "student")
    student_b_token = login_user(client, student_b_payload["email"], student_b_payload["password"], "student")

    first_publish = client.put(
        "/api/timetable/official?versionLabel=v-notify-1",
        json=_publish_payload(slot_a_start="08:50", slot_a_end="09:40"),
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert first_publish.status_code == 200

    before_a = client.get("/api/notifications", headers={"Authorization": f"Bearer {student_a_token}"})
    before_b = client.get("/api/notifications", headers={"Authorization": f"Bearer {student_b_token}"})
    assert before_a.status_code == 200
    assert before_b.status_code == 200
    baseline_a = len(before_a.json())
    baseline_b = len(before_b.json())

    second_publish = client.put(
        "/api/timetable/official?versionLabel=v-notify-2",
        json=_publish_payload(slot_a_start="14:05", slot_a_end="14:55"),
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert second_publish.status_code == 200

    after_a = client.get("/api/notifications", headers={"Authorization": f"Bearer {student_a_token}"})
    after_b = client.get("/api/notifications", headers={"Authorization": f"Bearer {student_b_token}"})
    assert after_a.status_code == 200
    assert after_b.status_code == 200

    new_a = after_a.json()[: len(after_a.json()) - baseline_a]
    new_b = after_b.json()[: len(after_b.json()) - baseline_b]

    assert any(item["title"] == "Class Schedule Updated" for item in new_a)
    assert not any(item["title"] == "Class Schedule Updated" for item in new_b)


def test_notifications_websocket_stream_receives_realtime_events(client):
    admin_payload = {
        "name": "Admin WS",
        "email": "admin-ws@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    student_payload = {
        "name": "Student WS",
        "email": "student-ws@example.com",
        "password": "password123",
        "role": "student",
        "department": "CSE",
        "section_name": "A",
    }
    register_user(client, admin_payload)
    register_user(client, student_payload)

    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")
    student_token = login_user(client, student_payload["email"], student_payload["password"], "student")

    with client.websocket_connect(f"/api/notifications/ws?token={student_token}") as websocket:
        connected = websocket.receive_json()
        assert connected["event"] == "connected"

        issue_response = client.post(
            "/api/issues",
            json={
                "category": "other",
                "description": "Realtime websocket event check",
            },
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert issue_response.status_code == 201

        created_event = websocket.receive_json()
        assert created_event["event"] == "notification.created"
        assert created_event["notification"]["notification_type"] == "issue"
