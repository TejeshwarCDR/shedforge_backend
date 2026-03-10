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


def next_weekday(start: date) -> date:
    candidate = start
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate


def _payload_with_time(start_time: str, end_time: str) -> dict:
    return {
        "facultyData": [
            {
                "id": "f1",
                "name": "Prof One",
                "department": "CSE",
                "workloadHours": 0,
                "maxHours": 20,
                "availability": ["Monday", "Tuesday"],
                "email": "prof1@example.com",
                "currentWorkload": 0,
            }
        ],
        "courseData": [
            {
                "id": "c1",
                "code": "CS100",
                "name": "Intro CSE",
                "type": "theory",
                "credits": 3,
                "facultyId": "f1",
                "duration": 1,
                "hoursPerWeek": 1,
            }
        ],
        "roomData": [
            {
                "id": "r1",
                "name": "A101",
                "capacity": 70,
                "type": "lecture",
                "building": "Main",
                "hasLabEquipment": False,
                "hasProjector": True,
                "utilization": 0,
            }
        ],
        "timetableData": [
            {
                "id": "slot-1",
                "day": "Monday",
                "startTime": start_time,
                "endTime": end_time,
                "courseId": "c1",
                "roomId": "r1",
                "facultyId": "f1",
                "section": "A",
                "studentCount": 60,
            }
        ],
    }


def test_timetable_versions_notifications_and_activity_logs(client):
    admin_payload = {
        "name": "Admin Governance",
        "email": "admin-governance@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    student_payload = {
        "name": "Student Governance",
        "email": "student-governance@example.com",
        "password": "password123",
        "role": "student",
        "department": "CSE",
        "section_name": "A",
    }
    register_user(client, admin_payload)
    register_user(client, student_payload)

    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")
    student_token = login_user(client, student_payload["email"], student_payload["password"], "student")

    publish_one = client.put(
        "/api/timetable/official?versionLabel=v-manual-1",
        json=_payload_with_time("08:50", "09:40"),
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert publish_one.status_code == 200

    publish_two = client.put(
        "/api/timetable/official?versionLabel=v-manual-2",
        json=_payload_with_time("09:40", "10:30"),
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert publish_two.status_code == 200

    versions_response = client.get(
        "/api/timetable/versions",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert versions_response.status_code == 200
    versions = versions_response.json()
    assert len(versions) >= 2

    compare_response = client.get(
        f"/api/timetable/versions/compare?from={versions[1]['id']}&to={versions[0]['id']}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert compare_response.status_code == 200
    compare = compare_response.json()
    assert compare["from_version_id"] == versions[1]["id"]
    assert compare["to_version_id"] == versions[0]["id"]

    trends_response = client.get(
        "/api/timetable/trends",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert trends_response.status_code == 200
    assert len(trends_response.json()) >= 2

    notifications_response = client.get(
        "/api/notifications",
        headers={"Authorization": f"Bearer {student_token}"},
    )
    assert notifications_response.status_code == 200
    notifications = notifications_response.json()
    assert notifications

    mark_read_response = client.post(
        f"/api/notifications/{notifications[0]['id']}/read",
        headers={"Authorization": f"Bearer {student_token}"},
    )
    assert mark_read_response.status_code == 200
    assert mark_read_response.json()["is_read"] is True

    logs_response = client.get(
        "/api/activity/logs",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert logs_response.status_code == 200
    assert logs_response.json()

    forbidden_logs_response = client.get(
        "/api/activity/logs",
        headers={"Authorization": f"Bearer {student_token}"},
    )
    assert forbidden_logs_response.status_code == 403


def test_substitute_suggestions_and_issue_workflow(client):
    admin_payload = {
        "name": "Admin Ops",
        "email": "admin-ops@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    student_payload = {
        "name": "Student Ops",
        "email": "student-ops@example.com",
        "password": "password123",
        "role": "student",
        "department": "CSE",
        "section_name": "A",
    }
    register_user(client, admin_payload)
    register_user(client, student_payload)

    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")
    student_token = login_user(client, student_payload["email"], student_payload["password"], "student")

    faculty_one = client.post(
        "/api/faculty",
        json={
            "name": "Faculty One",
            "designation": "Assistant Professor",
            "email": "faculty-one@example.com",
            "department": "CSE",
            "workload_hours": 10,
            "max_hours": 20,
            "availability": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
            "availability_windows": [],
            "avoid_back_to_back": False,
            "preferred_min_break_minutes": 0,
            "preference_notes": None,
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert faculty_one.status_code == 201
    faculty_one_id = faculty_one.json()["id"]

    faculty_two = client.post(
        "/api/faculty",
        json={
            "name": "Faculty Two",
            "designation": "Assistant Professor",
            "email": "faculty-two@example.com",
            "department": "CSE",
            "workload_hours": 4,
            "max_hours": 20,
            "availability": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
            "availability_windows": [],
            "avoid_back_to_back": False,
            "preferred_min_break_minutes": 0,
            "preference_notes": None,
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert faculty_two.status_code == 201
    faculty_two_id = faculty_two.json()["id"]

    course_response = client.post(
        "/api/courses",
        json={
            "code": "CS500",
            "name": "Advanced Topics",
            "type": "theory",
            "credits": 3,
            "duration_hours": 1,
            "sections": 1,
            "hours_per_week": 1,
            "faculty_id": faculty_one_id,
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert course_response.status_code == 201
    course_id = course_response.json()["id"]

    suggestion_response = client.get(
        f"/api/faculty/substitutes/suggestions?leave_date={next_weekday(date.today() + timedelta(days=1)).isoformat()}&course_id={course_id}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert suggestion_response.status_code == 200
    suggestions = suggestion_response.json()
    assert suggestions
    assert any(item["faculty_id"] == faculty_two_id for item in suggestions)

    issue_create = client.post(
        "/api/issues",
        json={
            "category": "conflict",
            "description": "Overlapping slot in section A",
            "affected_slot_id": "slot-1",
        },
        headers={"Authorization": f"Bearer {student_token}"},
    )
    assert issue_create.status_code == 201
    issue_id = issue_create.json()["id"]

    issue_update = client.put(
        f"/api/issues/{issue_id}",
        json={"status": "resolved", "resolution_notes": "Updated schedule"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert issue_update.status_code == 200
    assert issue_update.json()["status"] == "resolved"
