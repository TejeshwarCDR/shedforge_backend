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


def test_system_analytics_endpoint_aggregates_inventory_activity_and_operations(client):
    admin_payload = {
        "name": "Admin Analytics",
        "email": "admin-system-analytics@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    student_payload = {
        "name": "Student Analytics",
        "email": "student-system-analytics@example.com",
        "password": "password123",
        "role": "student",
        "department": "CSE",
        "section_name": "A",
    }
    register_user(client, admin_payload)
    register_user(client, student_payload)

    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")
    student_token = login_user(client, student_payload["email"], student_payload["password"], "student")

    program_response = client.post(
        "/api/programs/",
        json={
            "name": "BTech CSE",
            "code": "BTCS",
            "department": "CSE",
            "degree": "BS",
            "duration_years": 4,
            "sections": 1,
            "total_students": 60,
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert program_response.status_code == 201

    faculty_response = client.post(
        "/api/faculty/",
        json={
            "name": "Dr. Faculty One",
            "designation": "Professor",
            "email": "faculty-analytics@example.com",
            "department": "CSE",
            "workload_hours": 0,
            "max_hours": 20,
            "availability": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
            "availability_windows": [],
            "avoid_back_to_back": False,
            "preferred_min_break_minutes": 0,
            "preference_notes": None,
            "preferred_subject_codes": ["CS101"],
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert faculty_response.status_code == 201
    faculty_id = faculty_response.json()["id"]

    room_response = client.post(
        "/api/rooms/",
        json={
            "name": "A101",
            "building": "Main",
            "capacity": 70,
            "type": "lecture",
            "has_lab_equipment": False,
            "has_projector": True,
            "availability_windows": [],
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert room_response.status_code == 201
    room_id = room_response.json()["id"]

    course_response = client.post(
        "/api/courses/",
        json={
            "code": "CS101",
            "name": "Intro to Computing",
            "type": "theory",
            "credits": 3,
            "duration_hours": 1,
            "sections": 1,
            "hours_per_week": 1,
            "faculty_id": faculty_id,
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert course_response.status_code == 201
    course_id = course_response.json()["id"]

    publish_response = client.put(
        "/api/timetable/official?versionLabel=v-analytics",
        json={
            "facultyData": [
                {
                    "id": faculty_id,
                    "name": "Dr. Faculty One",
                    "department": "CSE",
                    "workloadHours": 0,
                    "maxHours": 20,
                    "availability": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                    "email": "faculty-analytics@example.com",
                    "currentWorkload": 0,
                }
            ],
            "courseData": [
                {
                    "id": course_id,
                    "code": "CS101",
                    "name": "Intro to Computing",
                    "type": "theory",
                    "credits": 3,
                    "facultyId": faculty_id,
                    "duration": 1,
                    "hoursPerWeek": 1,
                }
            ],
            "roomData": [
                {
                    "id": room_id,
                    "name": "A101",
                    "capacity": 70,
                    "type": "lecture",
                    "building": "Main",
                    "hasLabEquipment": False,
                    "hasProjector": True,
                }
            ],
            "timetableData": [
                {
                    "id": "slot-analytics-1",
                    "day": "Monday",
                    "startTime": "08:50",
                    "endTime": "09:40",
                    "courseId": course_id,
                    "roomId": room_id,
                    "facultyId": faculty_id,
                    "section": "A",
                    "studentCount": 60,
                }
            ],
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert publish_response.status_code == 200

    feedback_response = client.post(
        "/api/feedback",
        json={
            "subject": "Analytics feedback seed",
            "category": "technical",
            "priority": "medium",
            "message": "Testing system analytics counters.",
        },
        headers={"Authorization": f"Bearer {student_token}"},
    )
    assert feedback_response.status_code == 201

    issue_response = client.post(
        "/api/issues",
        json={
            "category": "conflict",
            "description": "Testing issue queue counters.",
            "affected_slot_id": "slot-analytics-1",
        },
        headers={"Authorization": f"Bearer {student_token}"},
    )
    assert issue_response.status_code == 201

    analytics_response = client.get(
        "/api/system/analytics?days=30",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert analytics_response.status_code == 200
    analytics = analytics_response.json()

    assert analytics["inventory"]["programs"] == 1
    assert analytics["inventory"]["courses"] == 1
    assert analytics["inventory"]["faculty"] == 1
    assert analytics["inventory"]["roomsTotal"] == 1
    assert analytics["timetable"]["isPublished"] is True
    assert analytics["timetable"]["totalSlots"] == 1
    assert analytics["utilization"]["roomUtilizationPercent"] == 100.0
    assert analytics["activity"]["windowDays"] == 30
    assert analytics["activity"]["totalLogs"] >= 3
    assert analytics["activity"]["actionsLastWindow"] >= 3
    assert analytics["activity"]["topActions"]
    assert analytics["operations"]["unreadNotifications"] >= 1
    assert any(item["label"] == "open" and item["value"] >= 1 for item in analytics["operations"]["issuesByStatus"])
    assert any(
        item["label"] == "open" and item["value"] >= 1 for item in analytics["operations"]["feedbackByStatus"]
    )

    forbidden_response = client.get(
        "/api/system/analytics",
        headers={"Authorization": f"Bearer {student_token}"},
    )
    assert forbidden_response.status_code == 403
