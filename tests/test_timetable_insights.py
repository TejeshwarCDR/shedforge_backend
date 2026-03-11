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


def test_timetable_conflicts_and_analytics_endpoints(client):
    admin_payload = {
        "name": "Admin User",
        "email": "admin-insights@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    student_payload = {
        "name": "Student User",
        "email": "student-insights@example.com",
        "password": "password123",
        "role": "student",
        "department": "CSE",
        "section_name": "A",
    }
    register_user(client, admin_payload)
    register_user(client, student_payload)

    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")
    student_token = login_user(client, student_payload["email"], student_payload["password"], "student")

    payload = {
        "facultyData": [
            {
                "id": "f1",
                "name": "Prof Availability",
                "department": "CSE",
                "workloadHours": 0,
                "maxHours": 20,
                "availability": ["Tuesday"],
                "email": "faculty@example.com",
                "currentWorkload": 0,
            }
        ],
        "courseData": [
            {
                "id": "c1",
                "code": "CSE211",
                "name": "Algorithms",
                "type": "theory",
                "credits": 4,
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
                "id": "ts1",
                "day": "Monday",
                "startTime": "08:50",
                "endTime": "09:40",
                "courseId": "c1",
                "roomId": "r1",
                "facultyId": "f1",
                "section": "A",
                "studentCount": 60,
            }
        ],
    }

    put_response = client.put(
        "/api/timetable/official",
        json=payload,
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert put_response.status_code == 200

    forbidden_conflicts_response = client.get(
        "/api/timetable/conflicts",
        headers={"Authorization": f"Bearer {student_token}"},
    )
    assert forbidden_conflicts_response.status_code == 403

    conflicts_response = client.get(
        "/api/timetable/conflicts",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert conflicts_response.status_code == 200
    conflicts = conflicts_response.json()
    assert any(item["type"] == "availability" for item in conflicts)

    forbidden_analytics_response = client.get(
        "/api/timetable/analytics",
        headers={"Authorization": f"Bearer {student_token}"},
    )
    assert forbidden_analytics_response.status_code == 403

    analytics_response = client.get(
        "/api/timetable/analytics",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert analytics_response.status_code == 200
    analytics = analytics_response.json()

    assert analytics["optimizationSummary"]["conflictsDetected"] >= 1
    assert analytics["constraintData"]
    assert analytics["workloadChartData"]
    assert analytics["dailyWorkloadData"]


def test_timetable_conflict_analysis_endpoint_supports_draft_payload(client):
    admin_payload = {
        "name": "Admin Draft Analyze",
        "email": "admin-draft-analyze@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    register_user(client, admin_payload)
    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")

    draft_payload = {
        "facultyData": [
            {
                "id": "f-a",
                "name": "Prof A",
                "department": "CSE",
                "workloadHours": 0,
                "maxHours": 20,
                "availability": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                "email": "prof-a@example.com",
                "currentWorkload": 0,
            },
            {
                "id": "f-b",
                "name": "Prof B",
                "department": "CSE",
                "workloadHours": 0,
                "maxHours": 20,
                "availability": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                "email": "prof-b@example.com",
                "currentWorkload": 0,
            },
        ],
        "courseData": [
            {
                "id": "c-a",
                "code": "CSE401",
                "name": "AI",
                "type": "theory",
                "credits": 3,
                "facultyId": "f-a",
                "duration": 1,
                "hoursPerWeek": 1,
            },
            {
                "id": "c-b",
                "code": "CSE402",
                "name": "ML",
                "type": "theory",
                "credits": 3,
                "facultyId": "f-b",
                "duration": 1,
                "hoursPerWeek": 1,
            },
        ],
        "roomData": [
            {
                "id": "r-101",
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
                "id": "s-a",
                "day": "Monday",
                "startTime": "08:50",
                "endTime": "09:40",
                "courseId": "c-a",
                "roomId": "r-101",
                "facultyId": "f-a",
                "section": "A",
                "studentCount": 60,
            },
            {
                "id": "s-b",
                "day": "Monday",
                "startTime": "08:50",
                "endTime": "09:40",
                "courseId": "c-b",
                "roomId": "r-101",
                "facultyId": "f-b",
                "section": "B",
                "studentCount": 60,
            },
        ],
    }

    response = client.post(
        "/api/timetable/conflicts/analyze",
        json=draft_payload,
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert response.status_code == 200
    conflicts = response.json()
    assert any(item["type"] == "room-overlap" for item in conflicts)


def test_conflict_decision_yes_applies_fix_and_tracks_resolution(client):
    admin_payload = {
        "name": "Admin Resolve",
        "email": "admin-resolve@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    register_user(client, admin_payload)
    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")

    payload = {
        "facultyData": [
            {
                "id": "f1",
                "name": "Prof One",
                "department": "CSE",
                "workloadHours": 0,
                "maxHours": 20,
                "availability": ["Tuesday", "Wednesday", "Thursday", "Friday"],
                "email": "prof1@example.com",
                "currentWorkload": 0,
            },
            {
                "id": "f2",
                "name": "Prof Two",
                "department": "CSE",
                "workloadHours": 0,
                "maxHours": 20,
                "availability": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                "email": "prof2@example.com",
                "currentWorkload": 0,
            },
        ],
        "courseData": [
            {
                "id": "c1",
                "code": "CSE301",
                "name": "Course One",
                "type": "theory",
                "credits": 3,
                "facultyId": "f1",
                "duration": 1,
                "hoursPerWeek": 1,
            },
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
            },
        ],
        "timetableData": [
            {
                "id": "ts1",
                "day": "Monday",
                "startTime": "08:50",
                "endTime": "09:40",
                "courseId": "c1",
                "roomId": "r1",
                "facultyId": "f1",
                "section": "A",
                "studentCount": 60,
            },
        ],
    }

    put_response = client.put(
        "/api/timetable/official",
        json=payload,
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert put_response.status_code == 200

    conflicts_response = client.get(
        "/api/timetable/conflicts",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert conflicts_response.status_code == 200
    conflicts = conflicts_response.json()
    availability_conflict = next(item for item in conflicts if item["id"].startswith("availability-faculty-day-"))

    decision_response = client.post(
        f"/api/timetable/conflicts/{availability_conflict['id']}/decision",
        json={"decision": "yes"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert decision_response.status_code == 200
    decision = decision_response.json()
    assert decision["resolved"] is True
    assert decision["published_version_label"] is not None

    official_after = client.get(
        "/api/timetable/official",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert official_after.status_code == 200
    slots = official_after.json()["timetableData"]
    slot = slots[0]
    assert slot["facultyId"] == "f2"

    updated_conflicts = client.get(
        "/api/timetable/conflicts",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert updated_conflicts.status_code == 200
    updated = updated_conflicts.json()
    tracked = next(item for item in updated if item["id"] == availability_conflict["id"])
    assert tracked["resolved"] is True


def test_conflict_decision_no_keeps_conflict_open(client):
    admin_payload = {
        "name": "Admin Skip",
        "email": "admin-skip@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    register_user(client, admin_payload)
    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")

    payload = {
        "facultyData": [
            {
                "id": "f1",
                "name": "Prof Availability",
                "department": "CSE",
                "workloadHours": 0,
                "maxHours": 20,
                "availability": ["Tuesday"],
                "email": "prof-availability@example.com",
                "currentWorkload": 0,
            }
        ],
        "courseData": [
            {
                "id": "c1",
                "code": "CSE211",
                "name": "Algorithms",
                "type": "theory",
                "credits": 4,
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
                "id": "ts1",
                "day": "Monday",
                "startTime": "08:50",
                "endTime": "09:40",
                "courseId": "c1",
                "roomId": "r1",
                "facultyId": "f1",
                "section": "A",
                "studentCount": 60,
            }
        ],
    }

    put_response = client.put(
        "/api/timetable/official",
        json=payload,
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert put_response.status_code == 200

    conflicts_response = client.get(
        "/api/timetable/conflicts",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert conflicts_response.status_code == 200
    availability_conflict = next(item for item in conflicts_response.json() if item["type"] == "availability")

    decision_response = client.post(
        f"/api/timetable/conflicts/{availability_conflict['id']}/decision",
        json={"decision": "no", "note": "Will handle manually later"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert decision_response.status_code == 200
    decision = decision_response.json()
    assert decision["resolved"] is False

    after_response = client.get(
        "/api/timetable/conflicts",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert after_response.status_code == 200
    tracked = next(item for item in after_response.json() if item["id"] == availability_conflict["id"])
    assert tracked["resolved"] is False


def test_conflict_analysis_flags_course_faculty_inconsistency(client):
    admin_payload = {
        "name": "Admin Faculty Conflict",
        "email": "admin-faculty-conflict@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    register_user(client, admin_payload)
    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")

    draft_payload = {
        "facultyData": [
            {
                "id": "f-a",
                "name": "Prof A",
                "department": "CSE",
                "workloadHours": 0,
                "maxHours": 20,
                "availability": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                "email": "prof-a@example.com",
                "currentWorkload": 0,
            },
            {
                "id": "f-b",
                "name": "Prof B",
                "department": "CSE",
                "workloadHours": 0,
                "maxHours": 20,
                "availability": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                "email": "prof-b@example.com",
                "currentWorkload": 0,
            },
        ],
        "courseData": [
            {
                "id": "c-1",
                "code": "CSE311",
                "name": "Software Engineering",
                "type": "theory",
                "credits": 4,
                "facultyId": "f-a",
                "duration": 1,
                "hoursPerWeek": 1,
            }
        ],
        "roomData": [
            {
                "id": "r-1",
                "name": "A101",
                "capacity": 70,
                "type": "lecture",
                "building": "Main",
                "hasLabEquipment": False,
                "hasProjector": True,
                "utilization": 0,
            },
            {
                "id": "r-2",
                "name": "A102",
                "capacity": 70,
                "type": "lecture",
                "building": "Main",
                "hasLabEquipment": False,
                "hasProjector": True,
                "utilization": 0,
            },
        ],
        "timetableData": [
            {
                "id": "s-a",
                "day": "Monday",
                "startTime": "08:50",
                "endTime": "09:40",
                "courseId": "c-1",
                "roomId": "r-1",
                "facultyId": "f-a",
                "section": "A",
                "studentCount": 60,
            },
            {
                "id": "s-b",
                "day": "Tuesday",
                "startTime": "08:50",
                "endTime": "09:40",
                "courseId": "c-1",
                "roomId": "r-2",
                "facultyId": "f-b",
                "section": "A",
                "studentCount": 60,
            },
        ],
    }

    response = client.post(
        "/api/timetable/conflicts/analyze",
        json=draft_payload,
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert response.status_code == 200
    conflicts = response.json()
    course_conflict = next((item for item in conflicts if item["type"] == "course-faculty-inconsistency"), None)
    assert course_conflict is not None
    assert set(course_conflict["affectedSlots"]) == {"s-a", "s-b"}


def test_conflict_analysis_flags_room_type_working_hours_and_workload_overflow(client):
    admin_payload = {
        "name": "Admin Constraint Inspector",
        "email": "admin-constraint-inspector@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    register_user(client, admin_payload)
    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")

    draft_payload = {
        "facultyData": [
            {
                "id": "f-lab",
                "name": "Prof Lab",
                "department": "CSE",
                "workloadHours": 0,
                "maxHours": 1,
                "availability": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                "email": "prof-lab@example.com",
                "currentWorkload": 0,
            }
        ],
        "courseData": [
            {
                "id": "c-lab",
                "code": "CSE451L",
                "name": "Systems Lab",
                "type": "lab",
                "credits": 1,
                "facultyId": "f-lab",
                "duration": 1,
                "hoursPerWeek": 2,
                "theoryHours": 0,
                "labHours": 2,
                "tutorialHours": 0,
                "practicalContiguousSlots": 1,
            }
        ],
        "roomData": [
            {
                "id": "r-lecture",
                "name": "A101",
                "capacity": 80,
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
                "startTime": "07:00",
                "endTime": "07:50",
                "courseId": "c-lab",
                "roomId": "r-lecture",
                "facultyId": "f-lab",
                "section": "A",
                "batch": "B1",
                "studentCount": 30,
                "sessionType": "lab",
            },
            {
                "id": "slot-2",
                "day": "Tuesday",
                "startTime": "08:50",
                "endTime": "09:40",
                "courseId": "c-lab",
                "roomId": "r-lecture",
                "facultyId": "f-lab",
                "section": "A",
                "batch": "B1",
                "studentCount": 30,
                "sessionType": "lab",
            },
        ],
    }

    response = client.post(
        "/api/timetable/conflicts/analyze",
        json=draft_payload,
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert response.status_code == 200

    conflicts = response.json()
    assert any(item["type"] == "room-type" for item in conflicts)
    assert any(item["type"] == "working-hours" for item in conflicts)
    assert any(item["type"] == "workload-overflow" for item in conflicts)

    room_type_conflict = next(item for item in conflicts if item["type"] == "room-type")
    assert "requires a lab room" in room_type_conflict["description"]

    working_hours_conflict = next(item for item in conflicts if item["type"] == "working-hours")
    assert "working hours" in working_hours_conflict["description"] or "teaching" in working_hours_conflict["description"]

    overload_conflict = next(item for item in conflicts if item["type"] == "workload-overflow")
    assert "exceeds the configured weekly maximum" in overload_conflict["description"]


def test_conflict_resolve_all_fixes_room_type_conflict(client):
    admin_payload = {
        "name": "Admin Room Type Resolve",
        "email": "admin-room-type-resolve@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    register_user(client, admin_payload)
    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")

    draft_payload = {
        "facultyData": [
            {
                "id": "f-lab",
                "name": "Prof Lab",
                "department": "CSE",
                "workloadHours": 0,
                "maxHours": 4,
                "availability": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                "email": "prof-lab-resolve@example.com",
                "currentWorkload": 0,
            }
        ],
        "courseData": [
            {
                "id": "c-lab",
                "code": "CSE452L",
                "name": "Embedded Lab",
                "type": "lab",
                "credits": 1,
                "facultyId": "f-lab",
                "duration": 1,
                "hoursPerWeek": 1,
                "theoryHours": 0,
                "labHours": 1,
                "tutorialHours": 0,
                "practicalContiguousSlots": 1,
            }
        ],
        "roomData": [
            {
                "id": "r-lecture",
                "name": "A101",
                "capacity": 80,
                "type": "lecture",
                "building": "Main",
                "hasLabEquipment": False,
                "hasProjector": True,
                "utilization": 0,
            },
            {
                "id": "r-lab",
                "name": "Lab-201",
                "capacity": 40,
                "type": "lab",
                "building": "Main",
                "hasLabEquipment": True,
                "hasProjector": True,
                "utilization": 0,
            },
        ],
        "timetableData": [
            {
                "id": "slot-room-type",
                "day": "Monday",
                "startTime": "08:50",
                "endTime": "09:40",
                "courseId": "c-lab",
                "roomId": "r-lecture",
                "facultyId": "f-lab",
                "section": "A",
                "batch": "B1",
                "studentCount": 30,
                "sessionType": "lab",
            }
        ],
    }

    response = client.post(
        "/api/timetable/conflicts/resolve-all",
        json={"payload": draft_payload, "scope": "hard", "promoteOfficial": False},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert response.status_code == 200

    body = response.json()
    assert body["resolvedCount"] >= 1
    assert body["remainingConflicts"] == []
    assert body["resolvedPayload"]["timetableData"][0]["roomId"] == "r-lab"
    assert any(
        "alternate compatible room" in item["resolution"].lower() or "compatible lab room" in item["resolution"].lower()
        for item in body["autoResolvedConflicts"]
    )


def test_conflict_review_supports_assistant_faculty_ids_in_payload(client):
    admin_payload = {
        "name": "Admin Conflict Review",
        "email": "admin-conflict-review@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    register_user(client, admin_payload)
    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")

    draft_payload = {
        "facultyData": [
            {
                "id": "f-primary",
                "name": "Prof Primary",
                "department": "CSE",
                "workloadHours": 0,
                "maxHours": 10,
                "availability": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                "email": "prof-primary@example.com",
                "currentWorkload": 0,
            },
            {
                "id": "f-other",
                "name": "Prof Other",
                "department": "CSE",
                "workloadHours": 0,
                "maxHours": 10,
                "availability": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                "email": "prof-other@example.com",
                "currentWorkload": 0,
            },
        ],
        "courseData": [
            {
                "id": "c-primary",
                "code": "CSE501",
                "name": "Distributed Systems",
                "type": "theory",
                "credits": 1,
                "facultyId": "f-primary",
                "duration": 1,
                "hoursPerWeek": 1,
            },
            {
                "id": "c-other",
                "code": "CSE502",
                "name": "Advanced Networks",
                "type": "theory",
                "credits": 1,
                "facultyId": "f-other",
                "duration": 1,
                "hoursPerWeek": 1,
            },
        ],
        "roomData": [
            {
                "id": "r-101",
                "name": "A101",
                "capacity": 70,
                "type": "lecture",
                "building": "Main",
                "hasLabEquipment": False,
                "hasProjector": True,
                "utilization": 0,
            },
            {
                "id": "r-102",
                "name": "A102",
                "capacity": 70,
                "type": "lecture",
                "building": "Main",
                "hasLabEquipment": False,
                "hasProjector": True,
                "utilization": 0,
            },
        ],
        "timetableData": [
            {
                "id": "slot-primary",
                "day": "Monday",
                "startTime": "08:50",
                "endTime": "09:40",
                "courseId": "c-primary",
                "roomId": "r-101",
                "facultyId": "f-primary",
                "section": "A",
                "studentCount": 40,
            },
            {
                "id": "slot-assistant",
                "day": "Monday",
                "startTime": "08:50",
                "endTime": "09:40",
                "courseId": "c-other",
                "roomId": "r-102",
                "facultyId": "f-other",
                "assistantFacultyIds": ["f-primary"],
                "section": "B",
                "studentCount": 35,
            },
        ],
    }

    response = client.post(
        "/api/timetable/conflicts/review",
        json={"payload": draft_payload},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["pendingConflicts"]
    assert any(item["type"] == "faculty-overlap" for item in body["pendingConflicts"])
