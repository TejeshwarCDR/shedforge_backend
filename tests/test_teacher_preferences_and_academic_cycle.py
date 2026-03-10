def register_user(client, payload):
    response = client.post("/api/auth/register", json=payload)
    return response


def login_user(client, email, password, role):
    response = client.post(
        "/api/auth/login",
        json={"email": email, "password": password, "role": role},
    )
    assert response.status_code == 200
    return response.json()["access_token"]


def test_student_section_required_and_faculty_workload_caps(client):
    admin_payload = {
        "name": "Admin User",
        "email": "admin-teacher-pref@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    assert register_user(client, admin_payload).status_code == 201
    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")

    missing_section_student = {
        "name": "Student User",
        "email": "student-missing-section@example.com",
        "password": "password123",
        "role": "student",
        "department": "CSE",
    }
    missing_response = register_user(client, missing_section_student)
    assert missing_response.status_code == 422
    assert "section_name is required" in str(missing_response.json())

    professor_payload = {
        "name": "Dr. Assoc Faculty",
        "designation": "Associate Professor",
        "email": "assoc-faculty@example.com",
        "department": "CSE",
        "workload_hours": 0,
        "max_hours": 20,
        "availability": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        "availability_windows": [],
        "semester_preferences": {"3": ["23cse211", "23cse212"]},
    }
    faculty_response = client.post(
        "/api/faculty",
        json=professor_payload,
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert faculty_response.status_code == 201
    assert faculty_response.json()["max_hours"] == 14
    assert faculty_response.json()["semester_preferences"] == {"3": ["23CSE211", "23CSE212"]}

    assistant_payload = {
        "name": "Dr. Assistant Faculty",
        "designation": "Assistant Professor",
        "email": "assistant-faculty@example.com",
        "department": "CSE",
        "workload_hours": 0,
        "max_hours": 25,
        "availability": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        "availability_windows": [],
    }
    assistant_response = client.post(
        "/api/faculty",
        json=assistant_payload,
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert assistant_response.status_code == 201
    assert assistant_response.json()["max_hours"] == 16


def test_academic_cycle_settings_and_faculty_mapping_endpoint(client):
    admin_payload = {
        "name": "Admin Mapping",
        "email": "admin-mapping@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    faculty_payload = {
        "name": "Faculty Mapping",
        "email": "faculty-mapping@example.com",
        "password": "password123",
        "role": "faculty",
        "department": "CSE",
        "preferred_subject_codes": ["23cse311"],
    }
    student_payload = {
        "name": "Student Mapping",
        "email": "student-mapping@example.com",
        "password": "password123",
        "role": "student",
        "department": "CSE",
        "section_name": "A",
    }

    assert register_user(client, admin_payload).status_code == 201
    assert register_user(client, faculty_payload).status_code == 201
    assert register_user(client, student_payload).status_code == 201

    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")
    faculty_token = login_user(client, faculty_payload["email"], faculty_payload["password"], "faculty")
    student_token = login_user(client, student_payload["email"], student_payload["password"], "student")

    default_cycle = client.get(
        "/api/settings/academic-cycle",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert default_cycle.status_code == 200
    assert default_cycle.json()["semester_cycle"] in {"odd", "even"}

    updated_cycle = client.put(
        "/api/settings/academic-cycle",
        json={"academic_year": "2026-2027", "semester_cycle": "even"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert updated_cycle.status_code == 200
    assert updated_cycle.json()["academic_year"] == "2026-2027"
    assert updated_cycle.json()["semester_cycle"] == "even"

    faculty_profile = client.get(
        "/api/faculty/me",
        headers={"Authorization": f"Bearer {faculty_token}"},
    )
    assert faculty_profile.status_code == 200
    faculty_id = faculty_profile.json()["id"]

    room_response = client.post(
        "/api/rooms",
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
        "/api/courses",
        json={
            "code": "23CSE311",
            "name": "Software Engineering",
            "type": "theory",
            "credits": 4,
            "duration_hours": 1,
            "sections": 1,
            "hours_per_week": 3,
            "semester_number": 6,
            "batch_year": 3,
            "theory_hours": 3,
            "lab_hours": 0,
            "tutorial_hours": 0,
            "faculty_id": faculty_id,
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert course_response.status_code == 201
    course_id = course_response.json()["id"]

    publish_response = client.put(
        "/api/timetable/official?versionLabel=v-map",
        json={
            "facultyData": [
                {
                    "id": faculty_id,
                    "name": "Faculty Mapping",
                    "department": "CSE",
                    "workloadHours": 0,
                    "maxHours": 16,
                    "availability": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                    "email": faculty_payload["email"],
                }
            ],
            "courseData": [
                {
                    "id": course_id,
                    "code": "23CSE311",
                    "name": "Software Engineering",
                    "type": "theory",
                    "credits": 4,
                    "facultyId": faculty_id,
                    "duration": 1,
                    "hoursPerWeek": 3,
                    "semesterNumber": 6,
                    "batchYear": 3,
                    "theoryHours": 3,
                    "labHours": 0,
                    "tutorialHours": 0,
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
                    "id": "slot-map-1",
                    "day": "Monday",
                    "startTime": "08:50",
                    "endTime": "09:40",
                    "courseId": course_id,
                    "roomId": room_id,
                    "facultyId": faculty_id,
                    "section": "A",
                    "studentCount": 60,
                },
                {
                    "id": "slot-map-2",
                    "day": "Tuesday",
                    "startTime": "08:50",
                    "endTime": "09:40",
                    "courseId": course_id,
                    "roomId": room_id,
                    "facultyId": faculty_id,
                    "section": "A",
                    "studentCount": 60,
                },
                {
                    "id": "slot-map-3",
                    "day": "Wednesday",
                    "startTime": "08:50",
                    "endTime": "09:40",
                    "courseId": course_id,
                    "roomId": room_id,
                    "facultyId": faculty_id,
                    "section": "A",
                    "studentCount": 60,
                },
            ],
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert publish_response.status_code == 200

    admin_mapping = client.get(
        "/api/timetable/official/faculty-mapping",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert admin_mapping.status_code == 200
    mapping_rows = admin_mapping.json()
    assert len(mapping_rows) == 1
    assert mapping_rows[0]["faculty_id"] == faculty_id
    assert mapping_rows[0]["assignments"][0]["course_code"] == "23CSE311"

    faculty_mapping = client.get(
        "/api/timetable/official/faculty-mapping",
        headers={"Authorization": f"Bearer {faculty_token}"},
    )
    assert faculty_mapping.status_code == 200
    assert len(faculty_mapping.json()) == 1

    student_mapping = client.get(
        "/api/timetable/official/faculty-mapping",
        headers={"Authorization": f"Bearer {student_token}"},
    )
    assert student_mapping.status_code == 403


def test_course_credit_split_validation(client):
    admin_payload = {
        "name": "Admin Credits",
        "email": "admin-credit-split@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    assert register_user(client, admin_payload).status_code == 201
    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")

    invalid_course = client.post(
        "/api/courses",
        json={
            "code": "23CSE999",
            "name": "Invalid Split",
            "type": "theory",
            "credits": 3,
            "duration_hours": 1,
            "sections": 1,
            "hours_per_week": 3,
            "semester_number": 5,
            "batch_year": 3,
            "theory_hours": 2,
            "lab_hours": 1,
            "tutorial_hours": 1,
            "faculty_id": None,
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert invalid_course.status_code == 422
    assert "hours_per_week must equal theory_hours + lab_hours + tutorial_hours" in str(invalid_course.json())
