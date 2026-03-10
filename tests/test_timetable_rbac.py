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


def test_timetable_access_control(client):
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

    payload = {
        "facultyData": [],
        "courseData": [],
        "roomData": [],
        "timetableData": [],
    }

    admin_response = client.put(
        "/api/timetable/official",
        json=payload,
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert admin_response.status_code == 200

    student_response = client.put(
        "/api/timetable/official",
        json=payload,
        headers={"Authorization": f"Bearer {student_token}"},
    )
    assert student_response.status_code == 403

    get_response = client.get(
        "/api/timetable/official",
        headers={"Authorization": f"Bearer {student_token}"},
    )
    assert get_response.status_code == 200


def test_official_timetable_is_scoped_for_student_and_faculty(client):
    admin_payload = {
        "name": "Admin User",
        "email": "admin-scope@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    faculty_payload = {
        "name": "Faculty User",
        "email": "faculty-scope@example.com",
        "password": "password123",
        "role": "faculty",
        "department": "CSE",
    }
    student_payload = {
        "name": "Student User",
        "email": "student-scope@example.com",
        "password": "password123",
        "role": "student",
        "department": "CSE",
        "section_name": "A",
    }
    register_user(client, admin_payload)
    register_user(client, faculty_payload)
    register_user(client, student_payload)

    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")
    faculty_token = login_user(client, faculty_payload["email"], faculty_payload["password"], "faculty")
    student_token = login_user(client, student_payload["email"], student_payload["password"], "student")

    faculty_profile = client.get(
        "/api/faculty/me",
        headers={"Authorization": f"Bearer {faculty_token}"},
    )
    assert faculty_profile.status_code == 200
    faculty_id = faculty_profile.json()["id"]

    payload = {
        "facultyData": [
            {
                "id": faculty_id,
                "name": "Faculty User",
                "department": "CSE",
                "workloadHours": 0,
                "maxHours": 20,
                "availability": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                "email": faculty_payload["email"],
            },
            {
                "id": "f-other",
                "name": "Other Faculty",
                "department": "CSE",
                "workloadHours": 0,
                "maxHours": 20,
                "availability": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                "email": "other-faculty@example.com",
            },
        ],
        "courseData": [
            {
                "id": "c-a",
                "code": "CS101",
                "name": "Course A",
                "type": "theory",
                "credits": 3,
                "facultyId": faculty_id,
                "duration": 1,
                "hoursPerWeek": 1,
            },
            {
                "id": "c-b",
                "code": "CS201",
                "name": "Course B",
                "type": "theory",
                "credits": 3,
                "facultyId": "f-other",
                "duration": 1,
                "hoursPerWeek": 1,
            },
        ],
        "roomData": [
            {
                "id": "r-1",
                "name": "A101",
                "capacity": 70,
                "type": "lecture",
                "building": "Main",
            },
            {
                "id": "r-2",
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
                "startTime": "08:50",
                "endTime": "09:40",
                "courseId": "c-a",
                "roomId": "r-1",
                "facultyId": faculty_id,
                "section": "A",
                "studentCount": 60,
            },
            {
                "id": "slot-b",
                "day": "Monday",
                "startTime": "09:40",
                "endTime": "10:30",
                "courseId": "c-b",
                "roomId": "r-2",
                "facultyId": "f-other",
                "section": "B",
                "studentCount": 60,
            },
        ],
    }

    publish_response = client.put(
        "/api/timetable/official",
        json=payload,
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert publish_response.status_code == 200, publish_response.text

    faculty_view = client.get(
        "/api/timetable/official",
        headers={"Authorization": f"Bearer {faculty_token}"},
    )
    assert faculty_view.status_code == 200
    faculty_slots = faculty_view.json()["timetableData"]
    assert len(faculty_slots) == 1
    assert faculty_slots[0]["facultyId"] == faculty_id

    student_view = client.get(
        "/api/timetable/official",
        headers={"Authorization": f"Bearer {student_token}"},
    )
    assert student_view.status_code == 200
    student_slots = student_view.json()["timetableData"]
    assert len(student_slots) == 1
    assert student_slots[0]["section"] == "A"


def test_published_timetable_is_isolated_per_temporary_teacher_and_student_accounts(client):
    admin_payload = {
        "name": "Publish Admin",
        "email": "admin-temp-scope@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    faculty_one_payload = {
        "name": "Faculty Temp One",
        "email": "faculty.temp.one+qa@example.com",
        "password": "password123",
        "role": "faculty",
        "department": "CSE",
    }
    faculty_two_payload = {
        "name": "Faculty Temp Two",
        "email": "faculty.temp.two+qa@example.com",
        "password": "password123",
        "role": "faculty",
        "department": "CSE",
    }
    student_a_payload = {
        "name": "Student Temp A",
        "email": "student.temp.a+qa@example.com",
        "password": "password123",
        "role": "student",
        "department": "CSE",
        "section_name": "a",
    }
    student_b_payload = {
        "name": "Student Temp B",
        "email": "student.temp.b+qa@example.com",
        "password": "password123",
        "role": "student",
        "department": "CSE",
        "section_name": "B",
    }

    register_user(client, admin_payload)
    register_user(client, faculty_one_payload)
    register_user(client, faculty_two_payload)
    register_user(client, student_a_payload)
    register_user(client, student_b_payload)

    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")
    faculty_one_token = login_user(client, faculty_one_payload["email"], faculty_one_payload["password"], "faculty")
    faculty_two_token = login_user(client, faculty_two_payload["email"], faculty_two_payload["password"], "faculty")
    student_a_token = login_user(client, student_a_payload["email"], student_a_payload["password"], "student")
    student_b_token = login_user(client, student_b_payload["email"], student_b_payload["password"], "student")

    faculty_one_profile = client.get(
        "/api/faculty/me",
        headers={"Authorization": f"Bearer {faculty_one_token}"},
    )
    faculty_two_profile = client.get(
        "/api/faculty/me",
        headers={"Authorization": f"Bearer {faculty_two_token}"},
    )
    assert faculty_one_profile.status_code == 200
    assert faculty_two_profile.status_code == 200
    faculty_one_id = faculty_one_profile.json()["id"]
    faculty_two_id = faculty_two_profile.json()["id"]

    publish_payload = {
        "facultyData": [
            {
                "id": faculty_one_id,
                "name": "Faculty Temp One",
                "department": "CSE",
                "workloadHours": 0,
                "maxHours": 20,
                "availability": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                "email": faculty_one_payload["email"],
            },
            {
                "id": faculty_two_id,
                "name": "Faculty Temp Two",
                "department": "CSE",
                "workloadHours": 0,
                "maxHours": 20,
                "availability": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                "email": faculty_two_payload["email"],
            },
        ],
        "courseData": [
            {
                "id": "c-temp-a",
                "code": "CSE101A",
                "name": "Temp Course A",
                "type": "theory",
                "credits": 3,
                "facultyId": faculty_one_id,
                "duration": 1,
                "hoursPerWeek": 1,
            },
            {
                "id": "c-temp-b",
                "code": "CSE101B",
                "name": "Temp Course B",
                "type": "theory",
                "credits": 3,
                "facultyId": faculty_two_id,
                "duration": 1,
                "hoursPerWeek": 1,
            },
        ],
        "roomData": [
            {
                "id": "r-temp-a",
                "name": "A101",
                "capacity": 70,
                "type": "lecture",
                "building": "Main",
            },
            {
                "id": "r-temp-b",
                "name": "A102",
                "capacity": 70,
                "type": "lecture",
                "building": "Main",
            },
        ],
        "timetableData": [
            {
                "id": "slot-temp-a1",
                "day": "Monday",
                "startTime": "08:50",
                "endTime": "09:40",
                "courseId": "c-temp-a",
                "roomId": "r-temp-a",
                "facultyId": faculty_one_id,
                "section": "A",
                "studentCount": 60,
            },
            {
                "id": "slot-temp-b1",
                "day": "Tuesday",
                "startTime": "09:40",
                "endTime": "10:30",
                "courseId": "c-temp-b",
                "roomId": "r-temp-b",
                "facultyId": faculty_two_id,
                "section": "B",
                "studentCount": 60,
            },
        ],
    }

    publish_response = client.put(
        "/api/timetable/official",
        json=publish_payload,
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert publish_response.status_code == 200
    assert len(publish_response.json()["timetableData"]) == 2

    for token, expected_email, expected_role in [
        (faculty_one_token, faculty_one_payload["email"], "faculty"),
        (faculty_two_token, faculty_two_payload["email"], "faculty"),
        (student_a_token, student_a_payload["email"], "student"),
        (student_b_token, student_b_payload["email"], "student"),
    ]:
        me_response = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert me_response.status_code == 200
        me = me_response.json()
        assert me["email"] == expected_email
        assert me["role"] == expected_role

    faculty_one_view = client.get(
        "/api/timetable/official",
        headers={"Authorization": f"Bearer {faculty_one_token}"},
    ).json()
    assert {slot["facultyId"] for slot in faculty_one_view["timetableData"]} == {faculty_one_id}
    assert {slot["section"] for slot in faculty_one_view["timetableData"]} == {"A"}

    faculty_two_view = client.get(
        "/api/timetable/official",
        headers={"Authorization": f"Bearer {faculty_two_token}"},
    ).json()
    assert {slot["facultyId"] for slot in faculty_two_view["timetableData"]} == {faculty_two_id}
    assert {slot["section"] for slot in faculty_two_view["timetableData"]} == {"B"}

    student_a_view = client.get(
        "/api/timetable/official",
        headers={"Authorization": f"Bearer {student_a_token}"},
    ).json()
    assert {slot["section"] for slot in student_a_view["timetableData"]} == {"A"}
    assert {slot["facultyId"] for slot in student_a_view["timetableData"]} == {faculty_one_id}

    student_b_view = client.get(
        "/api/timetable/official",
        headers={"Authorization": f"Bearer {student_b_token}"},
    ).json()
    assert {slot["section"] for slot in student_b_view["timetableData"]} == {"B"}
    assert {slot["facultyId"] for slot in student_b_view["timetableData"]} == {faculty_two_id}


def test_faculty_scope_and_mapping_include_assistant_assignments(client):
    admin_payload = {
        "name": "Admin User",
        "email": "admin-assist-scope@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    primary_faculty_payload = {
        "name": "Primary Faculty",
        "email": "primary-assist-scope@example.com",
        "password": "password123",
        "role": "faculty",
        "department": "CSE",
    }
    assistant_faculty_payload = {
        "name": "Assistant Faculty",
        "email": "assistant-assist-scope@example.com",
        "password": "password123",
        "role": "faculty",
        "department": "CSE",
    }

    register_user(client, admin_payload)
    register_user(client, primary_faculty_payload)
    register_user(client, assistant_faculty_payload)

    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")
    primary_token = login_user(client, primary_faculty_payload["email"], primary_faculty_payload["password"], "faculty")
    assistant_token = login_user(client, assistant_faculty_payload["email"], assistant_faculty_payload["password"], "faculty")

    primary_profile = client.get(
        "/api/faculty/me",
        headers={"Authorization": f"Bearer {primary_token}"},
    )
    assistant_profile = client.get(
        "/api/faculty/me",
        headers={"Authorization": f"Bearer {assistant_token}"},
    )
    assert primary_profile.status_code == 200
    assert assistant_profile.status_code == 200
    primary_id = primary_profile.json()["id"]
    assistant_id = assistant_profile.json()["id"]

    payload = {
        "facultyData": [
            {
                "id": primary_id,
                "name": "Primary Faculty",
                "department": "CSE",
                "workloadHours": 0,
                "maxHours": 20,
                "availability": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                "email": primary_faculty_payload["email"],
            },
            {
                "id": assistant_id,
                "name": "Assistant Faculty",
                "department": "CSE",
                "workloadHours": 0,
                "maxHours": 20,
                "availability": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                "email": assistant_faculty_payload["email"],
            },
        ],
        "courseData": [
            {
                "id": "c-lab-assist",
                "code": "CSLAB201",
                "name": "Systems Lab",
                "type": "lab",
                "credits": 1,
                "facultyId": primary_id,
                "duration": 1,
                "hoursPerWeek": 2,
                "theoryHours": 0,
                "tutorialHours": 0,
                "labHours": 2,
            },
        ],
        "roomData": [
            {
                "id": "r-lab-assist",
                "name": "Lab A",
                "capacity": 60,
                "type": "lab",
                "building": "Main",
            },
        ],
        "timetableData": [
            {
                "id": "slot-assist-1",
                "day": "Wednesday",
                "startTime": "08:50",
                "endTime": "09:40",
                "courseId": "c-lab-assist",
                "roomId": "r-lab-assist",
                "facultyId": primary_id,
                "assistantFacultyIds": [assistant_id],
                "section": "A",
                "batch": "B1",
                "studentCount": 30,
                "sessionType": "lab",
            },
            {
                "id": "slot-assist-2",
                "day": "Wednesday",
                "startTime": "09:40",
                "endTime": "10:30",
                "courseId": "c-lab-assist",
                "roomId": "r-lab-assist",
                "facultyId": primary_id,
                "assistantFacultyIds": [assistant_id],
                "section": "A",
                "batch": "B1",
                "studentCount": 30,
                "sessionType": "lab",
            },
        ],
    }

    publish_response = client.put(
        "/api/timetable/official",
        json=payload,
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert publish_response.status_code == 200, publish_response.text

    assistant_view = client.get(
        "/api/timetable/official",
        headers={"Authorization": f"Bearer {assistant_token}"},
    )
    assert assistant_view.status_code == 200
    assistant_slots = assistant_view.json()["timetableData"]
    assert len(assistant_slots) == 2
    assert {slot["id"] for slot in assistant_slots} == {"slot-assist-1", "slot-assist-2"}

    assistant_mapping_response = client.get(
        "/api/timetable/official/faculty-mapping",
        headers={"Authorization": f"Bearer {assistant_token}"},
    )
    assert assistant_mapping_response.status_code == 200
    assistant_mapping = assistant_mapping_response.json()
    assert len(assistant_mapping) == 1
    assert assistant_mapping[0]["faculty_id"] == assistant_id
    assert all(item["assignmentRole"] == "assistant" for item in assistant_mapping[0]["assignments"])
