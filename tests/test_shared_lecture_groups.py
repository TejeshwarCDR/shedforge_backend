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


def create_program_and_term(client, token):
    program = client.post(
        "/api/programs",
        json={
            "name": "B.Tech CSE",
            "code": "CSE-SHARED",
            "department": "CSE",
            "degree": "BS",
            "duration_years": 4,
            "sections": 2,
            "total_students": 100,
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert program.status_code == 201
    program_id = program.json()["id"]

    term = client.post(
        f"/api/programs/{program_id}/terms",
        json={"term_number": 3, "name": "Semester 3", "credits_required": 0},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert term.status_code == 201
    return program_id


def build_publish_payload(program_id: str, course_id: str, *, aligned: bool, room_capacity: int) -> dict:
    second_start = "08:50" if aligned else "09:40"
    second_end = "09:40" if aligned else "10:30"
    return {
        "programId": program_id,
        "termNumber": 3,
        "facultyData": [
            {
                "id": "f1",
                "name": "Faculty One",
                "department": "CSE",
                "workloadHours": 0,
                "maxHours": 20,
                "availability": ["Monday", "Tuesday"],
                "email": "faculty1@example.com",
                "currentWorkload": 0,
            }
        ],
        "courseData": [
            {
                "id": course_id,
                "code": "CS301",
                "name": "Shared Lecture Course",
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
                "capacity": room_capacity,
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
                "startTime": "08:50",
                "endTime": "09:40",
                "courseId": course_id,
                "roomId": "r1",
                "facultyId": "f1",
                "section": "A",
                "studentCount": 30,
            },
            {
                "id": "slot-2",
                "day": "Monday",
                "startTime": second_start,
                "endTime": second_end,
                "courseId": course_id,
                "roomId": "r1",
                "facultyId": "f1",
                "section": "B",
                "studentCount": 30,
            },
        ],
    }


def test_shared_lecture_group_validation_and_publish_behavior(client):
    admin_payload = {
        "name": "Admin Shared",
        "email": "admin-shared@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    register_user(client, admin_payload)
    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")

    program_id = create_program_and_term(client, admin_token)

    for section_name in ("A", "B"):
        section_response = client.post(
            f"/api/programs/{program_id}/sections",
            json={"term_number": 3, "name": section_name, "capacity": 30},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert section_response.status_code == 201

    course = client.post(
        "/api/courses",
        json={
            "code": "CS301",
            "name": "Shared Lecture Course",
            "type": "theory",
            "credits": 3,
            "duration_hours": 1,
            "sections": 1,
            "hours_per_week": 1,
            "faculty_id": None,
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert course.status_code == 201
    course_id = course.json()["id"]

    mapped_course = client.post(
        f"/api/programs/{program_id}/courses",
        json={
            "term_number": 3,
            "course_id": course_id,
            "is_required": True,
            "prerequisite_course_ids": [],
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert mapped_course.status_code == 201

    pre_group_publish = client.put(
        "/api/timetable/official",
        json=build_publish_payload(program_id, course_id, aligned=True, room_capacity=70),
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert pre_group_publish.status_code == 400
    assert "Room conflict" in pre_group_publish.json()["detail"]

    create_group = client.post(
        f"/api/programs/{program_id}/shared-lecture-groups",
        json={
            "term_number": 3,
            "name": "Shared CS301 for A+B",
            "course_id": course_id,
            "section_names": ["A", "B"],
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert create_group.status_code == 201

    insufficient_capacity_publish = client.put(
        "/api/timetable/official",
        json=build_publish_payload(program_id, course_id, aligned=True, room_capacity=50),
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert insufficient_capacity_publish.status_code == 400
    assert "exceeds room capacity" in insufficient_capacity_publish.json()["detail"]

    unsynced_publish = client.put(
        "/api/timetable/official",
        json=build_publish_payload(program_id, course_id, aligned=False, room_capacity=70),
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert unsynced_publish.status_code == 400
    assert "requires synchronized slots" in unsynced_publish.json()["detail"]

    valid_publish = client.put(
        "/api/timetable/official",
        json=build_publish_payload(program_id, course_id, aligned=True, room_capacity=70),
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert valid_publish.status_code == 200
