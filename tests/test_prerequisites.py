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


def test_prerequisite_constraints_across_program_assignment_generation_and_publish(client):
    admin_payload = {
        "name": "Admin Prereq",
        "email": "admin-prereq@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    register_user(client, admin_payload)
    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")

    create_program = client.post(
        "/api/programs",
        json={
            "name": "B.Tech CSE",
            "code": "CSE",
            "department": "CSE",
            "degree": "BS",
            "duration_years": 4,
            "sections": 1,
            "total_students": 60,
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert create_program.status_code == 201
    program_id = create_program.json()["id"]

    for term_number in (1, 2):
        term_response = client.post(
            f"/api/programs/{program_id}/terms",
            json={
                "term_number": term_number,
                "name": f"Semester {term_number}",
                "credits_required": 3,
            },
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert term_response.status_code == 201

    section_response = client.post(
        f"/api/programs/{program_id}/sections",
        json={"term_number": 2, "name": "A", "capacity": 50},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert section_response.status_code == 201

    intro_course = client.post(
        "/api/courses",
        json={
            "code": "CS101",
            "name": "Programming Fundamentals",
            "type": "theory",
            "credits": 3,
            "duration_hours": 1,
            "sections": 1,
            "hours_per_week": 1,
            "faculty_id": None,
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert intro_course.status_code == 201
    intro_course_id = intro_course.json()["id"]

    advanced_course = client.post(
        "/api/courses",
        json={
            "code": "CS201",
            "name": "Advanced Programming",
            "type": "theory",
            "credits": 3,
            "duration_hours": 1,
            "sections": 1,
            "hours_per_week": 1,
            "faculty_id": None,
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert advanced_course.status_code == 201
    advanced_course_id = advanced_course.json()["id"]

    invalid_prereq_add = client.post(
        f"/api/programs/{program_id}/courses",
        json={
            "term_number": 2,
            "course_id": advanced_course_id,
            "is_required": True,
            "prerequisite_course_ids": [intro_course_id],
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert invalid_prereq_add.status_code == 400
    assert "earlier terms" in invalid_prereq_add.json()["detail"]

    intro_program_course = client.post(
        f"/api/programs/{program_id}/courses",
        json={
            "term_number": 1,
            "course_id": intro_course_id,
            "is_required": True,
            "prerequisite_course_ids": [],
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert intro_program_course.status_code == 201
    intro_program_course_id = intro_program_course.json()["id"]

    valid_prereq_add = client.post(
        f"/api/programs/{program_id}/courses",
        json={
            "term_number": 2,
            "course_id": advanced_course_id,
            "is_required": True,
            "prerequisite_course_ids": [intro_course_id],
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert valid_prereq_add.status_code == 201
    assert valid_prereq_add.json()["prerequisite_course_ids"] == [intro_course_id]

    create_self_prereq = client.post(
        "/api/courses",
        json={
            "code": "CS202",
            "name": "Self Reference Course",
            "type": "theory",
            "credits": 3,
            "duration_hours": 1,
            "sections": 1,
            "hours_per_week": 1,
            "faculty_id": None,
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert create_self_prereq.status_code == 201
    self_course_id = create_self_prereq.json()["id"]

    invalid_self_prereq = client.post(
        f"/api/programs/{program_id}/courses",
        json={
            "term_number": 2,
            "course_id": self_course_id,
            "is_required": True,
            "prerequisite_course_ids": [self_course_id],
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert invalid_self_prereq.status_code == 400
    assert "cannot be a prerequisite of itself" in invalid_self_prereq.json()["detail"]

    delete_intro_mapping = client.delete(
        f"/api/programs/{program_id}/courses/{intro_program_course_id}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert delete_intro_mapping.status_code == 200

    generate_response = client.post(
        "/api/timetable/generate",
        json={
            "program_id": program_id,
            "term_number": 2,
            "alternative_count": 1,
            "persist_official": False,
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert generate_response.status_code == 400
    assert "Prerequisite constraints are not satisfied" in generate_response.json()["detail"]

    publish_response = client.put(
        "/api/timetable/official",
        json={
            "programId": program_id,
            "termNumber": 2,
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
                    "id": advanced_course_id,
                    "code": "CS201",
                    "name": "Advanced Programming",
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
                    "startTime": "08:50",
                    "endTime": "09:40",
                    "courseId": advanced_course_id,
                    "roomId": "r1",
                    "facultyId": "f1",
                    "section": "A",
                    "studentCount": 50,
                }
            ],
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert publish_response.status_code == 400
    assert "Prerequisite constraints are not satisfied" in publish_response.json()["detail"]
