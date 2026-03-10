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


def create_course(client, token, *, code: str, name: str, course_type: str) -> str:
    response = client.post(
        "/api/courses",
        json={
            "code": code,
            "name": name,
            "type": course_type,
            "credits": 3,
            "duration_hours": 1,
            "sections": 1,
            "hours_per_week": 1,
            "faculty_id": None,
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 201
    return response.json()["id"]


def test_elective_overlap_groups_publish_validation_and_cleanup(client):
    admin_payload = {
        "name": "Admin Elective",
        "email": "admin-elective@example.com",
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
            "sections": 2,
            "total_students": 120,
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert create_program.status_code == 201
    program_id = create_program.json()["id"]

    create_term = client.post(
        f"/api/programs/{program_id}/terms",
        json={
            "term_number": 5,
            "name": "Semester 5",
            "credits_required": 0,
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert create_term.status_code == 201

    elective_one_id = create_course(
        client,
        admin_token,
        code="EL501",
        name="Elective One",
        course_type="elective",
    )
    elective_two_id = create_course(
        client,
        admin_token,
        code="EL502",
        name="Elective Two",
        course_type="elective",
    )
    theory_id = create_course(
        client,
        admin_token,
        code="TH501",
        name="Theory Course",
        course_type="theory",
    )

    elective_one_map = client.post(
        f"/api/programs/{program_id}/courses",
        json={
            "term_number": 5,
            "course_id": elective_one_id,
            "is_required": False,
            "prerequisite_course_ids": [],
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert elective_one_map.status_code == 201
    elective_one_map_id = elective_one_map.json()["id"]

    elective_two_map = client.post(
        f"/api/programs/{program_id}/courses",
        json={
            "term_number": 5,
            "course_id": elective_two_id,
            "is_required": False,
            "prerequisite_course_ids": [],
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert elective_two_map.status_code == 201
    elective_two_map_id = elective_two_map.json()["id"]

    theory_map = client.post(
        f"/api/programs/{program_id}/courses",
        json={
            "term_number": 5,
            "course_id": theory_id,
            "is_required": True,
            "prerequisite_course_ids": [],
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert theory_map.status_code == 201

    invalid_group = client.post(
        f"/api/programs/{program_id}/elective-groups",
        json={
            "term_number": 5,
            "name": "Invalid Group",
            "program_course_ids": [elective_one_map_id, theory_map.json()["id"]],
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert invalid_group.status_code == 400
    assert "only include elective courses" in invalid_group.json()["detail"]

    create_group = client.post(
        f"/api/programs/{program_id}/elective-groups",
        json={
            "term_number": 5,
            "name": "S5 Elective Basket A",
            "program_course_ids": [elective_one_map_id, elective_two_map_id],
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert create_group.status_code == 201
    group_id = create_group.json()["id"]

    list_groups = client.get(
        f"/api/programs/{program_id}/elective-groups?term_number=5",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert list_groups.status_code == 200
    assert len(list_groups.json()) == 1

    overlap_publish = client.put(
        "/api/timetable/official",
        json={
            "programId": program_id,
            "termNumber": 5,
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
                },
                {
                    "id": "f2",
                    "name": "Faculty Two",
                    "department": "CSE",
                    "workloadHours": 0,
                    "maxHours": 20,
                    "availability": ["Monday", "Tuesday"],
                    "email": "faculty2@example.com",
                    "currentWorkload": 0,
                },
            ],
            "courseData": [
                {
                    "id": elective_one_id,
                    "code": "EL501",
                    "name": "Elective One",
                    "type": "elective",
                    "credits": 3,
                    "facultyId": "f1",
                    "duration": 1,
                    "hoursPerWeek": 1,
                },
                {
                    "id": elective_two_id,
                    "code": "EL502",
                    "name": "Elective Two",
                    "type": "elective",
                    "credits": 3,
                    "facultyId": "f2",
                    "duration": 1,
                    "hoursPerWeek": 1,
                },
                {
                    "id": theory_id,
                    "code": "TH501",
                    "name": "Theory Course",
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
                {
                    "id": "r2",
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
                    "id": "slot-1",
                    "day": "Monday",
                    "startTime": "08:50",
                    "endTime": "09:40",
                    "courseId": elective_one_id,
                    "roomId": "r1",
                    "facultyId": "f1",
                    "section": "A",
                    "studentCount": 60,
                },
                {
                    "id": "slot-2",
                    "day": "Monday",
                    "startTime": "08:50",
                    "endTime": "09:40",
                    "courseId": elective_two_id,
                    "roomId": "r2",
                    "facultyId": "f2",
                    "section": "B",
                    "studentCount": 60,
                },
                {
                    "id": "slot-3",
                    "day": "Tuesday",
                    "startTime": "08:50",
                    "endTime": "09:40",
                    "courseId": theory_id,
                    "roomId": "r1",
                    "facultyId": "f1",
                    "section": "A",
                    "studentCount": 60,
                },
            ],
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert overlap_publish.status_code == 400
    assert "Elective overlap constraints violated" in overlap_publish.json()["detail"]

    non_overlap_publish = client.put(
        "/api/timetable/official",
        json={
            "programId": program_id,
            "termNumber": 5,
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
                },
                {
                    "id": "f2",
                    "name": "Faculty Two",
                    "department": "CSE",
                    "workloadHours": 0,
                    "maxHours": 20,
                    "availability": ["Monday", "Tuesday"],
                    "email": "faculty2@example.com",
                    "currentWorkload": 0,
                },
            ],
            "courseData": [
                {
                    "id": elective_one_id,
                    "code": "EL501",
                    "name": "Elective One",
                    "type": "elective",
                    "credits": 3,
                    "facultyId": "f1",
                    "duration": 1,
                    "hoursPerWeek": 1,
                },
                {
                    "id": elective_two_id,
                    "code": "EL502",
                    "name": "Elective Two",
                    "type": "elective",
                    "credits": 3,
                    "facultyId": "f2",
                    "duration": 1,
                    "hoursPerWeek": 1,
                },
                {
                    "id": theory_id,
                    "code": "TH501",
                    "name": "Theory Course",
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
                {
                    "id": "r2",
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
                    "id": "slot-1",
                    "day": "Monday",
                    "startTime": "08:50",
                    "endTime": "09:40",
                    "courseId": elective_one_id,
                    "roomId": "r1",
                    "facultyId": "f1",
                    "section": "A",
                    "studentCount": 60,
                },
                {
                    "id": "slot-2",
                    "day": "Monday",
                    "startTime": "09:40",
                    "endTime": "10:30",
                    "courseId": elective_two_id,
                    "roomId": "r2",
                    "facultyId": "f2",
                    "section": "B",
                    "studentCount": 60,
                },
                {
                    "id": "slot-3",
                    "day": "Tuesday",
                    "startTime": "08:50",
                    "endTime": "09:40",
                    "courseId": theory_id,
                    "roomId": "r1",
                    "facultyId": "f1",
                    "section": "A",
                    "studentCount": 60,
                },
            ],
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert non_overlap_publish.status_code == 200

    delete_group = client.delete(
        f"/api/programs/{program_id}/elective-groups/{group_id}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert delete_group.status_code == 200

    recreate_group = client.post(
        f"/api/programs/{program_id}/elective-groups",
        json={
            "term_number": 5,
            "name": "S5 Elective Basket B",
            "program_course_ids": [elective_one_map_id, elective_two_map_id],
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert recreate_group.status_code == 201

    delete_member_course = client.delete(
        f"/api/programs/{program_id}/courses/{elective_one_map_id}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert delete_member_course.status_code == 200

    list_groups_after_cleanup = client.get(
        f"/api/programs/{program_id}/elective-groups?term_number=5",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert list_groups_after_cleanup.status_code == 200
    assert list_groups_after_cleanup.json() == []
