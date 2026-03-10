
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


def test_timetable_validation_rejects_bad_references(client):
    admin_payload = {
        "name": "Admin User",
        "email": "admin@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    register_user(client, admin_payload)
    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")

    bad_payload = {
        "facultyData": [],
        "courseData": [],
        "roomData": [],
        "timetableData": [
            {
                "id": "ts-1",
                "day": "Monday",
                "startTime": "09:00",
                "endTime": "10:00",
                "courseId": "missing-course",
                "roomId": "missing-room",
                "facultyId": "missing-faculty",
                "section": "A",
            }
        ],
    }

    response = client.put(
        "/api/timetable/official",
        json=bad_payload,
        headers={"Authorization": f"Bearer {admin_token}"},
    )

    assert response.status_code == 422


def test_timetable_validation_rejects_primary_faculty_in_assistant_list(client):
    admin_payload = {
        "name": "Admin User",
        "email": "admin-assistant-validation@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    faculty_payload = {
        "name": "Faculty User",
        "email": "faculty-assistant-validation@example.com",
        "password": "password123",
        "role": "faculty",
        "department": "CSE",
    }

    register_user(client, admin_payload)
    register_user(client, faculty_payload)

    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")
    faculty_token = login_user(client, faculty_payload["email"], faculty_payload["password"], "faculty")
    faculty_profile = client.get(
        "/api/faculty/me",
        headers={"Authorization": f"Bearer {faculty_token}"},
    )
    assert faculty_profile.status_code == 200
    faculty_id = faculty_profile.json()["id"]

    invalid_payload = {
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
        ],
        "courseData": [
            {
                "id": "c-lab",
                "code": "CSLAB1",
                "name": "Lab Course",
                "type": "lab",
                "credits": 1,
                "facultyId": faculty_id,
                "duration": 1,
                "hoursPerWeek": 2,
                "theoryHours": 0,
                "tutorialHours": 0,
                "labHours": 2,
            },
        ],
        "roomData": [
            {
                "id": "r-lab",
                "name": "Lab 1",
                "capacity": 70,
                "type": "lab",
                "building": "Main",
            },
        ],
        "timetableData": [
            {
                "id": "slot-lab-1",
                "day": "Monday",
                "startTime": "09:00",
                "endTime": "09:50",
                "courseId": "c-lab",
                "roomId": "r-lab",
                "facultyId": faculty_id,
                "assistantFacultyIds": [faculty_id],
                "section": "A",
                "batch": "B1",
                "studentCount": 35,
                "sessionType": "lab",
            },
            {
                "id": "slot-lab-2",
                "day": "Monday",
                "startTime": "09:50",
                "endTime": "10:40",
                "courseId": "c-lab",
                "roomId": "r-lab",
                "facultyId": faculty_id,
                "assistantFacultyIds": [faculty_id],
                "section": "A",
                "batch": "B1",
                "studentCount": 35,
                "sessionType": "lab",
            },
        ],
    }

    response = client.put(
        "/api/timetable/official",
        json=invalid_payload,
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert response.status_code == 422


def test_mixed_ltp_keeps_lecture_single_and_practical_contiguous(client):
    admin_payload = {
        "name": "Admin Mixed LTP",
        "email": "admin-mixed-ltp@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    register_user(client, admin_payload)
    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")

    policy_response = client.put(
        "/api/settings/schedule-policy",
        json={
            "period_minutes": 50,
            "lab_contiguous_slots": 2,
            "breaks": [
                {"name": "Short Break", "start_time": "10:30", "end_time": "10:45"},
                {"name": "Lunch Break", "start_time": "12:25", "end_time": "13:15"},
            ],
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert policy_response.status_code == 200

    payload = {
        "facultyData": [
            {
                "id": "f-1",
                "name": "Prof Mixed",
                "department": "CSE",
                "workloadHours": 0,
                "maxHours": 20,
                "availability": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                "email": "prof-mixed@example.com",
            }
        ],
        "courseData": [
            {
                "id": "c-mixed",
                "code": "CSE-MIX",
                "name": "Mixed LTP Course",
                "type": "theory",
                "credits": 4,
                "facultyId": "f-1",
                "duration": 1,
                "hoursPerWeek": 5,
                "theoryHours": 3,
                "tutorialHours": 0,
                "labHours": 2,
                "batchSegregation": False,
                "practicalContiguousSlots": 2,
            }
        ],
        "roomData": [
            {
                "id": "r-lecture",
                "name": "L-101",
                "capacity": 80,
                "type": "lecture",
                "building": "Main",
            },
            {
                "id": "r-lab",
                "name": "LAB-1",
                "capacity": 80,
                "type": "lab",
                "building": "Main",
            },
        ],
        "timetableData": [
            {
                "id": "slot-l-1",
                "day": "Monday",
                "startTime": "08:50",
                "endTime": "09:40",
                "courseId": "c-mixed",
                "roomId": "r-lecture",
                "facultyId": "f-1",
                "section": "A",
                "studentCount": 60,
                "sessionType": "theory",
            },
            {
                "id": "slot-l-2",
                "day": "Tuesday",
                "startTime": "08:50",
                "endTime": "09:40",
                "courseId": "c-mixed",
                "roomId": "r-lecture",
                "facultyId": "f-1",
                "section": "A",
                "studentCount": 60,
                "sessionType": "theory",
            },
            {
                "id": "slot-l-3",
                "day": "Wednesday",
                "startTime": "08:50",
                "endTime": "09:40",
                "courseId": "c-mixed",
                "roomId": "r-lecture",
                "facultyId": "f-1",
                "section": "A",
                "studentCount": 60,
                "sessionType": "theory",
            },
            {
                "id": "slot-p-1",
                "day": "Thursday",
                "startTime": "08:50",
                "endTime": "09:40",
                "courseId": "c-mixed",
                "roomId": "r-lab",
                "facultyId": "f-1",
                "section": "A",
                "studentCount": 60,
                "sessionType": "lab",
            },
            {
                "id": "slot-p-2",
                "day": "Thursday",
                "startTime": "09:40",
                "endTime": "10:30",
                "courseId": "c-mixed",
                "roomId": "r-lab",
                "facultyId": "f-1",
                "section": "A",
                "studentCount": 60,
                "sessionType": "lab",
            },
        ],
    }

    response = client.put(
        "/api/timetable/official",
        json=payload,
        headers={"Authorization": f"Bearer {admin_token}"},
    )

    assert response.status_code == 200
