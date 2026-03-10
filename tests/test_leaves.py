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


def test_faculty_leave_request_flow(client):
    admin_payload = {
        "name": "Admin User",
        "email": "admin-leaves@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    faculty_payload = {
        "name": "Faculty User",
        "email": "faculty-leaves@example.com",
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

    leave_date = (date.today() + timedelta(days=2)).isoformat()
    create_response = client.post(
        "/api/leaves",
        json={
            "leave_date": leave_date,
            "leave_type": "sick",
            "reason": "Medical appointment",
        },
        headers={"Authorization": f"Bearer {faculty_token}"},
    )
    assert create_response.status_code == 201
    leave_id = create_response.json()["id"]

    faculty_list = client.get(
        "/api/leaves",
        headers={"Authorization": f"Bearer {faculty_token}"},
    )
    assert faculty_list.status_code == 200
    assert any(item["id"] == leave_id for item in faculty_list.json())

    admin_list = client.get(
        "/api/leaves",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert admin_list.status_code == 200
    assert any(item["id"] == leave_id for item in admin_list.json())

    update_response = client.put(
        f"/api/leaves/{leave_id}/status",
        json={"status": "approved", "admin_comment": "Approved"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert update_response.status_code == 200
    assert update_response.json()["status"] == "approved"


def test_leave_approval_creates_substitute_offer_and_faculty_can_accept(client):
    admin_payload = {
        "name": "Admin Offer Flow",
        "email": "admin-offer-flow@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    leave_faculty_payload = {
        "name": "Leave Faculty",
        "email": "leave-faculty-offer@example.com",
        "password": "password123",
        "role": "faculty",
        "department": "CSE",
    }
    substitute_payload = {
        "name": "Offer Faculty",
        "email": "substitute-faculty-offer@example.com",
        "password": "password123",
        "role": "faculty",
        "department": "CSE",
        "preferred_subject_codes": ["CSOFFER101"],
    }
    register_user(client, admin_payload)
    register_user(client, leave_faculty_payload)
    register_user(client, substitute_payload)

    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")
    leave_token = login_user(client, leave_faculty_payload["email"], leave_faculty_payload["password"], "faculty")
    substitute_token = login_user(client, substitute_payload["email"], substitute_payload["password"], "faculty")

    leave_profile = client.get("/api/faculty/me", headers={"Authorization": f"Bearer {leave_token}"})
    substitute_profile = client.get("/api/faculty/me", headers={"Authorization": f"Bearer {substitute_token}"})
    assert leave_profile.status_code == 200
    assert substitute_profile.status_code == 200
    leave_faculty_id = leave_profile.json()["id"]
    substitute_faculty_id = substitute_profile.json()["id"]

    publish_response = client.put(
        "/api/timetable/official?versionLabel=v-offer-flow-base",
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
                    "name": substitute_payload["name"],
                    "department": "CSE",
                    "workloadHours": 0,
                    "maxHours": 20,
                    "availability": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                    "email": substitute_payload["email"],
                },
            ],
            "courseData": [
                {
                    "id": "course-offer-flow",
                    "code": "CSOFFER101",
                    "name": "Offer Systems",
                    "type": "theory",
                    "credits": 3,
                    "facultyId": leave_faculty_id,
                    "duration": 1,
                    "hoursPerWeek": 1,
                }
            ],
            "roomData": [
                {
                    "id": "room-offer-flow",
                    "name": "A101",
                    "capacity": 70,
                    "type": "lecture",
                    "building": "Main",
                }
            ],
            "timetableData": [
                {
                    "id": "slot-offer-flow",
                    "day": "Monday",
                    "startTime": "08:50",
                    "endTime": "09:40",
                    "courseId": "course-offer-flow",
                    "roomId": "room-offer-flow",
                    "facultyId": leave_faculty_id,
                    "section": "A",
                    "studentCount": 60,
                }
            ],
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert publish_response.status_code == 200

    leave_date = next_weekday(date.today() + timedelta(days=1), 0).isoformat()
    leave_create = client.post(
        "/api/leaves",
        json={
            "leave_date": leave_date,
            "leave_type": "casual",
            "reason": "Offer flow test",
        },
        headers={"Authorization": f"Bearer {leave_token}"},
    )
    assert leave_create.status_code == 201
    leave_id = leave_create.json()["id"]

    approve_response = client.put(
        f"/api/leaves/{leave_id}/status",
        json={"status": "approved"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert approve_response.status_code == 200

    pending_offers = client.get(
        "/api/leaves/substitute-offers?status=pending",
        headers={"Authorization": f"Bearer {substitute_token}"},
    )
    assert pending_offers.status_code == 200
    assert len(pending_offers.json()) == 1

    respond_offer = client.post(
        f"/api/leaves/substitute-offers/{pending_offers.json()[0]['id']}/respond",
        json={"decision": "accept"},
        headers={"Authorization": f"Bearer {substitute_token}"},
    )
    assert respond_offer.status_code == 200
    assert respond_offer.json()["status"] == "accepted"

    official_after = client.get("/api/timetable/official", headers={"Authorization": f"Bearer {admin_token}"})
    assert official_after.status_code == 200
    slot = next(item for item in official_after.json()["timetableData"] if item["id"] == "slot-offer-flow")
    assert slot["facultyId"] == substitute_faculty_id


def test_rejecting_last_substitute_offer_triggers_reschedule(client):
    admin_payload = {
        "name": "Admin Reschedule",
        "email": "admin-reschedule@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    leave_faculty_payload = {
        "name": "Faculty Reschedule Leave",
        "email": "faculty-reschedule-leave@example.com",
        "password": "password123",
        "role": "faculty",
        "department": "CSE",
    }
    substitute_payload = {
        "name": "Faculty Rejects",
        "email": "faculty-reschedule-substitute@example.com",
        "password": "password123",
        "role": "faculty",
        "department": "CSE",
        "preferred_subject_codes": ["CSRESCH101"],
    }
    register_user(client, admin_payload)
    register_user(client, leave_faculty_payload)
    register_user(client, substitute_payload)

    admin_token = login_user(client, admin_payload["email"], admin_payload["password"], "admin")
    leave_token = login_user(client, leave_faculty_payload["email"], leave_faculty_payload["password"], "faculty")
    substitute_token = login_user(client, substitute_payload["email"], substitute_payload["password"], "faculty")

    leave_profile = client.get("/api/faculty/me", headers={"Authorization": f"Bearer {leave_token}"})
    substitute_profile = client.get("/api/faculty/me", headers={"Authorization": f"Bearer {substitute_token}"})
    assert leave_profile.status_code == 200
    assert substitute_profile.status_code == 200
    leave_faculty_id = leave_profile.json()["id"]
    substitute_faculty_id = substitute_profile.json()["id"]

    publish_response = client.put(
        "/api/timetable/official?versionLabel=v-reschedule-flow-base",
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
                    "name": substitute_payload["name"],
                    "department": "CSE",
                    "workloadHours": 0,
                    "maxHours": 20,
                    "availability": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
                    "email": substitute_payload["email"],
                },
            ],
            "courseData": [
                {
                    "id": "course-reschedule-flow",
                    "code": "CSRESCH101",
                    "name": "Reschedule Systems",
                    "type": "theory",
                    "credits": 3,
                    "facultyId": leave_faculty_id,
                    "duration": 1,
                    "hoursPerWeek": 1,
                }
            ],
            "roomData": [
                {
                    "id": "room-reschedule-flow",
                    "name": "A101",
                    "capacity": 70,
                    "type": "lecture",
                    "building": "Main",
                }
            ],
            "timetableData": [
                {
                    "id": "slot-reschedule-flow",
                    "day": "Monday",
                    "startTime": "08:50",
                    "endTime": "09:40",
                    "courseId": "course-reschedule-flow",
                    "roomId": "room-reschedule-flow",
                    "facultyId": leave_faculty_id,
                    "section": "A",
                    "studentCount": 60,
                }
            ],
        },
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert publish_response.status_code == 200

    leave_date = next_weekday(date.today() + timedelta(days=1), 0).isoformat()
    leave_create = client.post(
        "/api/leaves",
        json={
            "leave_date": leave_date,
            "leave_type": "casual",
            "reason": "Fallback reschedule test",
        },
        headers={"Authorization": f"Bearer {leave_token}"},
    )
    assert leave_create.status_code == 201
    leave_id = leave_create.json()["id"]

    approve_response = client.put(
        f"/api/leaves/{leave_id}/status",
        json={"status": "approved"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert approve_response.status_code == 200

    pending_offers = client.get(
        "/api/leaves/substitute-offers?status=pending",
        headers={"Authorization": f"Bearer {substitute_token}"},
    )
    assert pending_offers.status_code == 200
    assert len(pending_offers.json()) == 1

    reject_offer = client.post(
        f"/api/leaves/substitute-offers/{pending_offers.json()[0]['id']}/respond",
        json={"decision": "reject"},
        headers={"Authorization": f"Bearer {substitute_token}"},
    )
    assert reject_offer.status_code == 200
    assert reject_offer.json()["status"] == "rejected"

    official_after = client.get("/api/timetable/official", headers={"Authorization": f"Bearer {admin_token}"})
    assert official_after.status_code == 200
    slot = next(item for item in official_after.json()["timetableData"] if item["id"] == "slot-reschedule-flow")
    assert slot["facultyId"] == leave_faculty_id
    assert (slot["day"], slot["startTime"], slot["endTime"]) != ("Monday", "08:50", "09:40")
