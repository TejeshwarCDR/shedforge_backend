
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


def test_password_reset_flow(client):
    payload = {
        "name": "Admin User",
        "email": "admin@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    register_user(client, payload)

    reset_request = client.post("/api/auth/password/forgot", json={"email": payload["email"]})
    assert reset_request.status_code == 200
    token = reset_request.json().get("reset_token")
    assert token

    reset_confirm = client.post(
        "/api/auth/password/reset",
        json={"token": token, "new_password": "newpassword123"},
    )
    assert reset_confirm.status_code == 200

    login_response = client.post(
        "/api/auth/login",
        json={"email": payload["email"], "password": "newpassword123", "role": "admin"},
    )
    assert login_response.status_code == 200


def test_password_change_flow(client):
    payload = {
        "name": "Scheduler User",
        "email": "scheduler@example.com",
        "password": "password123",
        "role": "scheduler",
        "department": "Scheduling",
    }
    register_user(client, payload)
    token = login_user(client, payload["email"], payload["password"], "scheduler")

    change_response = client.post(
        "/api/auth/password/change",
        json={"current_password": "password123", "new_password": "updatedpass123"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert change_response.status_code == 200

    login_response = client.post(
        "/api/auth/login",
        json={"email": payload["email"], "password": "updatedpass123", "role": "scheduler"},
    )
    assert login_response.status_code == 200
