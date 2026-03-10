from app.services.email import EmailDeliveryError


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


def test_smtp_config_endpoint_for_admin(client):
    payload = {
        "name": "SMTP Admin",
        "email": "smtp-admin@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    register_user(client, payload)
    token = login_user(client, payload["email"], payload["password"], "admin")

    response = client.get("/api/settings/smtp/config", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    data = response.json()
    assert "configured" in data
    assert "port" in data
    assert "use_tls" in data
    assert "use_ssl" in data


def test_smtp_test_endpoint_returns_503_with_actionable_message(client, monkeypatch):
    payload = {
        "name": "SMTP Admin",
        "email": "smtp-admin-2@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    register_user(client, payload)
    token = login_user(client, payload["email"], payload["password"], "admin")

    def fake_send_email(*, to_email, subject, text_content, html_content=None):
        raise EmailDeliveryError("SMTP authentication failed")

    monkeypatch.setattr("app.api.routes.settings.send_email", fake_send_email)

    response = client.post(
        "/api/settings/smtp/test",
        json={"to_email": "recipient@example.com"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 503
    assert "SMTP authentication failed" in response.json()["detail"]


def test_smtp_test_endpoint_returns_rate_limit_error_message(client, monkeypatch):
    payload = {
        "name": "SMTP Admin",
        "email": "smtp-admin-rate-limit@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    register_user(client, payload)
    token = login_user(client, payload["email"], payload["password"], "admin")

    def fake_send_email(*, to_email, subject, text_content, html_content=None):
        raise EmailDeliveryError("SMTP sender rate limited")

    monkeypatch.setattr("app.api.routes.settings.send_email", fake_send_email)

    response = client.post(
        "/api/settings/smtp/test",
        json={"to_email": "recipient@example.com"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 503
    assert "rate-limited" in response.json()["detail"].lower()
