
import re

import pytest
from fastapi import HTTPException
from sqlalchemy.exc import ProgrammingError

from app.services.email import EmailDeliveryError
from app.services.rate_limit import clear_rate_limiter


def test_register_login_logout(client):
    register_payload = {
        "name": "Admin User",
        "email": "admin@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }

    register_response = client.post("/api/auth/register", json=register_payload)
    assert register_response.status_code == 201
    data = register_response.json()
    assert data["email"] == register_payload["email"]
    assert data["role"] == register_payload["role"]

    login_response = client.post(
        "/api/auth/login",
        json={
            "email": register_payload["email"],
            "password": register_payload["password"],
            "role": "admin",
        },
    )
    assert login_response.status_code == 200
    login_data = login_response.json()
    assert "access_token" in login_data
    assert login_data["token_type"] == "bearer"

    token = login_data["access_token"]

    me_response = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_response.status_code == 200
    me_data = me_response.json()
    assert me_data["email"] == register_payload["email"]

    logout_response = client.post("/api/auth/logout", headers={"Authorization": f"Bearer {token}"})
    assert logout_response.status_code == 200
    assert logout_response.json()["success"] is True


def test_login_with_email_otp_verification(client, monkeypatch):
    register_payload = {
        "name": "Faculty User",
        "email": "faculty@example.com",
        "password": "password123",
        "role": "faculty",
        "department": "CSE",
    }
    register_response = client.post("/api/auth/register", json=register_payload)
    assert register_response.status_code == 201

    sent_mail: dict[str, str] = {}

    def fake_send_email(*, to_email: str, subject: str, text_content: str, html_content=None):
        sent_mail["to"] = to_email
        sent_mail["subject"] = subject
        sent_mail["body"] = text_content

    monkeypatch.setattr("app.api.routes.auth.send_email", fake_send_email)

    request_otp_response = client.post(
        "/api/auth/login/request-otp",
        json={
            "email": register_payload["email"],
            "password": register_payload["password"],
            "role": "faculty",
        },
    )
    assert request_otp_response.status_code == 200
    challenge = request_otp_response.json()
    assert "challenge_id" in challenge
    assert challenge["email"] == register_payload["email"]
    assert sent_mail["to"] == register_payload["email"]

    code_match = re.search(r"\b(\d{6})\b", sent_mail["body"])
    assert code_match is not None
    otp_code = code_match.group(1)

    verify_response = client.post(
        "/api/auth/login/verify-otp",
        json={
            "challenge_id": challenge["challenge_id"],
            "otp_code": otp_code,
        },
    )
    assert verify_response.status_code == 200
    token_data = verify_response.json()
    assert token_data["token_type"] == "bearer"
    assert "access_token" in token_data


def test_login_otp_returns_actionable_error_when_smtp_not_configured(client, monkeypatch):
    import app.api.routes.auth as auth_module

    register_payload = {
        "name": "Admin User",
        "email": "ops-admin@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    register_response = client.post("/api/auth/register", json=register_payload)
    assert register_response.status_code == 201

    def fake_send_email(*, to_email: str, subject: str, text_content: str, html_content=None):
        raise EmailDeliveryError("SMTP is not configured")

    monkeypatch.setattr("app.api.routes.auth.send_email", fake_send_email)

    original_log_to_terminal = auth_module.settings.login_otp_log_to_terminal
    original_terminal_fallback = auth_module.settings.login_otp_allow_terminal_fallback
    try:
        auth_module.settings.login_otp_log_to_terminal = False
        auth_module.settings.login_otp_allow_terminal_fallback = False
        request_otp_response = client.post(
            "/api/auth/login/request-otp",
            json={
                "email": register_payload["email"],
                "password": register_payload["password"],
                "role": "admin",
            },
        )
    finally:
        auth_module.settings.login_otp_log_to_terminal = original_log_to_terminal
        auth_module.settings.login_otp_allow_terminal_fallback = original_terminal_fallback
    assert request_otp_response.status_code == 503
    assert request_otp_response.json()["detail"] == (
        "Email service not configured. Set SMTP settings in backend/.env and restart backend."
    )


def test_login_otp_returns_actionable_error_when_smtp_auth_fails(client, monkeypatch):
    import app.api.routes.auth as auth_module

    register_payload = {
        "name": "Scheduler User",
        "email": "ops-scheduler@example.com",
        "password": "password123",
        "role": "scheduler",
        "department": "Administration",
    }
    register_response = client.post("/api/auth/register", json=register_payload)
    assert register_response.status_code == 201

    def fake_send_email(*, to_email: str, subject: str, text_content: str, html_content=None):
        raise EmailDeliveryError("SMTP authentication failed")

    monkeypatch.setattr("app.api.routes.auth.send_email", fake_send_email)

    original_log_to_terminal = auth_module.settings.login_otp_log_to_terminal
    original_terminal_fallback = auth_module.settings.login_otp_allow_terminal_fallback
    try:
        auth_module.settings.login_otp_log_to_terminal = False
        auth_module.settings.login_otp_allow_terminal_fallback = False
        request_otp_response = client.post(
            "/api/auth/login/request-otp",
            json={
                "email": register_payload["email"],
                "password": register_payload["password"],
                "role": "scheduler",
            },
        )
    finally:
        auth_module.settings.login_otp_log_to_terminal = original_log_to_terminal
        auth_module.settings.login_otp_allow_terminal_fallback = original_terminal_fallback
    assert request_otp_response.status_code == 503
    assert request_otp_response.json()["detail"] == (
        "Email authentication failed. Verify SMTP username/password (or app password) and try again."
    )


def test_login_otp_returns_actionable_error_when_smtp_rate_limited(client, monkeypatch):
    import app.api.routes.auth as auth_module

    register_payload = {
        "name": "Rate Limited User",
        "email": "ops-rate-limited@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    register_response = client.post("/api/auth/register", json=register_payload)
    assert register_response.status_code == 201

    def fake_send_email(*, to_email: str, subject: str, text_content: str, html_content=None):
        raise EmailDeliveryError("SMTP sender rate limited")

    monkeypatch.setattr("app.api.routes.auth.send_email", fake_send_email)

    original_log_to_terminal = auth_module.settings.login_otp_log_to_terminal
    original_terminal_fallback = auth_module.settings.login_otp_allow_terminal_fallback
    try:
        auth_module.settings.login_otp_log_to_terminal = False
        auth_module.settings.login_otp_allow_terminal_fallback = False
        request_otp_response = client.post(
            "/api/auth/login/request-otp",
            json={
                "email": register_payload["email"],
                "password": register_payload["password"],
                "role": "admin",
            },
        )
    finally:
        auth_module.settings.login_otp_log_to_terminal = original_log_to_terminal
        auth_module.settings.login_otp_allow_terminal_fallback = original_terminal_fallback
    assert request_otp_response.status_code == 503
    assert request_otp_response.json()["detail"] == (
        "Email sending limit reached for the configured SMTP account. Try again later or use another SMTP account."
    )


def test_login_otp_can_fallback_to_terminal_when_enabled(client, monkeypatch):
    import app.api.routes.auth as auth_module

    register_payload = {
        "name": "Fallback User",
        "email": "ops-terminal-fallback@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    register_response = client.post("/api/auth/register", json=register_payload)
    assert register_response.status_code == 201

    def fake_send_email(*, to_email: str, subject: str, text_content: str, html_content=None):
        raise EmailDeliveryError("SMTP sender rate limited")

    monkeypatch.setattr("app.api.routes.auth.send_email", fake_send_email)

    original_log_to_terminal = auth_module.settings.login_otp_log_to_terminal
    original_terminal_fallback = auth_module.settings.login_otp_allow_terminal_fallback
    try:
        auth_module.settings.login_otp_log_to_terminal = True
        auth_module.settings.login_otp_allow_terminal_fallback = True

        request_otp_response = client.post(
            "/api/auth/login/request-otp",
            json={
                "email": register_payload["email"],
                "password": register_payload["password"],
                "role": "admin",
            },
        )
    finally:
        auth_module.settings.login_otp_log_to_terminal = original_log_to_terminal
        auth_module.settings.login_otp_allow_terminal_fallback = original_terminal_fallback

    assert request_otp_response.status_code == 200
    data = request_otp_response.json()
    assert data["challenge_id"]
    assert "terminal log" in data["message"].lower()


def test_faculty_otp_login_recreates_missing_faculty_profile(client, monkeypatch):
    admin_payload = {
        "name": "Admin User",
        "email": "admin-faculty-map@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    faculty_payload = {
        "name": "Faculty User",
        "email": "faculty-remap@example.com",
        "password": "password123",
        "role": "faculty",
        "department": "CSE",
    }
    assert client.post("/api/auth/register", json=admin_payload).status_code == 201
    assert client.post("/api/auth/register", json=faculty_payload).status_code == 201

    admin_login = client.post(
        "/api/auth/login",
        json={
            "email": admin_payload["email"],
            "password": admin_payload["password"],
            "role": "admin",
        },
    )
    assert admin_login.status_code == 200
    admin_token = admin_login.json()["access_token"]

    list_response = client.get("/api/faculty", headers={"Authorization": f"Bearer {admin_token}"})
    assert list_response.status_code == 200
    faculty_items = list_response.json()
    faculty_profile = next(item for item in faculty_items if item["email"] == faculty_payload["email"])

    delete_response = client.delete(
        f"/api/faculty/{faculty_profile['id']}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert delete_response.status_code == 200

    sent_mail: dict[str, str] = {}

    def fake_send_email(*, to_email: str, subject: str, text_content: str, html_content=None):
        sent_mail["to"] = to_email
        sent_mail["subject"] = subject
        sent_mail["body"] = text_content

    monkeypatch.setattr("app.api.routes.auth.send_email", fake_send_email)

    otp_request = client.post(
        "/api/auth/login/request-otp",
        json={
            "email": faculty_payload["email"],
            "password": faculty_payload["password"],
            "role": "faculty",
        },
    )
    assert otp_request.status_code == 200
    challenge = otp_request.json()
    code_match = re.search(r"\b(\d{6})\b", sent_mail["body"])
    assert code_match is not None

    verify_response = client.post(
        "/api/auth/login/verify-otp",
        json={
            "challenge_id": challenge["challenge_id"],
            "otp_code": code_match.group(1),
        },
    )
    assert verify_response.status_code == 200
    faculty_token = verify_response.json()["access_token"]

    my_profile_response = client.get("/api/faculty/me", headers={"Authorization": f"Bearer {faculty_token}"})
    assert my_profile_response.status_code == 200
    assert my_profile_response.json()["email"] == faculty_payload["email"]


def test_login_rate_limit_blocks_excessive_attempts(client):
    clear_rate_limiter()
    register_payload = {
        "name": "Rate Limit User",
        "email": "rate-limit-user@example.com",
        "password": "password123",
        "role": "admin",
        "department": "Administration",
    }
    register_response = client.post("/api/auth/register", json=register_payload)
    assert register_response.status_code == 201

    import app.api.routes.auth as auth_module

    original_limit = auth_module.settings.auth_rate_limit_login_max_requests
    original_window = auth_module.settings.auth_rate_limit_window_seconds
    auth_module.settings.auth_rate_limit_login_max_requests = 2
    auth_module.settings.auth_rate_limit_window_seconds = 300
    try:
        for _ in range(2):
            response = client.post(
                "/api/auth/login",
                json={
                    "email": register_payload["email"],
                    "password": "wrong-password123",
                    "role": "admin",
                },
            )
            assert response.status_code == 401

        blocked = client.post(
            "/api/auth/login",
            json={
                "email": register_payload["email"],
                "password": "wrong-password123",
                "role": "admin",
            },
        )
        assert blocked.status_code == 429
        assert "Too many requests for auth.login" in blocked.json()["detail"]
    finally:
        auth_module.settings.auth_rate_limit_login_max_requests = original_limit
        auth_module.settings.auth_rate_limit_window_seconds = original_window


def test_query_user_by_email_retries_after_schema_bootstrap(monkeypatch):
    import app.api.routes.auth as auth_module

    expected_user = object()
    bootstrap_calls = {"count": 0}

    class DummyResult:
        def scalar_one_or_none(self):
            return expected_user

    class DummyDB:
        def __init__(self):
            self.execute_calls = 0
            self.rollback_calls = 0

        def execute(self, _statement):
            self.execute_calls += 1
            if self.execute_calls == 1:
                raise ProgrammingError(
                    "SELECT users.id FROM users",
                    {},
                    Exception("column users.section_name does not exist"),
                )
            return DummyResult()

        def rollback(self):
            self.rollback_calls += 1

    db = DummyDB()

    def fake_bootstrap():
        bootstrap_calls["count"] += 1

    monkeypatch.setattr(auth_module, "ensure_runtime_schema_compatibility", fake_bootstrap)

    result = auth_module._query_user_by_email(db, "admin@example.com")

    assert result is expected_user
    assert db.execute_calls == 2
    assert db.rollback_calls == 1
    assert bootstrap_calls["count"] == 1


def test_query_user_by_email_returns_actionable_error_on_unrecoverable_schema_drift(monkeypatch):
    import app.api.routes.auth as auth_module

    class DummyDB:
        def execute(self, _statement):
            raise ProgrammingError(
                "SELECT users.id FROM users",
                {},
                Exception("column users.section_name does not exist"),
            )

        def rollback(self):
            return None

    monkeypatch.setattr(
        auth_module,
        "ensure_runtime_schema_compatibility",
        lambda: (_ for _ in ()).throw(RuntimeError("bootstrap failed")),
    )

    with pytest.raises(HTTPException) as exc_info:
        auth_module._query_user_by_email(DummyDB(), "ops@example.com")

    assert exc_info.value.status_code == 503
    assert "alembic upgrade head" in str(exc_info.value.detail)
