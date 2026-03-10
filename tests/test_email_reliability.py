from __future__ import annotations

import smtplib
import time
from types import SimpleNamespace

import pytest

from app.services.email import EmailDeliveryError
from app.services import email as email_service


def _settings(**overrides):
    defaults = dict(
        smtp_host="smtp.primary.test",
        smtp_port=587,
        smtp_username="primary-user",
        smtp_password="primary-pass",
        smtp_from_email="primary@example.com",
        smtp_from_name="Primary Sender",
        smtp_use_tls=True,
        smtp_use_ssl=False,
        smtp_backup_host=None,
        smtp_backup_port=587,
        smtp_backup_username=None,
        smtp_backup_password=None,
        smtp_backup_from_email=None,
        smtp_backup_from_name="Backup Sender",
        smtp_backup_use_tls=True,
        smtp_backup_use_ssl=False,
        smtp_notification_prefer_backup=False,
        smtp_retry_attempts=2,
        smtp_retry_backoff_seconds=0.0,
        smtp_rate_limit_cooldown_seconds=600,
        smtp_timeout_seconds=5,
        sendgrid_api_key=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.fixture(autouse=True)
def _clear_smtp_cooldowns():
    email_service._SMTP_ENDPOINT_COOLDOWN_UNTIL.clear()
    yield
    email_service._SMTP_ENDPOINT_COOLDOWN_UNTIL.clear()


def test_send_email_uses_backup_after_primary_rate_limit(monkeypatch):
    settings = _settings(
        smtp_backup_host="smtp.backup.test",
        smtp_backup_username="backup-user",
        smtp_backup_password="backup-pass",
        smtp_backup_from_email="backup@example.com",
    )
    monkeypatch.setattr(email_service, "get_settings", lambda: settings)

    sends: list[tuple[str, int, str]] = []

    class FakeSMTP:
        def __init__(self, host, port, timeout=None):
            self.host = host
            self.port = port

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def starttls(self, context=None):
            return None

        def login(self, username, password):
            return None

        def send_message(self, message):
            sends.append((self.host, self.port, message["From"]))
            if self.host == "smtp.primary.test":
                raise smtplib.SMTPDataError(550, b"Daily user sending limit exceeded")
            return {}

    monkeypatch.setattr(email_service.smtplib, "SMTP", FakeSMTP)
    monkeypatch.setattr(email_service.smtplib, "SMTP_SSL", FakeSMTP)

    email_service.send_email(
        to_email="recipient@example.com",
        subject="Your ShedForge Login Verification Code",
        text_content="OTP body",
    )

    assert sends[0][0] == "smtp.primary.test"
    assert sends[1][0] == "smtp.backup.test"
    assert "backup@example.com" in sends[1][2]

    primary_key = "smtp.primary.test:587:primary-user:primary@example.com"
    assert email_service._SMTP_ENDPOINT_COOLDOWN_UNTIL[primary_key] > time.time()


def test_send_email_retries_connection_drop_and_succeeds(monkeypatch):
    settings = _settings(smtp_retry_attempts=2, smtp_retry_backoff_seconds=0.0)
    monkeypatch.setattr(email_service, "get_settings", lambda: settings)

    attempt_counter = {"count": 0}

    class FlakySMTP:
        def __init__(self, host, port, timeout=None):
            self.host = host
            self.port = port

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def starttls(self, context=None):
            return None

        def login(self, username, password):
            return None

        def send_message(self, message):
            attempt_counter["count"] += 1
            if attempt_counter["count"] == 1:
                raise smtplib.SMTPServerDisconnected("network drop")
            return {}

    monkeypatch.setattr(email_service.smtplib, "SMTP", FlakySMTP)
    monkeypatch.setattr(email_service.smtplib, "SMTP_SSL", FlakySMTP)

    email_service.send_email(
        to_email="recipient@example.com",
        subject="Critical system alert",
        text_content="hello",
    )

    assert attempt_counter["count"] == 2


def test_notification_email_prefers_backup_when_enabled(monkeypatch):
    settings = _settings(
        smtp_backup_host="smtp.backup.test",
        smtp_backup_username="backup-user",
        smtp_backup_password="backup-pass",
        smtp_backup_from_email="backup@example.com",
        smtp_notification_prefer_backup=True,
    )
    monkeypatch.setattr(email_service, "get_settings", lambda: settings)

    contacted_hosts: list[str] = []

    class FakeSMTP:
        def __init__(self, host, port, timeout=None):
            self.host = host
            self.port = port

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def starttls(self, context=None):
            return None

        def login(self, username, password):
            return None

        def send_message(self, message):
            contacted_hosts.append(self.host)
            return {}

    monkeypatch.setattr(email_service.smtplib, "SMTP", FakeSMTP)
    monkeypatch.setattr(email_service.smtplib, "SMTP_SSL", FakeSMTP)

    email_service.send_email(
        to_email="recipient@example.com",
        subject="ShedForge Notification: Timetable Updated",
        text_content="updated",
    )

    assert contacted_hosts == ["smtp.backup.test"]


def test_send_email_raises_rate_limit_error_when_all_endpoints_exhausted(monkeypatch):
    settings = _settings(
        smtp_backup_host="smtp.backup.test",
        smtp_backup_username="backup-user",
        smtp_backup_password="backup-pass",
        smtp_backup_from_email="backup@example.com",
    )
    monkeypatch.setattr(email_service, "get_settings", lambda: settings)

    class FakeSMTP:
        def __init__(self, host, port, timeout=None):
            self.host = host
            self.port = port

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def starttls(self, context=None):
            return None

        def login(self, username, password):
            return None

        def send_message(self, message):
            raise smtplib.SMTPDataError(550, b"sending limit exceeded")

    monkeypatch.setattr(email_service.smtplib, "SMTP", FakeSMTP)
    monkeypatch.setattr(email_service.smtplib, "SMTP_SSL", FakeSMTP)

    with pytest.raises(EmailDeliveryError) as exc_info:
        email_service.send_email(
            to_email="recipient@example.com",
            subject="Your ShedForge Login Verification Code",
            text_content="OTP body",
        )

    assert str(exc_info.value) == "SMTP sender rate limited"
