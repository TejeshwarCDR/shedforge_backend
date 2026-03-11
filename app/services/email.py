from __future__ import annotations

from dataclasses import dataclass
from email.message import EmailMessage
import certifi
import smtplib
import ssl
import time

from app.core.config import get_settings


class EmailDeliveryError(RuntimeError):
    pass


@dataclass(frozen=True)
class _SmtpEndpoint:
    host: str
    port: int
    username: str | None
    password: str
    from_email: str
    from_name: str | None
    use_tls: bool
    use_ssl: bool


# Avoid repeatedly hitting a provider that just rate-limited us.
_SMTP_ENDPOINT_COOLDOWN_UNTIL: dict[str, float] = {}


def _classify_smtp_data_error(exc: smtplib.SMTPDataError) -> str:
    smtp_error = exc.smtp_error
    if isinstance(smtp_error, bytes):
        message = smtp_error.decode("utf-8", errors="ignore").lower()
    else:
        message = str(smtp_error).lower()

    if "daily user sending limit exceeded" in message or "sending limit" in message or "quota" in message:
        return "SMTP sender rate limited"
    if "too many messages" in message or "rate limit" in message:
        return "SMTP sender rate limited"
    if "recipient address rejected" in message or "recipient rejected" in message:
        return "SMTP recipient rejected"
    if "sender address rejected" in message or "sender rejected" in message:
        return "SMTP sender rejected"
    return "SMTP data rejected"


def _build_from_header(from_email: str, from_name: str | None) -> str:
    if from_name:
        return f"{from_name} <{from_email}>"
    return from_email


def _resolve_smtp_password(host: str | None, raw_password: str | None) -> str:
    password = raw_password or ""
    if host and host.lower() == "smtp.gmail.com":
        # Gmail app-passwords are often copied with spaces; normalize transparently.
        return "".join(password.split())
    return password


def _build_message(
    *,
    from_email: str,
    from_name: str | None,
    to_email: str,
    subject: str,
    text_content: str,
    html_content: str | None,
) -> EmailMessage:
    message = EmailMessage()
    message["From"] = _build_from_header(from_email, from_name)
    message["To"] = to_email
    message["Subject"] = subject
    message.set_content(text_content)
    if html_content:
        message.add_alternative(html_content, subtype="html")
    return message


def _build_primary_endpoint(settings) -> _SmtpEndpoint | None:
    if not settings.smtp_host or not settings.smtp_from_email:
        return None
    return _SmtpEndpoint(
        host=settings.smtp_host,
        port=settings.smtp_port,
        username=settings.smtp_username,
        password=_resolve_smtp_password(settings.smtp_host, settings.smtp_password),
        from_email=settings.smtp_from_email,
        from_name=settings.smtp_from_name,
        use_tls=settings.smtp_use_tls,
        use_ssl=settings.smtp_use_ssl,
    )


def _build_backup_endpoint(settings) -> _SmtpEndpoint | None:
    if not settings.smtp_backup_host:
        return None
    from_email = settings.smtp_backup_from_email or settings.smtp_from_email
    if not from_email:
        return None
    return _SmtpEndpoint(
        host=settings.smtp_backup_host,
        port=settings.smtp_backup_port,
        username=settings.smtp_backup_username,
        password=_resolve_smtp_password(settings.smtp_backup_host, settings.smtp_backup_password),
        from_email=from_email,
        from_name=settings.smtp_backup_from_name or settings.smtp_from_name,
        use_tls=settings.smtp_backup_use_tls,
        use_ssl=settings.smtp_backup_use_ssl,
    )


def _endpoint_key(endpoint: _SmtpEndpoint) -> str:
    return f"{endpoint.host}:{endpoint.port}:{endpoint.username or ''}:{endpoint.from_email}"


def _infer_delivery_profile(subject: str) -> str:
    lowered = subject.lower()
    if "verification code" in lowered or "otp" in lowered:
        return "critical"
    if "password reset" in lowered:
        return "critical"
    if lowered.startswith("shedforge notification"):
        return "notification"
    return "standard"


def _resolve_endpoints(settings, *, delivery_profile: str) -> list[_SmtpEndpoint]:
    primary = _build_primary_endpoint(settings)
    backup = _build_backup_endpoint(settings)

    if delivery_profile == "notification" and settings.smtp_notification_prefer_backup:
        candidates = [backup, primary]
    else:
        candidates = [primary, backup]

    endpoints: list[_SmtpEndpoint] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate is None:
            continue
        key = _endpoint_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        endpoints.append(candidate)

    if not endpoints:
        raise EmailDeliveryError("SMTP is not configured")

    now = time.time()
    active = [
        endpoint
        for endpoint in endpoints
        if _SMTP_ENDPOINT_COOLDOWN_UNTIL.get(_endpoint_key(endpoint), 0.0) <= now
    ]
    if active:
        return active

    # Relaxed: If all are in cooldown, try them anyway as requested by user.
    return endpoints


def _transport_attempts(endpoint: _SmtpEndpoint) -> list[tuple[int, bool, bool]]:
    attempts: list[tuple[int, bool, bool]] = [(endpoint.port, endpoint.use_tls, endpoint.use_ssl)]
    if endpoint.host.lower() == "smtp.gmail.com":
        fallback = (587, True, False) if endpoint.use_ssl else (465, False, True)
        if fallback not in attempts:
            attempts.append(fallback)
    return attempts


def _mark_endpoint_rate_limited(endpoint: _SmtpEndpoint, cooldown_seconds: int) -> None:
    if cooldown_seconds <= 0:
        return
    _SMTP_ENDPOINT_COOLDOWN_UNTIL[_endpoint_key(endpoint)] = time.time() + cooldown_seconds


def _is_connection_issue(exc: Exception) -> bool:
    return isinstance(
        exc,
        (
            smtplib.SMTPConnectError,
            smtplib.SMTPServerDisconnected,
            smtplib.SMTPHeloError,
            OSError,
            TimeoutError,
        ),
    )


try:
    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import Mail
except ImportError:
    SendGridAPIClient = None
    Mail = None


import logging
logger = logging.getLogger(__name__)

def send_email(*, to_email: str, subject: str, text_content: str, html_content: str | None = None) -> None:
    logger.info("DEMO MODE: Email delivery skipped for %s with subject '%s'.", to_email, subject)

