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
    settings = get_settings()
    if not settings.email_enabled:
        logger.info("Email delivery disabled. Skipping outbound email to %s with subject '%s'.", to_email, subject)
        return

    # PRIORitize SendGrid SDK if API key is present
    if settings.sendgrid_api_key and SendGridAPIClient:
        try:
            sg = SendGridAPIClient(api_key=settings.sendgrid_api_key)
            from_email = settings.smtp_from_email or "notifications@shedforge.com"
            from_name = settings.smtp_from_name or "ShedForge"

            message = Mail(
                from_email=(from_email, from_name),
                to_emails=to_email,
                subject=subject,
                plain_text_content=text_content,
                html_content=html_content,
            )
            response = sg.send(message)
            if 200 <= response.status_code < 300:
                return
        except Exception:
            # Fallback to SMTP if SendGrid fails
            pass

    delivery_profile = _infer_delivery_profile(subject)
    endpoints = _resolve_endpoints(settings, delivery_profile=delivery_profile)
    timeout = max(1, settings.smtp_timeout_seconds)
    retry_attempts = max(1, settings.smtp_retry_attempts)
    retry_backoff_seconds = max(0.0, settings.smtp_retry_backoff_seconds)
    cooldown_seconds = max(0, settings.smtp_rate_limit_cooldown_seconds)

    last_error: Exception | None = None
    last_error_message = "Unable to deliver email"

    # Use certifi for SSL context to fix macOS certificate issues
    ssl_context = ssl.create_default_context(cafile=certifi.where())

    for endpoint in endpoints:
        endpoint_fatal = False
        for port, use_tls, use_ssl in _transport_attempts(endpoint):
            for attempt in range(1, retry_attempts + 1):
                try:
                    message = _build_message(
                        from_email=endpoint.from_email,
                        from_name=endpoint.from_name,
                        to_email=to_email,
                        subject=subject,
                        text_content=text_content,
                        html_content=html_content,
                    )
                    if use_ssl:
                        with smtplib.SMTP_SSL(endpoint.host, port, timeout=timeout, context=ssl_context) as smtp:
                            if endpoint.username:
                                smtp.login(endpoint.username, endpoint.password)
                            smtp.send_message(message)
                        return

                    with smtplib.SMTP(endpoint.host, port, timeout=timeout) as smtp:
                        if use_tls:
                            smtp.starttls(context=ssl_context)
                        if endpoint.username:
                            smtp.login(endpoint.username, endpoint.password)
                        smtp.send_message(message)
                    return
                except smtplib.SMTPAuthenticationError as exc:  # pragma: no cover
                    last_error = exc
                    last_error_message = "SMTP authentication failed"
                    endpoint_fatal = True
                    break
                except smtplib.SMTPDataError as exc:  # pragma: no cover
                    last_error = exc
                    last_error_message = _classify_smtp_data_error(exc)
                    if last_error_message == "SMTP sender rate limited":
                        _mark_endpoint_rate_limited(endpoint, cooldown_seconds)
                    endpoint_fatal = True
                    break
                except smtplib.SMTPRecipientsRefused as exc:  # pragma: no cover
                    last_error = exc
                    last_error_message = "SMTP recipient rejected"
                    endpoint_fatal = True
                    break
                except smtplib.SMTPSenderRefused as exc:  # pragma: no cover
                    last_error = exc
                    last_error_message = "SMTP sender rejected"
                    endpoint_fatal = True
                    break
                except Exception as exc:  # pragma: no cover
                    last_error = exc
                    if _is_connection_issue(exc):
                        last_error_message = "SMTP connection failed"
                        if attempt < retry_attempts:
                            if retry_backoff_seconds > 0:
                                time.sleep(retry_backoff_seconds * attempt)
                            continue
                        break
                    last_error_message = "Unable to deliver email"
                    break
            if endpoint_fatal:
                break

    raise EmailDeliveryError(last_error_message) from last_error
