from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db, require_roles
from app.core.config import get_settings
from app.models.institution_settings import InstitutionSettings
from app.models.notification import NotificationType
from app.models.user import User, UserRole
from app.schemas.settings import (
    AcademicCycleSettings,
    DEFAULT_ACADEMIC_CYCLE,
    DEFAULT_SCHEDULE_POLICY,
    DEFAULT_WORKING_HOURS,
    SchedulePolicyUpdate,
    SmtpConfigurationOut,
    SmtpTestRequest,
    SmtpTestResponse,
    WorkingHoursUpdate,
)
from app.services.email import EmailDeliveryError, send_email
from app.services.notifications import notify_all_users

router = APIRouter()
settings = get_settings()


def get_settings_record(db: Session) -> InstitutionSettings | None:
    return db.execute(select(InstitutionSettings).where(InstitutionSettings.id == 1)).scalar_one_or_none()


def build_default_settings_record() -> InstitutionSettings:
    return InstitutionSettings(
        id=1,
        working_hours=[entry.model_dump() for entry in DEFAULT_WORKING_HOURS],
        period_minutes=DEFAULT_SCHEDULE_POLICY.period_minutes,
        lab_contiguous_slots=DEFAULT_SCHEDULE_POLICY.lab_contiguous_slots,
        break_windows=[item.model_dump() for item in DEFAULT_SCHEDULE_POLICY.breaks],
        academic_year=DEFAULT_ACADEMIC_CYCLE.academic_year,
        semester_cycle=DEFAULT_ACADEMIC_CYCLE.semester_cycle,
    )


def build_schedule_policy(record: InstitutionSettings | None) -> SchedulePolicyUpdate:
    if record is None:
        return DEFAULT_SCHEDULE_POLICY

    period_minutes = record.period_minutes or DEFAULT_SCHEDULE_POLICY.period_minutes
    lab_contiguous_slots = record.lab_contiguous_slots or DEFAULT_SCHEDULE_POLICY.lab_contiguous_slots
    break_windows = record.break_windows or [item.model_dump() for item in DEFAULT_SCHEDULE_POLICY.breaks]
    return SchedulePolicyUpdate(
        period_minutes=period_minutes,
        lab_contiguous_slots=lab_contiguous_slots,
        breaks=break_windows,
    )


def build_academic_cycle(record: InstitutionSettings | None) -> AcademicCycleSettings:
    if record is None:
        return DEFAULT_ACADEMIC_CYCLE

    year = record.academic_year or DEFAULT_ACADEMIC_CYCLE.academic_year
    cycle = record.semester_cycle or DEFAULT_ACADEMIC_CYCLE.semester_cycle
    return AcademicCycleSettings(academic_year=year, semester_cycle=cycle)


@router.get("/settings/working-hours", response_model=WorkingHoursUpdate)
def get_working_hours(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> WorkingHoursUpdate:
    record = get_settings_record(db)
    if record is None:
        return WorkingHoursUpdate(hours=DEFAULT_WORKING_HOURS)
    return WorkingHoursUpdate(hours=record.working_hours)


@router.put("/settings/working-hours", response_model=WorkingHoursUpdate)
def update_working_hours(
    payload: WorkingHoursUpdate,
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> WorkingHoursUpdate:
    record = get_settings_record(db)
    hours_payload = [entry.model_dump() for entry in payload.hours]

    if record is None:
        record = build_default_settings_record()
        record.working_hours = hours_payload
        db.add(record)
    else:
        record.working_hours = hours_payload

    notify_all_users(
        db,
        title="Working Hours Updated",
        message=(
            f"Working hours were updated by {current_user.name}. "
            "Future timetable generation and validation will use the new timings."
        ),
        notification_type=NotificationType.system,
        exclude_user_id=current_user.id,
        deliver_email=True,
    )
    db.commit()
    db.refresh(record)
    return WorkingHoursUpdate(hours=record.working_hours)


@router.get("/settings/schedule-policy", response_model=SchedulePolicyUpdate)
def get_schedule_policy(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SchedulePolicyUpdate:
    record = get_settings_record(db)
    return build_schedule_policy(record)


@router.put("/settings/schedule-policy", response_model=SchedulePolicyUpdate)
def update_schedule_policy(
    payload: SchedulePolicyUpdate,
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> SchedulePolicyUpdate:
    record = get_settings_record(db)
    if record is None:
        record = build_default_settings_record()
        db.add(record)

    record.period_minutes = payload.period_minutes
    record.lab_contiguous_slots = payload.lab_contiguous_slots
    record.break_windows = [entry.model_dump() for entry in payload.breaks]

    notify_all_users(
        db,
        title="Schedule Policy Updated",
        message=(
            f"Period/break policy was updated by {current_user.name}. "
            "Regenerate timetables to apply the latest policy."
        ),
        notification_type=NotificationType.system,
        exclude_user_id=current_user.id,
        deliver_email=True,
    )
    db.commit()
    db.refresh(record)
    return build_schedule_policy(record)


@router.get("/settings/academic-cycle", response_model=AcademicCycleSettings)
def get_academic_cycle(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AcademicCycleSettings:
    record = get_settings_record(db)
    return build_academic_cycle(record)


@router.put("/settings/academic-cycle", response_model=AcademicCycleSettings)
def update_academic_cycle(
    payload: AcademicCycleSettings,
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> AcademicCycleSettings:
    record = get_settings_record(db)
    if record is None:
        record = build_default_settings_record()
        db.add(record)

    record.academic_year = payload.academic_year
    record.semester_cycle = payload.semester_cycle

    notify_all_users(
        db,
        title="Academic Cycle Updated",
        message=(
            f"Academic cycle was set to {payload.academic_year} ({payload.semester_cycle}) by {current_user.name}."
        ),
        notification_type=NotificationType.system,
        exclude_user_id=current_user.id,
        deliver_email=True,
    )
    db.commit()
    db.refresh(record)
    return build_academic_cycle(record)


@router.get("/settings/smtp/config", response_model=SmtpConfigurationOut)
def get_smtp_configuration(
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
) -> SmtpConfigurationOut:
    return SmtpConfigurationOut(
        configured=bool(settings.smtp_host and settings.smtp_from_email),
        host=settings.smtp_host,
        port=settings.smtp_port,
        username_set=bool(settings.smtp_username),
        from_email=settings.smtp_from_email,
        from_name=settings.smtp_from_name,
        use_tls=settings.smtp_use_tls,
        use_ssl=settings.smtp_use_ssl,
        backup_configured=bool(settings.smtp_backup_host and (settings.smtp_backup_from_email or settings.smtp_from_email)),
        backup_host=settings.smtp_backup_host,
        backup_port=settings.smtp_backup_port,
        notification_prefer_backup=settings.smtp_notification_prefer_backup,
        retry_attempts=settings.smtp_retry_attempts,
        retry_backoff_seconds=settings.smtp_retry_backoff_seconds,
        rate_limit_cooldown_seconds=settings.smtp_rate_limit_cooldown_seconds,
        timeout_seconds=settings.smtp_timeout_seconds,
    )


@router.post("/settings/smtp/test", response_model=SmtpTestResponse)
def send_smtp_test_email(
    payload: SmtpTestRequest,
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
) -> SmtpTestResponse:
    recipient = payload.to_email or current_user.email
    if not recipient:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A recipient email is required for SMTP test.",
        )

    subject = "ShedForge SMTP Test"
    timestamp = datetime.now(timezone.utc).isoformat()
    message = (
        "This is a SMTP test email from ShedForge.\n\n"
        f"Triggered by: {current_user.email}\n"
        f"Timestamp (UTC): {timestamp}\n"
    )
    try:
        send_email(
            to_email=recipient,
            subject=subject,
            text_content=message,
        )
    except EmailDeliveryError as exc:
        detail = "Unable to send test email. Verify SMTP settings."
        if str(exc) == "SMTP is not configured":
            detail = "SMTP is not configured. Set SMTP_HOST and SMTP_FROM_EMAIL in backend/.env."
        elif str(exc) == "SMTP authentication failed":
            detail = "SMTP authentication failed. Verify username/password or app password."
        elif str(exc) == "SMTP connection failed":
            detail = "SMTP connection failed. Verify host, port, TLS/SSL, and firewall."
        elif str(exc) == "SMTP sender rate limited":
            detail = "SMTP sender is rate-limited by provider. Wait and retry, or switch SMTP account."
        elif str(exc) == "SMTP recipient rejected":
            detail = "SMTP recipient was rejected by provider."
        elif str(exc) == "SMTP sender rejected":
            detail = "SMTP sender was rejected. Verify SMTP_FROM_EMAIL."
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail) from exc

    return SmtpTestResponse(
        success=True,
        message="SMTP test email sent successfully.",
        recipient=recipient,
    )
