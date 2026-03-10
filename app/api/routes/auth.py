from datetime import datetime, timedelta, timezone
import hashlib
import logging
import secrets

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, ProgrammingError
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db
from app.core.config import get_settings
from app.core.security import create_access_token, get_password_hash, verify_password
from app.db.bootstrap import ensure_runtime_schema_compatibility
from app.models.faculty import Faculty
from app.models.login_otp import LoginOtpChallenge
from app.models.password_reset import PasswordResetToken
from app.models.program import Program
from app.models.user import User, UserRole
from app.schemas.password import PasswordChange, PasswordResetConfirm, PasswordResetRequest
from app.schemas.user import (
    LoginOtpChallengeOut,
    LoginOtpRequest,
    LoginOtpVerify,
    Token,
    UserCreate,
    UserLogin,
    UserOut,
)
from app.services.email import EmailDeliveryError, send_email
from app.services.rate_limit import enforce_rate_limit
from app.services.workload import constrained_max_hours

settings = get_settings()
router = APIRouter()
logger = logging.getLogger(__name__)
DEFAULT_FACULTY_AVAILABILITY = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def _resolve_program_id_for_faculty(db: Session, requested_program_id: str | None) -> str:
    if requested_program_id:
        return requested_program_id
    fallback_program_id = db.execute(select(Program.id).order_by(Program.created_at.asc())).scalar_one_or_none()
    if fallback_program_id is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No program available. Create a program before creating faculty accounts.",
        )
    return str(fallback_program_id)


def ensure_faculty_profile(
    db: Session,
    *,
    name: str,
    email: str,
    program_id: str | None,
    department: str | None,
    preferred_subject_codes: list[str] | None = None,
) -> bool:
    resolved_program_id = _resolve_program_id_for_faculty(db, program_id)
    default_designation = "Assistant Professor"
    default_max_hours = constrained_max_hours(default_designation, None)
    faculty = db.execute(select(Faculty).where(Faculty.email == email)).scalar_one_or_none()
    if faculty is None:
        db.add(
            Faculty(
                name=name,
                program_id=resolved_program_id,
                designation=default_designation,
                email=email,
                department=department or "General",
                workload_hours=0,
                max_hours=default_max_hours,
                availability=DEFAULT_FACULTY_AVAILABILITY,
                availability_windows=[],
                avoid_back_to_back=False,
                preferred_min_break_minutes=0,
                preference_notes=None,
                preferred_subject_codes=preferred_subject_codes or [],
                semester_preferences={},
            )
        )
        return True

    updated = False
    if faculty.program_id != resolved_program_id:
        faculty.program_id = resolved_program_id
        updated = True
    if not faculty.availability:
        faculty.availability = DEFAULT_FACULTY_AVAILABILITY
        updated = True
    if faculty.department is None or not faculty.department.strip():
        faculty.department = department or "General"
        updated = True
    if faculty.name is None or not faculty.name.strip():
        faculty.name = name
        updated = True
    if faculty.availability_windows is None:
        faculty.availability_windows = []
        updated = True
    if faculty.preferred_subject_codes is None:
        faculty.preferred_subject_codes = []
        updated = True
    if faculty.semester_preferences is None:
        faculty.semester_preferences = {}
        updated = True
    constrained_hours = constrained_max_hours(faculty.designation, faculty.max_hours)
    if faculty.max_hours != constrained_hours:
        faculty.max_hours = constrained_hours
        updated = True
    if preferred_subject_codes:
        normalized = list(dict.fromkeys(code.strip().upper() for code in preferred_subject_codes if code.strip()))
        if normalized and faculty.preferred_subject_codes != normalized:
            faculty.preferred_subject_codes = normalized
            updated = True
    return updated


def _query_user_by_email(db: Session, email: str) -> User | None:
    statement = select(User).where(User.email == email)
    try:
        return db.execute(statement).scalar_one_or_none()
    except ProgrammingError:
        # Auto-heal additive schema drift for long-lived developer databases.
        db.rollback()
        try:
            ensure_runtime_schema_compatibility()
            return db.execute(statement).scalar_one_or_none()
        except Exception as bootstrap_exc:
            logger.exception("Database schema compatibility check failed during auth lookup")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database schema is outdated. Run `alembic upgrade head` and restart backend.",
            ) from bootstrap_exc


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def register(payload: UserCreate, request: Request, db: Session = Depends(get_db)) -> UserOut:
    enforce_rate_limit(
        request=request,
        scope="auth.register",
        limit=settings.auth_rate_limit_register_max_requests,
        window_seconds=settings.auth_rate_limit_window_seconds,
        identity=payload.email,
    )
    existing = _query_user_by_email(db, payload.email)
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")

    resolved_program_id = payload.program_id
    if payload.role in {UserRole.faculty, UserRole.student}:
        resolved_program_id = _resolve_program_id_for_faculty(db, payload.program_id)

    user = User(
        name=payload.name,
        email=payload.email,
        hashed_password=get_password_hash(payload.password),
        role=payload.role,
        program_id=resolved_program_id,
        department=payload.department,
        section_name=payload.section_name,
        semester_number=payload.semester_number,
        batch_year=payload.batch_year,
        roll_number=payload.roll_number,
    )
    db.add(user)

    if payload.role == UserRole.faculty:
        ensure_faculty_profile(
            db,
            name=payload.name,
            email=payload.email,
            program_id=resolved_program_id,
            department=payload.department,
            preferred_subject_codes=payload.preferred_subject_codes,
        )

    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered") from exc

    db.refresh(user)
    return user


def validate_login_user(payload: UserLogin, db: Session) -> User:
    user = _query_user_by_email(db, payload.email)
    if user is None or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User account is inactive")
    if payload.role and payload.role != user.role:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Role does not match user account")
    return user


@router.post("/login", response_model=Token)
def login(payload: UserLogin, request: Request, db: Session = Depends(get_db)) -> Token:
    enforce_rate_limit(
        request=request,
        scope="auth.login",
        limit=settings.auth_rate_limit_login_max_requests,
        window_seconds=settings.auth_rate_limit_window_seconds,
        identity=payload.email,
    )
    user = validate_login_user(payload, db)

    if user.role == UserRole.faculty and ensure_faculty_profile(
        db,
        name=user.name,
        email=user.email,
        program_id=user.program_id,
        department=user.department,
    ):
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            if db.execute(select(Faculty).where(Faculty.email == user.email)).scalar_one_or_none() is None:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Unable to ensure faculty profile for this account.",
                )

    access_token = create_access_token(user.id, expires_delta=timedelta(minutes=settings.access_token_expire_minutes))
    return Token(access_token=access_token, token_type="bearer", user=user)


@router.get("/me", response_model=UserOut)
def me(current_user: User = Depends(get_current_user)) -> UserOut:
    return current_user


@router.post("/logout")
def logout(current_user: User = Depends(get_current_user)) -> dict:
    return {"success": True}


def hash_login_otp(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_dt(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


@router.post("/login/request-otp", response_model=LoginOtpChallengeOut)
def request_login_otp(payload: LoginOtpRequest, request: Request, db: Session = Depends(get_db)) -> LoginOtpChallengeOut:
    if not settings.login_otp_enabled:
        user = validate_login_user(payload, db)
        return LoginOtpChallengeOut(
            challenge_id="otp-disabled",
            email=user.email,
            expires_in_seconds=0,
            message="OTP verification is disabled. Use direct login with email and password.",
            otp_hint=None,
        )

    enforce_rate_limit(
        request=request,
        scope="auth.login.request_otp",
        limit=settings.auth_rate_limit_otp_request_max_requests,
        window_seconds=settings.auth_rate_limit_window_seconds,
        identity=payload.email,
    )
    user = validate_login_user(payload, db)
    if user.role == UserRole.faculty and ensure_faculty_profile(
        db,
        name=user.name,
        email=user.email,
        program_id=user.program_id,
        department=user.department,
    ):
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            if db.execute(select(Faculty).where(Faculty.email == user.email)).scalar_one_or_none() is None:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Unable to ensure faculty profile for this account.",
                )

    now = datetime.now(timezone.utc)

    existing = db.execute(
        select(LoginOtpChallenge).where(
            LoginOtpChallenge.user_id == user.id,
            LoginOtpChallenge.used_at.is_(None),
        )
    ).scalars()
    for row in existing:
        row.used_at = now

    otp_code = f"{secrets.randbelow(1_000_000):06d}"
    challenge = LoginOtpChallenge(
        user_id=user.id,
        code_hash=hash_login_otp(otp_code),
        expires_at=now + timedelta(minutes=settings.login_otp_expire_minutes),
        max_attempts=settings.login_otp_max_attempts,
    )
    db.add(challenge)
    db.flush()
    if settings.login_otp_log_to_terminal:
        logger.warning(
            "LOGIN OTP | email=%s | challenge_id=%s | otp=%s | expires_in_min=%s",
            user.email,
            challenge.id,
            otp_code,
            settings.login_otp_expire_minutes,
        )

    subject = "Your ShedForge Login Verification Code"
    message = (
        f"Hello {user.name},\n\n"
        f"Your ShedForge verification code is: {otp_code}\n"
        f"This code will expire in {settings.login_otp_expire_minutes} minutes.\n\n"
        "If you did not request this, please ignore this email."
    )
    try:
        send_email(
            to_email=user.email,
            subject=subject,
            text_content=message,
        )
    except EmailDeliveryError as exc:
        logger.exception("Failed to send login OTP email for %s", user.email)
        if settings.login_otp_log_to_terminal and settings.login_otp_allow_terminal_fallback:
            db.commit()
            hint = otp_code if settings.expose_login_otp else None
            return LoginOtpChallengeOut(
                challenge_id=challenge.id,
                email=user.email,
                expires_in_seconds=settings.login_otp_expire_minutes * 60,
                message="Verification code generated. Email delivery failed, check backend terminal log for OTP.",
                otp_hint=hint,
            )
        db.rollback()
        detail = "Unable to send verification email. Please try again later."
        if str(exc) == "SMTP is not configured":
            detail = "Email service not configured. Set SMTP settings in backend/.env and restart backend."
        elif str(exc) == "SMTP authentication failed":
            detail = "Email authentication failed. Verify SMTP username/password (or app password) and try again."
        elif str(exc) == "SMTP connection failed":
            detail = "Cannot connect to SMTP server. Verify SMTP host/port and TLS/SSL settings."
        elif str(exc) == "SMTP sender rate limited":
            detail = "Email sending limit reached for the configured SMTP account. Try again later or use another SMTP account."
        elif str(exc) == "SMTP recipient rejected":
            detail = "Recipient email was rejected by the SMTP provider."
        elif str(exc) == "SMTP sender rejected":
            detail = "Sender email was rejected by the SMTP provider. Verify SMTP_FROM_EMAIL."
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
        ) from exc

    db.commit()

    hint = otp_code if settings.expose_login_otp else None
    return LoginOtpChallengeOut(
        challenge_id=challenge.id,
        email=user.email,
        expires_in_seconds=settings.login_otp_expire_minutes * 60,
        message="Verification code sent to your email.",
        otp_hint=hint,
    )


@router.post("/login/verify-otp", response_model=Token)
def verify_login_otp(payload: LoginOtpVerify, request: Request, db: Session = Depends(get_db)) -> Token:
    if not settings.login_otp_enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="OTP verification is disabled. Use direct login with email and password.",
        )

    enforce_rate_limit(
        request=request,
        scope="auth.login.verify_otp",
        limit=settings.auth_rate_limit_otp_verify_max_requests,
        window_seconds=settings.auth_rate_limit_window_seconds,
        identity=payload.challenge_id,
    )
    challenge = db.get(LoginOtpChallenge, payload.challenge_id)
    if challenge is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid verification challenge")

    now = datetime.now(timezone.utc)
    expires_at = normalize_dt(challenge.expires_at)
    if challenge.used_at is not None or expires_at is None or expires_at < now:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Verification code has expired")
    if challenge.attempt_count >= challenge.max_attempts:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Maximum verification attempts exceeded")

    challenge.attempt_count += 1
    expected_hash = hash_login_otp(payload.otp_code)
    if not secrets.compare_digest(expected_hash, challenge.code_hash):
        db.commit()
        remaining = max(0, challenge.max_attempts - challenge.attempt_count)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid verification code. {remaining} attempt(s) remaining.",
        )

    user = db.get(User, challenge.user_id)
    if user is None or not user.is_active:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid verification challenge")

    challenge.used_at = now
    access_token = create_access_token(user.id, expires_delta=timedelta(minutes=settings.access_token_expire_minutes))
    db.commit()
    return Token(access_token=access_token, token_type="bearer", user=user)


def hash_reset_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


@router.post("/password/forgot")
def request_password_reset(payload: PasswordResetRequest, request: Request, db: Session = Depends(get_db)) -> dict:
    if not settings.password_reset_enabled:
        return {"message": "Password reset email flow is disabled in this environment."}

    enforce_rate_limit(
        request=request,
        scope="auth.password.forgot",
        limit=settings.auth_rate_limit_password_reset_max_requests,
        window_seconds=settings.auth_rate_limit_window_seconds,
        identity=payload.email,
    )
    user = _query_user_by_email(db, payload.email)
    response = {"message": "If an account exists, a reset token has been sent."}

    if user and user.is_active:
        now = datetime.now(timezone.utc)
        token = secrets.token_urlsafe(32)
        token_hash = hash_reset_token(token)
        expires_at = now + timedelta(minutes=settings.reset_token_expire_minutes)

        # Invalidate any existing unused tokens for this user
        existing_tokens = db.execute(
            select(PasswordResetToken).where(
                PasswordResetToken.user_id == user.id,
                PasswordResetToken.used_at.is_(None),
            )
        ).scalars()
        for existing in existing_tokens:
            existing.used_at = now

        reset_token = PasswordResetToken(
            user_id=user.id,
            token_hash=token_hash,
            expires_at=expires_at,
        )
        db.add(reset_token)
        db.commit()

        if settings.expose_reset_token:
            response["reset_token"] = token

    return response


@router.post("/password/reset")
def confirm_password_reset(payload: PasswordResetConfirm, request: Request, db: Session = Depends(get_db)) -> dict:
    if not settings.password_reset_enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password reset flow is disabled in this environment.",
        )

    enforce_rate_limit(
        request=request,
        scope="auth.password.reset",
        limit=settings.auth_rate_limit_password_reset_max_requests,
        window_seconds=settings.auth_rate_limit_window_seconds,
        identity=payload.token[:24],
    )
    token_hash = hash_reset_token(payload.token)
    now = datetime.now(timezone.utc)

    token_entry = db.execute(
        select(PasswordResetToken).where(PasswordResetToken.token_hash == token_hash)
    ).scalar_one_or_none()

    expires_at = token_entry.expires_at if token_entry is not None else None
    if expires_at is not None and expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)

    if token_entry is None or token_entry.used_at is not None or expires_at is None or expires_at < now:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid or expired reset token")

    user = db.get(User, token_entry.user_id)
    if user is None or not user.is_active:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid reset request")

    user.hashed_password = get_password_hash(payload.new_password)
    token_entry.used_at = now
    db.commit()
    return {"success": True}


@router.post("/password/change")
def change_password(
    payload: PasswordChange,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    if not verify_password(payload.current_password, current_user.hashed_password):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Current password is incorrect")

    if verify_password(payload.new_password, current_user.hashed_password):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="New password must be different")

    current_user.hashed_password = get_password_hash(payload.new_password)
    db.commit()
    return {"success": True}
