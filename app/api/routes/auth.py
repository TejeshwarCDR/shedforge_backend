from datetime import datetime, timedelta
import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db
from app.core.config import get_settings
from app.core.security import create_access_token, get_password_hash, verify_password
from app.models.faculty import Faculty
from app.models.program import Program
from app.models.user import User, UserRole
from app.schemas.user import (
    Token,
    UserCreate,
    UserLogin,
    UserOut,
)
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
