from datetime import date
import re

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db, require_roles
from app.models.course import Course
from app.models.faculty import Faculty
from app.models.program import Program
from app.models.timetable import OfficialTimetable
from app.models.user import User, UserRole
from app.schemas.faculty import FacultyCreate, FacultyOut, FacultyUpdate
from app.schemas.timetable import OfficialTimetablePayload
from app.services.workload import constrained_max_hours

router = APIRouter()
SELF_EDITABLE_FACULTY_FIELDS = {
    "availability",
    "availability_windows",
    "max_hours",
    "avoid_back_to_back",
    "preferred_min_break_minutes",
    "preference_notes",
    "preferred_subject_codes",
    "semester_preferences",
}


def _normalized_identity(value: str | None) -> str:
    text = (value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _resolve_faculty_for_user(db: Session, current_user: User) -> Faculty | None:
    email = (current_user.email or "").strip().lower()
    if email:
        by_email = db.execute(select(Faculty).where(func.lower(Faculty.email) == email)).scalar_one_or_none()
        if by_email is not None:
            return by_email

    normalized_user_name = _normalized_identity(current_user.name)
    if not normalized_user_name:
        return None

    candidates = list(db.execute(select(Faculty)).scalars())
    name_matches: list[Faculty] = [
        item for item in candidates if _normalized_identity(item.name) == normalized_user_name
    ]
    if not name_matches:
        return None

    if current_user.program_id:
        scoped = [item for item in name_matches if item.program_id == current_user.program_id]
        if len(scoped) == 1:
            return scoped[0]
        if scoped:
            name_matches = scoped

    if len(name_matches) == 1:
        return name_matches[0]
    return None


def _resolve_program_id(db: Session, requested_program_id: str | None) -> str:
    if requested_program_id:
        program = db.get(Program, requested_program_id)
        if program is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Program not found")
        return program.id

    default_program = db.execute(select(Program).order_by(Program.created_at.asc())).scalars().first()
    if default_program is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No program available. Create a program before managing faculty.",
        )
    return default_program.id


@router.get("/", response_model=list[FacultyOut])
def list_faculty(
    program_id: str | None = Query(default=None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[FacultyOut]:
    if current_user.role in {UserRole.admin, UserRole.scheduler}:
        statement = select(Faculty)
        if program_id:
            statement = statement.where(Faculty.program_id == program_id)
        return list(db.execute(statement).scalars())
    if current_user.role == UserRole.faculty:
        item = _resolve_faculty_for_user(db, current_user)
        if item is None:
            return []
        if program_id and item.program_id != program_id:
            return []
        return [item] if item is not None else []
    return []


@router.get("/me", response_model=FacultyOut)
def get_my_faculty_profile(
    current_user: User = Depends(require_roles(UserRole.faculty)),
    db: Session = Depends(get_db),
) -> FacultyOut:
    item = _resolve_faculty_for_user(db, current_user)
    if item is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Faculty profile not linked")
    return item


@router.post("/", response_model=FacultyOut, status_code=status.HTTP_201_CREATED)
def create_faculty(
    payload: FacultyCreate,
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> FacultyOut:
    existing = db.execute(select(Faculty).where(Faculty.email == payload.email)).scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Faculty email already exists")
    values = payload.model_dump()
    values["program_id"] = _resolve_program_id(db, values.get("program_id"))
    values["max_hours"] = constrained_max_hours(values.get("designation"), values.get("max_hours"))
    faculty = Faculty(**values)
    db.add(faculty)
    db.commit()
    db.refresh(faculty)
    return faculty


@router.put("/{faculty_id}", response_model=FacultyOut)
def update_faculty(
    faculty_id: str,
    payload: FacultyUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> FacultyOut:
    faculty = db.get(Faculty, faculty_id)
    if faculty is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Faculty member not found")

    data = payload.model_dump(exclude_unset=True)

    if current_user.role in {UserRole.admin, UserRole.scheduler}:
        pass
    elif current_user.role == UserRole.faculty and current_user.email.lower() == faculty.email.lower():
        disallowed = sorted(set(data.keys()) - SELF_EDITABLE_FACULTY_FIELDS)
        if disallowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Faculty can only update preference fields: {', '.join(sorted(SELF_EDITABLE_FACULTY_FIELDS))}",
            )
    else:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")

    if "email" in data:
        existing = db.execute(select(Faculty).where(Faculty.email == data["email"], Faculty.id != faculty_id)).scalar_one_or_none()
        if existing:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Faculty email already exists")
    if "program_id" in data:
        data["program_id"] = _resolve_program_id(db, data["program_id"])

    updated_designation = data.get("designation", faculty.designation)
    if "designation" in data or "max_hours" in data:
        data["max_hours"] = constrained_max_hours(updated_designation, data.get("max_hours", faculty.max_hours))

    for key, value in data.items():
        setattr(faculty, key, value)
    db.commit()
    db.refresh(faculty)
    return faculty


@router.delete("/{faculty_id}")
def delete_faculty(
    faculty_id: str,
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> dict:
    faculty = db.get(Faculty, faculty_id)
    if faculty is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Faculty member not found")
    assigned_courses = list(
        db.execute(select(Course).where(Course.faculty_id == faculty_id)).scalars()
    )
    for course in assigned_courses:
        course.faculty_id = None
    db.delete(faculty)
    db.commit()
    return {"success": True, "unassigned_course_count": len(assigned_courses)}


@router.get("/substitutes/suggestions")
def substitute_suggestions(
    leave_date: date = Query(..., description="Date requiring a substitute"),
    course_id: str | None = Query(default=None),
    program_id: str | None = Query(default=None),
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> list[dict]:
    day_name = leave_date.strftime("%A")
    statement = select(Faculty)
    if program_id:
        statement = statement.where(Faculty.program_id == program_id)
    all_faculty = list(db.execute(statement).scalars())
    if not all_faculty:
        return []

    course = db.get(Course, course_id) if course_id else None
    if program_id and course is not None and course.program_id != program_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Course does not belong to the selected program")
    target_department = None
    target_faculty_id = None
    if course and course.faculty_id:
        target = db.get(Faculty, course.faculty_id)
        if target is not None:
            target_department = target.department
            target_faculty_id = target.id

    occupied: set[str] = set()
    official = db.get(OfficialTimetable, 1)
    if official is not None:
        payload = OfficialTimetablePayload.model_validate(official.payload)
        for slot in payload.timetable_data:
            if slot.day != day_name:
                continue
            occupied.add(slot.facultyId)
            for assistant_id in slot.assistant_faculty_ids:
                occupied.add(assistant_id)

    candidates: list[dict] = []
    for item in all_faculty:
        if target_faculty_id and item.id == target_faculty_id:
            continue
        if item.availability and day_name not in item.availability:
            continue
        score = 0
        if target_department and item.department == target_department:
            score += 50
        score += max(0, item.max_hours - item.workload_hours)
        if item.id in occupied:
            score -= 20
        candidates.append(
            {
                "faculty_id": item.id,
                "name": item.name,
                "department": item.department,
                "designation": item.designation,
                "workload_hours": item.workload_hours,
                "max_hours": item.max_hours,
                "score": score,
                "occupied_on_day": item.id in occupied,
            }
        )

    candidates.sort(key=lambda value: value["score"], reverse=True)
    return candidates[:10]
