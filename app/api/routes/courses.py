from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db, require_roles
from app.models.course import Course
from app.models.program import Program
from app.models.program_structure import ProgramCourse
from app.models.user import User, UserRole
from app.schemas.course import CourseBase, CourseCreate, CourseOut, CourseUpdate
from app.services.notifications import notify_admin_update
from app.services.reevaluation import record_curriculum_change

router = APIRouter()


@router.get("/", response_model=list[CourseOut])
def list_courses(
    program_id: str | None = Query(default=None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[CourseOut]:
    statement = select(Course)
    if program_id:
        statement = statement.where(Course.program_id == program_id)
    elif current_user.role in {UserRole.faculty, UserRole.student} and current_user.program_id:
        statement = statement.where(Course.program_id == current_user.program_id)
    return list(db.execute(statement).scalars())


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
            detail="No program available. Create a program before adding courses.",
        )
    return default_program.id


@router.post("/", response_model=CourseOut, status_code=status.HTTP_201_CREATED)
def create_course(
    payload: CourseCreate,
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> CourseOut:
    payload_data = payload.model_dump()
    resolved_program_id = _resolve_program_id(db, payload_data.get("program_id"))
    payload_data["program_id"] = resolved_program_id

    existing = db.execute(
        select(Course).where(
            Course.program_id == resolved_program_id,
            Course.code == payload_data["code"],
        )
    ).scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Course code already exists in this program")
    course = Course(**payload_data)
    db.add(course)
    notify_admin_update(
        db,
        title="Course Created",
        message=f"{current_user.name} created course {payload.code} ({payload.name}).",
        actor_user_id=current_user.id,
    )
    db.commit()
    db.refresh(course)
    return course


@router.put("/{course_id}", response_model=CourseOut)
def update_course(
    course_id: str,
    payload: CourseUpdate,
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> CourseOut:
    course = db.get(Course, course_id)
    if course is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Course not found")

    data = payload.model_dump(exclude_unset=True)
    if "program_id" in data:
        data["program_id"] = _resolve_program_id(db, data["program_id"])

    if "code" in data or "program_id" in data:
        candidate_code = data.get("code", course.code)
        candidate_program_id = data.get("program_id", course.program_id)
        existing = db.execute(
            select(Course).where(
                Course.program_id == candidate_program_id,
                Course.code == candidate_code,
                Course.id != course_id,
            )
        ).scalar_one_or_none()
        if existing:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Course code already exists in this program")

    if data:
        merged = {
            "program_id": data.get("program_id", course.program_id),
            "code": data.get("code", course.code),
            "name": data.get("name", course.name),
            "type": data.get("type", course.type),
            "credits": data.get("credits", course.credits),
            "duration_hours": data.get("duration_hours", course.duration_hours),
            "sections": data.get("sections", course.sections),
            "hours_per_week": data.get("hours_per_week", course.hours_per_week),
            "semester_number": data.get("semester_number", course.semester_number),
            "batch_year": data.get("batch_year", course.batch_year),
            "theory_hours": data.get("theory_hours", course.theory_hours),
            "lab_hours": data.get("lab_hours", course.lab_hours),
            "tutorial_hours": data.get("tutorial_hours", course.tutorial_hours),
            "batch_segregation": data.get("batch_segregation", course.batch_segregation),
            "practical_contiguous_slots": data.get(
                "practical_contiguous_slots",
                course.practical_contiguous_slots,
            ),
            "assign_faculty": data.get("assign_faculty", course.assign_faculty),
            "assign_classroom": data.get("assign_classroom", course.assign_classroom),
            "default_room_id": data.get("default_room_id", course.default_room_id),
            "elective_category": data.get("elective_category", course.elective_category),
            "faculty_id": data.get("faculty_id", course.faculty_id),
        }
        normalized = CourseBase.model_validate(merged).model_dump()
        normalized_keys = {
            "credits",
            "hours_per_week",
            "theory_hours",
            "lab_hours",
            "tutorial_hours",
            "batch_segregation",
            "practical_contiguous_slots",
            "assign_faculty",
            "assign_classroom",
            "default_room_id",
            "elective_category",
            "faculty_id",
        }
        if "program_id" in data:
            normalized_keys.add("program_id")
        for key in normalized_keys:
            data[key] = normalized[key]

    for key, value in data.items():
        setattr(course, key, value)
    if data:
        mappings = list(
            db.execute(
                select(ProgramCourse.program_id, ProgramCourse.term_number).where(
                    ProgramCourse.course_id == course_id
                )
            ).all()
        )
        for program_id, term_number in mappings:
            record_curriculum_change(
                db,
                program_id=program_id,
                term_number=term_number,
                change_type="course_updated",
                entity_type="course",
                entity_id=course_id,
                description=f"Course {course.code} updated",
                details={"changed_fields": sorted(data.keys())},
                triggered_by=current_user,
            )
        notify_admin_update(
            db,
            title="Course Updated",
            message=f"{current_user.name} updated course {course.code}.",
            actor_user_id=current_user.id,
        )
    db.commit()
    db.refresh(course)
    return course


@router.delete("/{course_id}")
def delete_course(
    course_id: str,
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> dict:
    course = db.get(Course, course_id)
    if course is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Course not found")
    mappings = list(
        db.execute(
            select(ProgramCourse.program_id, ProgramCourse.term_number).where(
                ProgramCourse.course_id == course_id
            )
        ).all()
    )
    for program_id, term_number in mappings:
        record_curriculum_change(
            db,
            program_id=program_id,
            term_number=term_number,
            change_type="course_deleted",
            entity_type="course",
            entity_id=course_id,
            description=f"Course {course.code} deleted",
            details={},
            triggered_by=current_user,
        )
    notify_admin_update(
        db,
        title="Course Deleted",
        message=f"{current_user.name} deleted course {course.code}.",
        actor_user_id=current_user.id,
    )
    db.delete(course)
    db.commit()
    return {"success": True}
