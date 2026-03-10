from collections import defaultdict

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db, require_roles
from app.models.course import Course, CourseType
from app.models.program import Program
from app.models.program_structure import (
    ProgramCourse,
    ProgramElectiveGroup,
    ProgramElectiveGroupMember,
    ProgramSection,
    ProgramSharedLectureGroup,
    ProgramSharedLectureGroupMember,
    ProgramTerm,
)
from app.models.user import User, UserRole
from app.schemas.program_structure import (
    ProgramCourseCreate,
    ProgramCourseOut,
    ProgramElectiveGroupCreate,
    ProgramElectiveGroupOut,
    ProgramElectiveGroupUpdate,
    ProgramSectionCreate,
    ProgramSectionOut,
    ProgramSharedLectureGroupCreate,
    ProgramSharedLectureGroupOut,
    ProgramSharedLectureGroupUpdate,
    ProgramTermCreate,
    ProgramTermOut,
)
from app.services.reevaluation import record_curriculum_change

router = APIRouter()


def get_program(program_id: str, db: Session) -> Program:
    program = db.get(Program, program_id)
    if program is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Program not found")
    return program


def validate_program_course_prerequisites(
    *,
    db: Session,
    program_id: str,
    payload: ProgramCourseCreate,
) -> None:
    prerequisite_ids = payload.prerequisite_course_ids
    if not prerequisite_ids:
        return

    if payload.course_id in prerequisite_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A course cannot be a prerequisite of itself",
        )

    existing_courses = set(
        db.execute(
            select(Course.id).where(
                Course.id.in_(prerequisite_ids),
                Course.program_id == program_id,
            )
        ).scalars().all()
    )
    missing_courses = sorted(set(prerequisite_ids) - existing_courses)
    if missing_courses:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prerequisite courses not found in this program: {', '.join(missing_courses)}",
        )

    mapped_prior_courses = set(
        db.execute(
            select(ProgramCourse.course_id).where(
                ProgramCourse.program_id == program_id,
                ProgramCourse.course_id.in_(prerequisite_ids),
                ProgramCourse.term_number < payload.term_number,
            )
        )
        .scalars()
        .all()
    )
    missing_prior_mapping = sorted(set(prerequisite_ids) - mapped_prior_courses)
    if missing_prior_mapping:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Prerequisites must be mapped to earlier terms in the same program: "
                f"{', '.join(missing_prior_mapping)}"
            ),
        )


def validate_elective_group_members(
    *,
    db: Session,
    program_id: str,
    payload: ProgramElectiveGroupCreate | ProgramElectiveGroupUpdate,
) -> None:
    rows = (
        db.execute(
            select(ProgramCourse, Course)
            .join(Course, Course.id == ProgramCourse.course_id)
            .where(
                ProgramCourse.id.in_(payload.program_course_ids),
                ProgramCourse.program_id == program_id,
                ProgramCourse.term_number == payload.term_number,
            )
        )
        .all()
    )
    found_program_course_ids = {program_course.id for program_course, _ in rows}
    missing = sorted(set(payload.program_course_ids) - found_program_course_ids)
    if missing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Elective group members must be program courses in the same term. "
                f"Unknown ids: {', '.join(missing)}"
            ),
        )

    non_electives = sorted(
        {
            course.code
            for _, course in rows
            if course.type != CourseType.elective
        }
    )
    if non_electives:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Elective overlap groups can only include elective courses. "
                f"Invalid courses: {', '.join(non_electives)}"
            ),
        )


def build_elective_group_response(
    *,
    group: ProgramElectiveGroup,
    member_ids: list[str],
) -> ProgramElectiveGroupOut:
    return ProgramElectiveGroupOut(
        id=group.id,
        term_number=group.term_number,
        name=group.name,
        conflict_policy=group.conflict_policy,
        program_course_ids=member_ids,
    )


def validate_shared_lecture_group(
    *,
    db: Session,
    program_id: str,
    payload: ProgramSharedLectureGroupCreate | ProgramSharedLectureGroupUpdate,
) -> None:
    course = db.get(Course, payload.course_id)
    if course is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Course not found")
    if course.program_id != program_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Shared lecture group course must belong to the same program",
        )
    if course.type == CourseType.lab:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Shared lecture groups cannot be configured for lab courses",
        )

    mapped_course = db.execute(
        select(ProgramCourse.id).where(
            ProgramCourse.program_id == program_id,
            ProgramCourse.term_number == payload.term_number,
            ProgramCourse.course_id == payload.course_id,
        )
    ).scalar_one_or_none()
    if mapped_course is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Course must be assigned to the same program term before creating a shared lecture group",
        )

    existing_sections = set(
        db.execute(
            select(ProgramSection.name).where(
                ProgramSection.program_id == program_id,
                ProgramSection.term_number == payload.term_number,
                ProgramSection.name.in_(payload.section_names),
            )
        ).scalars()
    )
    missing_sections = sorted(set(payload.section_names) - existing_sections)
    if missing_sections:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Section(s) not found in the same term: {', '.join(missing_sections)}",
        )


def build_shared_lecture_group_response(
    *,
    group: ProgramSharedLectureGroup,
    section_names: list[str],
) -> ProgramSharedLectureGroupOut:
    return ProgramSharedLectureGroupOut(
        id=group.id,
        term_number=group.term_number,
        name=group.name,
        course_id=group.course_id,
        section_names=section_names,
    )


def track_curriculum_program_term_change(
    *,
    db: Session,
    current_user: User,
    program_id: str,
    term_number: int | None,
    change_type: str,
    entity_type: str,
    entity_id: str | None,
    description: str,
    details: dict | None = None,
) -> None:
    record_curriculum_change(
        db,
        program_id=program_id,
        term_number=term_number,
        change_type=change_type,
        entity_type=entity_type,
        entity_id=entity_id,
        description=description,
        details=details,
        triggered_by=current_user,
    )


@router.get("/programs/{program_id}/terms", response_model=list[ProgramTermOut])
def list_terms(
    program_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[ProgramTermOut]:
    get_program(program_id, db)
    return list(db.execute(select(ProgramTerm).where(ProgramTerm.program_id == program_id)).scalars())


@router.post("/programs/{program_id}/terms", response_model=ProgramTermOut, status_code=status.HTTP_201_CREATED)
def create_term(
    program_id: str,
    payload: ProgramTermCreate,
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> ProgramTermOut:
    get_program(program_id, db)
    existing = db.execute(
        select(ProgramTerm).where(
            ProgramTerm.program_id == program_id,
            ProgramTerm.term_number == payload.term_number,
        )
    ).scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Term already exists")
    term = ProgramTerm(program_id=program_id, **payload.model_dump())
    db.add(term)
    track_curriculum_program_term_change(
        db=db,
        current_user=current_user,
        program_id=program_id,
        term_number=payload.term_number,
        change_type="term_created",
        entity_type="program_term",
        entity_id=None,
        description=f"Term {payload.term_number} created for program",
        details={"term_name": payload.name},
    )
    db.commit()
    db.refresh(term)
    return term


@router.delete("/programs/{program_id}/terms/{term_id}")
def delete_term(
    program_id: str,
    term_id: str,
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> dict:
    get_program(program_id, db)
    term = db.get(ProgramTerm, term_id)
    if term is None or term.program_id != program_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Term not found")
    track_curriculum_program_term_change(
        db=db,
        current_user=current_user,
        program_id=program_id,
        term_number=term.term_number,
        change_type="term_deleted",
        entity_type="program_term",
        entity_id=term_id,
        description=f"Term {term.term_number} deleted from program",
        details={"term_name": term.name},
    )
    db.delete(term)
    db.commit()
    return {"success": True}


@router.get("/programs/{program_id}/sections", response_model=list[ProgramSectionOut])
def list_sections(
    program_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[ProgramSectionOut]:
    get_program(program_id, db)
    return list(db.execute(select(ProgramSection).where(ProgramSection.program_id == program_id)).scalars())


@router.post("/programs/{program_id}/sections", response_model=ProgramSectionOut, status_code=status.HTTP_201_CREATED)
def create_section(
    program_id: str,
    payload: ProgramSectionCreate,
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> ProgramSectionOut:
    get_program(program_id, db)
    existing = db.execute(
        select(ProgramSection).where(
            ProgramSection.program_id == program_id,
            ProgramSection.term_number == payload.term_number,
            ProgramSection.name == payload.name,
        )
    ).scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Section already exists")
    section = ProgramSection(program_id=program_id, **payload.model_dump())
    db.add(section)
    track_curriculum_program_term_change(
        db=db,
        current_user=current_user,
        program_id=program_id,
        term_number=payload.term_number,
        change_type="section_created",
        entity_type="program_section",
        entity_id=None,
        description=f"Section {payload.name} created",
        details={"capacity": payload.capacity},
    )
    db.commit()
    db.refresh(section)
    return section


@router.delete("/programs/{program_id}/sections/{section_id}")
def delete_section(
    program_id: str,
    section_id: str,
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> dict:
    get_program(program_id, db)
    section = db.get(ProgramSection, section_id)
    if section is None or section.program_id != program_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Section not found")

    affected_shared_group_ids = set(
        db.execute(
            select(ProgramSharedLectureGroupMember.group_id)
            .join(ProgramSharedLectureGroup, ProgramSharedLectureGroup.id == ProgramSharedLectureGroupMember.group_id)
            .where(
                ProgramSharedLectureGroup.program_id == program_id,
                ProgramSharedLectureGroup.term_number == section.term_number,
                ProgramSharedLectureGroupMember.section_name == section.name,
            )
        ).scalars()
    )
    if affected_shared_group_ids:
        db.execute(
            delete(ProgramSharedLectureGroupMember).where(
                ProgramSharedLectureGroupMember.group_id.in_(affected_shared_group_ids),
                ProgramSharedLectureGroupMember.section_name == section.name,
            )
        )
        for group in db.execute(
            select(ProgramSharedLectureGroup).where(
                ProgramSharedLectureGroup.id.in_(affected_shared_group_ids),
                ProgramSharedLectureGroup.program_id == program_id,
            )
        ).scalars():
            remaining_sections = list(
                db.execute(
                    select(ProgramSharedLectureGroupMember.section_name).where(
                        ProgramSharedLectureGroupMember.group_id == group.id
                    )
                ).scalars()
            )
            if len(remaining_sections) < 2:
                db.execute(
                    delete(ProgramSharedLectureGroupMember).where(
                        ProgramSharedLectureGroupMember.group_id == group.id
                    )
                )
                db.delete(group)

    track_curriculum_program_term_change(
        db=db,
        current_user=current_user,
        program_id=program_id,
        term_number=section.term_number,
        change_type="section_deleted",
        entity_type="program_section",
        entity_id=section_id,
        description=f"Section {section.name} deleted",
        details={},
    )
    db.delete(section)
    db.commit()
    return {"success": True}


@router.get("/programs/{program_id}/courses", response_model=list[ProgramCourseOut])
def list_program_courses(
    program_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[ProgramCourseOut]:
    get_program(program_id, db)
    return list(db.execute(select(ProgramCourse).where(ProgramCourse.program_id == program_id)).scalars())


@router.post("/programs/{program_id}/courses", response_model=ProgramCourseOut, status_code=status.HTTP_201_CREATED)
def add_program_course(
    program_id: str,
    payload: ProgramCourseCreate,
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> ProgramCourseOut:
    get_program(program_id, db)
    course = db.get(Course, payload.course_id)
    if course is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Course not found")
    if course.program_id != program_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Course must belong to the same program before assignment",
        )
    existing = db.execute(
        select(ProgramCourse).where(
            ProgramCourse.program_id == program_id,
            ProgramCourse.term_number == payload.term_number,
            ProgramCourse.course_id == payload.course_id,
        )
    ).scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Course already assigned")
    validate_program_course_prerequisites(db=db, program_id=program_id, payload=payload)
    program_course = ProgramCourse(program_id=program_id, **payload.model_dump())
    db.add(program_course)
    track_curriculum_program_term_change(
        db=db,
        current_user=current_user,
        program_id=program_id,
        term_number=payload.term_number,
        change_type="program_course_added",
        entity_type="program_course",
        entity_id=None,
        description=f"Course {payload.course_id} assigned to term {payload.term_number}",
        details={"is_required": payload.is_required},
    )
    db.commit()
    db.refresh(program_course)
    return program_course


@router.delete("/programs/{program_id}/courses/{program_course_id}")
def delete_program_course(
    program_id: str,
    program_course_id: str,
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> dict:
    get_program(program_id, db)
    program_course = db.get(ProgramCourse, program_course_id)
    if program_course is None or program_course.program_id != program_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Program course not found")

    affected_group_ids = set(
        db.execute(
            select(ProgramElectiveGroupMember.group_id).where(
                ProgramElectiveGroupMember.program_course_id == program_course_id
            )
        ).scalars()
    )
    if affected_group_ids:
        db.execute(
            delete(ProgramElectiveGroupMember).where(
                ProgramElectiveGroupMember.program_course_id == program_course_id
            )
        )

    for group in db.execute(
        select(ProgramElectiveGroup).where(
            ProgramElectiveGroup.id.in_(affected_group_ids),
            ProgramElectiveGroup.program_id == program_id,
        )
    ).scalars():
        remaining_member_ids = list(
            db.execute(
                select(ProgramElectiveGroupMember.program_course_id).where(
                    ProgramElectiveGroupMember.group_id == group.id
                )
            ).scalars()
        )
        if len(remaining_member_ids) < 2:
            db.execute(delete(ProgramElectiveGroupMember).where(ProgramElectiveGroupMember.group_id == group.id))
            db.delete(group)

    shared_groups = list(
        db.execute(
            select(ProgramSharedLectureGroup).where(
                ProgramSharedLectureGroup.program_id == program_id,
                ProgramSharedLectureGroup.term_number == program_course.term_number,
                ProgramSharedLectureGroup.course_id == program_course.course_id,
            )
        ).scalars()
    )
    for shared_group in shared_groups:
        db.execute(
            delete(ProgramSharedLectureGroupMember).where(
                ProgramSharedLectureGroupMember.group_id == shared_group.id
            )
        )
        db.delete(shared_group)

    track_curriculum_program_term_change(
        db=db,
        current_user=current_user,
        program_id=program_id,
        term_number=program_course.term_number,
        change_type="program_course_deleted",
        entity_type="program_course",
        entity_id=program_course_id,
        description=f"Course {program_course.course_id} removed from term {program_course.term_number}",
        details={},
    )
    db.delete(program_course)
    db.commit()
    return {"success": True}


@router.get("/programs/{program_id}/elective-groups", response_model=list[ProgramElectiveGroupOut])
def list_program_elective_groups(
    program_id: str,
    term_number: int | None = Query(default=None, ge=1, le=20),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[ProgramElectiveGroupOut]:
    get_program(program_id, db)
    query = select(ProgramElectiveGroup).where(ProgramElectiveGroup.program_id == program_id)
    if term_number is not None:
        query = query.where(ProgramElectiveGroup.term_number == term_number)
    groups = list(
        db.execute(
            query.order_by(ProgramElectiveGroup.term_number.asc(), ProgramElectiveGroup.name.asc())
        ).scalars()
    )
    if not groups:
        return []

    group_ids = [group.id for group in groups]
    members_by_group: dict[str, list[str]] = defaultdict(list)
    for member in db.execute(
        select(ProgramElectiveGroupMember).where(ProgramElectiveGroupMember.group_id.in_(group_ids))
    ).scalars():
        members_by_group[member.group_id].append(member.program_course_id)

    return [
        build_elective_group_response(
            group=group,
            member_ids=members_by_group.get(group.id, []),
        )
        for group in groups
    ]


@router.post(
    "/programs/{program_id}/elective-groups",
    response_model=ProgramElectiveGroupOut,
    status_code=status.HTTP_201_CREATED,
)
def create_program_elective_group(
    program_id: str,
    payload: ProgramElectiveGroupCreate,
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> ProgramElectiveGroupOut:
    get_program(program_id, db)
    validate_elective_group_members(db=db, program_id=program_id, payload=payload)

    existing = db.execute(
        select(ProgramElectiveGroup).where(
            ProgramElectiveGroup.program_id == program_id,
            ProgramElectiveGroup.term_number == payload.term_number,
            ProgramElectiveGroup.name == payload.name,
        )
    ).scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Elective group name already exists")

    group = ProgramElectiveGroup(
        program_id=program_id,
        term_number=payload.term_number,
        name=payload.name,
        conflict_policy=payload.conflict_policy,
    )
    db.add(group)
    db.flush()
    db.add_all(
        [
            ProgramElectiveGroupMember(group_id=group.id, program_course_id=program_course_id)
            for program_course_id in payload.program_course_ids
        ]
    )
    track_curriculum_program_term_change(
        db=db,
        current_user=current_user,
        program_id=program_id,
        term_number=payload.term_number,
        change_type="elective_group_created",
        entity_type="program_elective_group",
        entity_id=group.id,
        description=f"Elective overlap group '{payload.name}' created",
        details={"member_count": len(payload.program_course_ids)},
    )
    db.commit()
    db.refresh(group)
    return build_elective_group_response(group=group, member_ids=payload.program_course_ids)


@router.put("/programs/{program_id}/elective-groups/{group_id}", response_model=ProgramElectiveGroupOut)
def update_program_elective_group(
    program_id: str,
    group_id: str,
    payload: ProgramElectiveGroupUpdate,
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> ProgramElectiveGroupOut:
    get_program(program_id, db)
    group = db.get(ProgramElectiveGroup, group_id)
    if group is None or group.program_id != program_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Elective group not found")

    existing = db.execute(
        select(ProgramElectiveGroup).where(
            ProgramElectiveGroup.program_id == program_id,
            ProgramElectiveGroup.term_number == payload.term_number,
            ProgramElectiveGroup.name == payload.name,
            ProgramElectiveGroup.id != group_id,
        )
    ).scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Elective group name already exists")

    validate_elective_group_members(db=db, program_id=program_id, payload=payload)

    group.term_number = payload.term_number
    group.name = payload.name
    group.conflict_policy = payload.conflict_policy
    db.execute(delete(ProgramElectiveGroupMember).where(ProgramElectiveGroupMember.group_id == group_id))
    db.add_all(
        [
            ProgramElectiveGroupMember(group_id=group_id, program_course_id=program_course_id)
            for program_course_id in payload.program_course_ids
        ]
    )
    track_curriculum_program_term_change(
        db=db,
        current_user=current_user,
        program_id=program_id,
        term_number=payload.term_number,
        change_type="elective_group_updated",
        entity_type="program_elective_group",
        entity_id=group_id,
        description=f"Elective overlap group '{payload.name}' updated",
        details={"member_count": len(payload.program_course_ids)},
    )
    db.commit()
    db.refresh(group)
    return build_elective_group_response(group=group, member_ids=payload.program_course_ids)


@router.delete("/programs/{program_id}/elective-groups/{group_id}")
def delete_program_elective_group(
    program_id: str,
    group_id: str,
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> dict:
    get_program(program_id, db)
    group = db.get(ProgramElectiveGroup, group_id)
    if group is None or group.program_id != program_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Elective group not found")
    track_curriculum_program_term_change(
        db=db,
        current_user=current_user,
        program_id=program_id,
        term_number=group.term_number,
        change_type="elective_group_deleted",
        entity_type="program_elective_group",
        entity_id=group_id,
        description=f"Elective overlap group '{group.name}' deleted",
        details={},
    )
    db.execute(delete(ProgramElectiveGroupMember).where(ProgramElectiveGroupMember.group_id == group_id))
    db.delete(group)
    db.commit()
    return {"success": True}


@router.get("/programs/{program_id}/shared-lecture-groups", response_model=list[ProgramSharedLectureGroupOut])
def list_program_shared_lecture_groups(
    program_id: str,
    term_number: int | None = Query(default=None, ge=1, le=20),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[ProgramSharedLectureGroupOut]:
    get_program(program_id, db)
    query = select(ProgramSharedLectureGroup).where(ProgramSharedLectureGroup.program_id == program_id)
    if term_number is not None:
        query = query.where(ProgramSharedLectureGroup.term_number == term_number)
    groups = list(
        db.execute(
            query.order_by(ProgramSharedLectureGroup.term_number.asc(), ProgramSharedLectureGroup.name.asc())
        ).scalars()
    )
    if not groups:
        return []

    group_ids = [group.id for group in groups]
    sections_by_group: dict[str, list[str]] = defaultdict(list)
    for member in db.execute(
        select(ProgramSharedLectureGroupMember).where(ProgramSharedLectureGroupMember.group_id.in_(group_ids))
    ).scalars():
        sections_by_group[member.group_id].append(member.section_name)

    return [
        build_shared_lecture_group_response(
            group=group,
            section_names=sorted(sections_by_group.get(group.id, [])),
        )
        for group in groups
    ]


@router.post(
    "/programs/{program_id}/shared-lecture-groups",
    response_model=ProgramSharedLectureGroupOut,
    status_code=status.HTTP_201_CREATED,
)
def create_program_shared_lecture_group(
    program_id: str,
    payload: ProgramSharedLectureGroupCreate,
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> ProgramSharedLectureGroupOut:
    get_program(program_id, db)
    validate_shared_lecture_group(db=db, program_id=program_id, payload=payload)

    existing = db.execute(
        select(ProgramSharedLectureGroup).where(
            ProgramSharedLectureGroup.program_id == program_id,
            ProgramSharedLectureGroup.term_number == payload.term_number,
            ProgramSharedLectureGroup.name == payload.name,
        )
    ).scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Shared lecture group name already exists")

    group = ProgramSharedLectureGroup(
        program_id=program_id,
        term_number=payload.term_number,
        name=payload.name,
        course_id=payload.course_id,
    )
    db.add(group)
    db.flush()
    db.add_all(
        [
            ProgramSharedLectureGroupMember(group_id=group.id, section_name=section_name)
            for section_name in payload.section_names
        ]
    )
    track_curriculum_program_term_change(
        db=db,
        current_user=current_user,
        program_id=program_id,
        term_number=payload.term_number,
        change_type="shared_lecture_group_created",
        entity_type="program_shared_lecture_group",
        entity_id=group.id,
        description=f"Shared lecture group '{payload.name}' created",
        details={"section_count": len(payload.section_names), "course_id": payload.course_id},
    )
    db.commit()
    db.refresh(group)
    return build_shared_lecture_group_response(group=group, section_names=payload.section_names)


@router.put("/programs/{program_id}/shared-lecture-groups/{group_id}", response_model=ProgramSharedLectureGroupOut)
def update_program_shared_lecture_group(
    program_id: str,
    group_id: str,
    payload: ProgramSharedLectureGroupUpdate,
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> ProgramSharedLectureGroupOut:
    get_program(program_id, db)
    group = db.get(ProgramSharedLectureGroup, group_id)
    if group is None or group.program_id != program_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Shared lecture group not found")

    existing = db.execute(
        select(ProgramSharedLectureGroup).where(
            ProgramSharedLectureGroup.program_id == program_id,
            ProgramSharedLectureGroup.term_number == payload.term_number,
            ProgramSharedLectureGroup.name == payload.name,
            ProgramSharedLectureGroup.id != group_id,
        )
    ).scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Shared lecture group name already exists")

    validate_shared_lecture_group(db=db, program_id=program_id, payload=payload)

    group.term_number = payload.term_number
    group.name = payload.name
    group.course_id = payload.course_id
    db.execute(
        delete(ProgramSharedLectureGroupMember).where(
            ProgramSharedLectureGroupMember.group_id == group_id
        )
    )
    db.add_all(
        [
            ProgramSharedLectureGroupMember(group_id=group_id, section_name=section_name)
            for section_name in payload.section_names
        ]
    )
    track_curriculum_program_term_change(
        db=db,
        current_user=current_user,
        program_id=program_id,
        term_number=payload.term_number,
        change_type="shared_lecture_group_updated",
        entity_type="program_shared_lecture_group",
        entity_id=group_id,
        description=f"Shared lecture group '{payload.name}' updated",
        details={"section_count": len(payload.section_names), "course_id": payload.course_id},
    )
    db.commit()
    db.refresh(group)
    return build_shared_lecture_group_response(group=group, section_names=payload.section_names)


@router.delete("/programs/{program_id}/shared-lecture-groups/{group_id}")
def delete_program_shared_lecture_group(
    program_id: str,
    group_id: str,
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
    db: Session = Depends(get_db),
) -> dict:
    get_program(program_id, db)
    group = db.get(ProgramSharedLectureGroup, group_id)
    if group is None or group.program_id != program_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Shared lecture group not found")
    track_curriculum_program_term_change(
        db=db,
        current_user=current_user,
        program_id=program_id,
        term_number=group.term_number,
        change_type="shared_lecture_group_deleted",
        entity_type="program_shared_lecture_group",
        entity_id=group_id,
        description=f"Shared lecture group '{group.name}' deleted",
        details={"course_id": group.course_id},
    )
    db.execute(
        delete(ProgramSharedLectureGroupMember).where(
            ProgramSharedLectureGroupMember.group_id == group_id
        )
    )
    db.delete(group)
    db.commit()
    return {"success": True}
