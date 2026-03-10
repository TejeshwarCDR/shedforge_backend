from pydantic import BaseModel, Field, field_validator

from app.models.program_structure import ElectiveConflictPolicy


class ProgramTermBase(BaseModel):
    term_number: int = Field(ge=1, le=20)
    name: str = Field(min_length=1, max_length=100)
    credits_required: int = Field(ge=0, le=300)


class ProgramTermCreate(ProgramTermBase):
    pass


class ProgramTermOut(ProgramTermBase):
    id: str

    model_config = {"from_attributes": True}


class ProgramSectionBase(BaseModel):
    term_number: int = Field(ge=1, le=20)
    name: str = Field(min_length=1, max_length=50)
    capacity: int = Field(ge=0, le=500)


class ProgramSectionCreate(ProgramSectionBase):
    pass


class ProgramSectionOut(ProgramSectionBase):
    id: str

    model_config = {"from_attributes": True}


class ProgramCourseBase(BaseModel):
    term_number: int = Field(ge=1, le=20)
    course_id: str = Field(min_length=1, max_length=36)
    is_required: bool = True
    lab_batch_count: int = Field(default=1, ge=1, le=20)
    allow_parallel_batches: bool = True
    prerequisite_course_ids: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("prerequisite_course_ids")
    @classmethod
    def validate_prerequisites(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for value in values:
            course_id = value.strip()
            if not course_id:
                raise ValueError("Prerequisite course id cannot be empty")
            if len(course_id) > 36:
                raise ValueError("Prerequisite course id is too long")
            if course_id in seen:
                continue
            seen.add(course_id)
            cleaned.append(course_id)
        return cleaned


class ProgramCourseCreate(ProgramCourseBase):
    pass


class ProgramCourseOut(ProgramCourseBase):
    id: str

    model_config = {"from_attributes": True}


class ProgramElectiveGroupBase(BaseModel):
    term_number: int = Field(ge=1, le=20)
    name: str = Field(min_length=1, max_length=100)
    conflict_policy: ElectiveConflictPolicy = ElectiveConflictPolicy.no_overlap
    program_course_ids: list[str] = Field(min_length=2, max_length=30)

    @field_validator("program_course_ids")
    @classmethod
    def validate_program_course_ids(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for value in values:
            program_course_id = value.strip()
            if not program_course_id:
                raise ValueError("Program course id cannot be empty")
            if len(program_course_id) > 36:
                raise ValueError("Program course id is too long")
            if program_course_id in seen:
                continue
            seen.add(program_course_id)
            cleaned.append(program_course_id)
        if len(cleaned) < 2:
            raise ValueError("At least two unique program courses are required")
        return cleaned


class ProgramElectiveGroupCreate(ProgramElectiveGroupBase):
    pass


class ProgramElectiveGroupUpdate(ProgramElectiveGroupBase):
    pass


class ProgramElectiveGroupOut(ProgramElectiveGroupBase):
    id: str

    model_config = {"from_attributes": True}


class ProgramSharedLectureGroupBase(BaseModel):
    term_number: int = Field(ge=1, le=20)
    name: str = Field(min_length=1, max_length=100)
    course_id: str = Field(min_length=1, max_length=36)
    section_names: list[str] = Field(min_length=2, max_length=20)

    @field_validator("section_names")
    @classmethod
    def validate_section_names(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for value in values:
            section_name = value.strip()
            if not section_name:
                raise ValueError("Section name cannot be empty")
            if len(section_name) > 50:
                raise ValueError("Section name is too long")
            if section_name in seen:
                continue
            seen.add(section_name)
            cleaned.append(section_name)
        if len(cleaned) < 2:
            raise ValueError("At least two unique sections are required")
        return cleaned


class ProgramSharedLectureGroupCreate(ProgramSharedLectureGroupBase):
    pass


class ProgramSharedLectureGroupUpdate(ProgramSharedLectureGroupBase):
    pass


class ProgramSharedLectureGroupOut(ProgramSharedLectureGroupBase):
    id: str

    model_config = {"from_attributes": True}
