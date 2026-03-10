from math import floor, isclose

from pydantic import BaseModel, Field, model_validator

from app.models.course import CourseType


class CourseBase(BaseModel):
    program_id: str | None = Field(default=None, min_length=1, max_length=36)
    code: str = Field(min_length=1, max_length=50)
    name: str = Field(min_length=1, max_length=200)
    type: CourseType
    credits: float = Field(ge=0, le=40, multiple_of=0.5)
    duration_hours: int = Field(ge=1, le=8)
    sections: int = Field(ge=1, le=50)
    hours_per_week: int = Field(ge=1, le=40)
    semester_number: int = Field(default=1, ge=1, le=20)
    batch_year: int = Field(default=1, ge=1, le=4)
    theory_hours: int = Field(default=0, ge=0, le=40)
    lab_hours: int = Field(default=0, ge=0, le=40)
    tutorial_hours: int = Field(default=0, ge=0, le=40)
    batch_segregation: bool = True
    practical_contiguous_slots: int = Field(default=2, ge=1, le=40)
    assign_faculty: bool = True
    assign_classroom: bool = True
    default_room_id: str | None = Field(default=None, max_length=36)
    elective_category: str | None = Field(default=None, max_length=120)
    faculty_id: str | None = Field(default=None, max_length=36)

    @model_validator(mode="after")
    def validate_credit_split(self) -> "CourseBase":
        split_total = self.theory_hours + self.tutorial_hours + self.lab_hours
        if split_total == 0:
            if self.type == CourseType.lab:
                self.lab_hours = self.hours_per_week
            else:
                self.theory_hours = self.hours_per_week
        elif split_total != self.hours_per_week:
            raise ValueError("hours_per_week must equal lecture + tutorial + practical (L + T + P)")

        expected = self.theory_hours + self.tutorial_hours + self.lab_hours
        if expected != self.hours_per_week:
            raise ValueError("hours_per_week must equal lecture + tutorial + practical (L + T + P)")
        if expected <= 0:
            raise ValueError("Course must define at least one theory/lab/tutorial hour")

        raw_computed_credits = float(self.theory_hours + self.tutorial_hours + (self.lab_hours / 2.0))
        institutional_designated_credits = float(max(0, floor(raw_computed_credits + 1e-9)))
        if not isclose(self.credits, institutional_designated_credits, abs_tol=0.01):
            self.credits = institutional_designated_credits

        if self.lab_hours <= 0:
            self.practical_contiguous_slots = 1
        else:
            if self.practical_contiguous_slots > self.lab_hours:
                raise ValueError("practical_contiguous_slots must be <= practical hours (P)")
        if self.type == CourseType.elective:
            if "assign_faculty" not in self.model_fields_set:
                self.assign_faculty = False
            if "assign_classroom" not in self.model_fields_set:
                self.assign_classroom = False

        if not self.assign_faculty:
            self.faculty_id = None
        if not self.assign_classroom:
            self.default_room_id = None
        return self


class CourseCreate(CourseBase):
    pass


class CourseUpdate(BaseModel):
    program_id: str | None = Field(default=None, min_length=1, max_length=36)
    code: str | None = Field(default=None, min_length=1, max_length=50)
    name: str | None = Field(default=None, min_length=1, max_length=200)
    type: CourseType | None = None
    credits: float | None = Field(default=None, ge=0, le=40, multiple_of=0.5)
    duration_hours: int | None = Field(default=None, ge=1, le=8)
    sections: int | None = Field(default=None, ge=1, le=50)
    hours_per_week: int | None = Field(default=None, ge=1, le=40)
    semester_number: int | None = Field(default=None, ge=1, le=20)
    batch_year: int | None = Field(default=None, ge=1, le=4)
    theory_hours: int | None = Field(default=None, ge=0, le=40)
    lab_hours: int | None = Field(default=None, ge=0, le=40)
    tutorial_hours: int | None = Field(default=None, ge=0, le=40)
    batch_segregation: bool | None = None
    practical_contiguous_slots: int | None = Field(default=None, ge=1, le=40)
    assign_faculty: bool | None = None
    assign_classroom: bool | None = None
    default_room_id: str | None = Field(default=None, max_length=36)
    elective_category: str | None = Field(default=None, max_length=120)
    faculty_id: str | None = Field(default=None, max_length=36)


class CourseOut(CourseBase):
    id: str

    model_config = {"from_attributes": True}
