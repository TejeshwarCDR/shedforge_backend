from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, EmailStr, Field, field_validator, model_validator

DAY_VALUES = {
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
}

AVAILABILITY_VALUES = DAY_VALUES | {
    "Mon",
    "Tue",
    "Wed",
    "Thu",
    "Fri",
    "Sat",
    "Sun",
}

TIME_PATTERN = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


def parse_time_to_minutes(value: str) -> int:
    if not TIME_PATTERN.match(value):
        raise ValueError("Time must be in HH:MM 24-hour format")
    hours, minutes = value.split(":")
    return int(hours) * 60 + int(minutes)


class FacultyPayload(BaseModel):
    id: str = Field(min_length=1, max_length=36)
    name: str = Field(min_length=1, max_length=200)
    department: str = Field(min_length=1, max_length=200)
    workloadHours: int = Field(ge=0, le=200)
    maxHours: int = Field(ge=0, le=200)
    availability: list[str] = Field(default_factory=list, max_length=14)
    email: EmailStr
    currentWorkload: int | None = None

    @field_validator("availability")
    @classmethod
    def validate_availability(cls, value: list[str]) -> list[str]:
        cleaned = [day.strip() for day in value if day.strip()]
        invalid = [day for day in cleaned if day not in AVAILABILITY_VALUES]
        if invalid:
            raise ValueError(f"Invalid availability day(s): {', '.join(invalid)}")
        return cleaned


class CoursePayload(BaseModel):
    id: str = Field(min_length=1, max_length=36)
    code: str = Field(min_length=1, max_length=50)
    name: str = Field(min_length=1, max_length=200)
    type: Literal["theory", "lab", "elective"]
    credits: float = Field(ge=0, le=40, multiple_of=0.5)
    facultyId: str = Field(min_length=1, max_length=36)
    duration: int = Field(ge=1, le=8)
    sections: int | None = None
    hoursPerWeek: int = Field(ge=1, le=40)
    semester_number: int | None = Field(default=None, alias="semesterNumber", ge=1, le=20)
    batch_year: int | None = Field(default=None, alias="batchYear", ge=1, le=4)
    theory_hours: int | None = Field(default=None, alias="theoryHours", ge=0, le=40)
    lab_hours: int | None = Field(default=None, alias="labHours", ge=0, le=40)
    tutorial_hours: int | None = Field(default=None, alias="tutorialHours", ge=0, le=40)
    batch_segregation: bool = Field(default=True, alias="batchSegregation")
    practical_contiguous_slots: int = Field(default=2, alias="practicalContiguousSlots", ge=1, le=40)
    assign_faculty: bool = Field(default=True, alias="assignFaculty")
    assign_classroom: bool = Field(default=True, alias="assignClassroom")
    default_room_id: str | None = Field(default=None, alias="defaultRoomId", min_length=1, max_length=36)
    elective_category: str | None = Field(default=None, alias="electiveCategory", max_length=120)

    @model_validator(mode="after")
    def validate_credit_split(self) -> "CoursePayload":
        has_explicit_split = any(value is not None for value in (self.theory_hours, self.lab_hours, self.tutorial_hours))
        theory = 0 if self.theory_hours is None else self.theory_hours
        lab = 0 if self.lab_hours is None else self.lab_hours
        tutorial = 0 if self.tutorial_hours is None else self.tutorial_hours

        if not has_explicit_split:
            if self.type == "lab":
                theory = 0
                lab = self.hoursPerWeek
                tutorial = 0
            else:
                theory = self.hoursPerWeek
                lab = 0
                tutorial = 0

        self.theory_hours = theory
        self.lab_hours = lab
        self.tutorial_hours = tutorial

        split_total = self.theory_hours + self.lab_hours + self.tutorial_hours
        if split_total <= 0:
            raise ValueError("Course must include at least one theory/tutorial/lab hour")
        if split_total != self.hoursPerWeek:
            raise ValueError("hoursPerWeek must equal theoryHours + labHours + tutorialHours")
        computed_credits = float(self.theory_hours + self.tutorial_hours + (self.lab_hours / 2.0))
        if abs(self.credits - computed_credits) > 0.01:
            self.credits = round(computed_credits, 2)
        if self.lab_hours <= 0:
            self.practical_contiguous_slots = 1
        elif self.practical_contiguous_slots > self.lab_hours:
            raise ValueError("practicalContiguousSlots must be <= labHours")
        if self.type == "elective":
            if "assign_faculty" not in self.model_fields_set and "assignFaculty" not in self.model_fields_set:
                self.assign_faculty = False
            if "assign_classroom" not in self.model_fields_set and "assignClassroom" not in self.model_fields_set:
                self.assign_classroom = False
        return self


class RoomPayload(BaseModel):
    id: str = Field(min_length=1, max_length=36)
    name: str = Field(min_length=1, max_length=100)
    capacity: int = Field(ge=1, le=1000)
    type: Literal["lecture", "lab", "seminar"]
    building: str = Field(min_length=1, max_length=200)
    hasLabEquipment: bool | None = None
    utilization: int | None = None
    hasProjector: bool | None = None


class TimeSlotPayload(BaseModel):
    id: str = Field(min_length=1, max_length=36)
    day: str
    startTime: str
    endTime: str
    courseId: str = Field(min_length=1, max_length=36)
    roomId: str = Field(min_length=1, max_length=36)
    facultyId: str = Field(min_length=1, max_length=36)
    section: str = Field(min_length=1, max_length=50)
    batch: str | None = Field(default=None, min_length=1, max_length=50)
    studentCount: int | None = Field(default=None, alias="studentCount", ge=1, le=2000)
    sessionType: Literal["theory", "tutorial", "lab"] | None = Field(default=None, alias="sessionType")
    assistant_faculty_ids: list[str] = Field(default_factory=list, alias="assistantFacultyIds")

    @field_validator("assistant_faculty_ids")
    @classmethod
    def validate_assistant_faculty_ids(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in value:
            faculty_id = (item or "").strip()
            if not faculty_id:
                continue
            if len(faculty_id) > 36:
                raise ValueError("assistantFacultyIds values must be 36 characters or fewer")
            if faculty_id in seen:
                continue
            seen.add(faculty_id)
            cleaned.append(faculty_id)
        if len(cleaned) > 8:
            raise ValueError("assistantFacultyIds can include at most 8 faculty IDs")
        return cleaned

    @field_validator("day")
    @classmethod
    def validate_day(cls, value: str) -> str:
        day = value.strip()
        if day not in DAY_VALUES:
            raise ValueError("Invalid day value")
        return day

    @field_validator("startTime", "endTime")
    @classmethod
    def validate_time_format(cls, value: str) -> str:
        if not TIME_PATTERN.match(value):
            raise ValueError("Time must be in HH:MM 24-hour format")
        return value

    @model_validator(mode="after")
    def validate_time_order(self) -> "TimeSlotPayload":
        start = parse_time_to_minutes(self.startTime)
        end = parse_time_to_minutes(self.endTime)
        if end <= start:
            raise ValueError("End time must be after start time")
        return self


class OfficialTimetablePayload(BaseModel):
    program_id: str | None = Field(default=None, alias="programId", min_length=1, max_length=36)
    term_number: int | None = Field(default=None, alias="termNumber", ge=1, le=20)
    faculty_data: list[FacultyPayload] = Field(default_factory=list, alias="facultyData")
    course_data: list[CoursePayload] = Field(default_factory=list, alias="courseData")
    room_data: list[RoomPayload] = Field(default_factory=list, alias="roomData")
    timetable_data: list[TimeSlotPayload] = Field(default_factory=list, alias="timetableData")

    model_config = {
        "populate_by_name": True,
        "from_attributes": True,
    }

    @model_validator(mode="after")
    def validate_references(self) -> "OfficialTimetablePayload":
        course_by_id = {course.id: course for course in self.course_data}
        course_ids = set(course_by_id.keys())
        room_ids = {room.id for room in self.room_data}
        faculty_ids = {faculty.id for faculty in self.faculty_data}

        for slot in self.timetable_data:
            course = course_by_id.get(slot.courseId)
            if course is None:
                raise ValueError(f"Timeslot {slot.id} references unknown courseId {slot.courseId}")
            if slot.roomId not in room_ids:
                raise ValueError(f"Timeslot {slot.id} references unknown roomId {slot.roomId}")
            if slot.facultyId not in faculty_ids:
                raise ValueError(f"Timeslot {slot.id} references unknown facultyId {slot.facultyId}")
            for assistant_id in slot.assistant_faculty_ids:
                if assistant_id not in faculty_ids:
                    raise ValueError(
                        f"Timeslot {slot.id} references unknown assistantFacultyId {assistant_id}"
                    )
                if assistant_id == slot.facultyId:
                    raise ValueError(
                        f"Timeslot {slot.id} assistantFacultyIds cannot include primary facultyId"
                    )
            if slot.sessionType is None:
                slot.sessionType = "lab" if course.type == "lab" else "theory"
            course_practical_hours = int(course.lab_hours or 0)
            course_theory_tutorial_hours = int(course.theory_hours or 0) + int(course.tutorial_hours or 0)
            if slot.sessionType == "lab" and course_practical_hours <= 0:
                raise ValueError(
                    f"Timeslot {slot.id} uses sessionType=lab but course has no practical hours configured"
                )
            if course.type == "lab" and course_theory_tutorial_hours <= 0 and slot.sessionType != "lab":
                raise ValueError(f"Timeslot {slot.id} for lab course must use sessionType=lab")

        def ensure_unique(label: str, items: list[BaseModel]) -> None:
            seen: set[str] = set()
            duplicates: set[str] = set()
            for item in items:
                if item.id in seen:
                    duplicates.add(item.id)
                else:
                    seen.add(item.id)
            if duplicates:
                raise ValueError(f"Duplicate {label} id(s): {', '.join(sorted(duplicates))}")

        ensure_unique("faculty", self.faculty_data)
        ensure_unique("course", self.course_data)
        ensure_unique("room", self.room_data)
        ensure_unique("timeslot", self.timetable_data)

        return self


class OfflinePublishFilters(BaseModel):
    department: str | None = Field(default=None, max_length=200)
    program_id: str | None = Field(default=None, alias="programId", min_length=1, max_length=36)
    term_number: int | None = Field(default=None, alias="termNumber", ge=1, le=20)
    section_name: str | None = Field(default=None, alias="sectionName", min_length=1, max_length=50)
    faculty_id: str | None = Field(default=None, alias="facultyId", min_length=1, max_length=36)

    model_config = {
        "populate_by_name": True,
        "from_attributes": True,
    }

    @field_validator("department")
    @classmethod
    def normalize_department(cls, value: str | None) -> str | None:
        if value is None:
            return None
        trimmed = value.strip()
        return trimmed or None

    @field_validator("section_name")
    @classmethod
    def normalize_section_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        trimmed = value.strip()
        return trimmed or None


class OfflinePublishRequest(BaseModel):
    filters: OfflinePublishFilters | None = None


class OfflinePublishResponse(BaseModel):
    attempted: int
    sent: int
    skipped: int
    failed: int
    recipients: list[EmailStr] = Field(default_factory=list)
    failed_recipients: list[EmailStr] = Field(default_factory=list)
    message: str


class FacultyCourseSectionAssignment(BaseModel):
    course_id: str = Field(min_length=1, max_length=36)
    course_code: str = Field(min_length=1, max_length=50)
    course_name: str = Field(min_length=1, max_length=200)
    section: str = Field(min_length=1, max_length=50)
    batch: str | None = Field(default=None, min_length=1, max_length=50)
    day: str
    start_time: str = Field(alias="startTime")
    end_time: str = Field(alias="endTime")
    room_id: str = Field(min_length=1, max_length=36)
    room_name: str = Field(min_length=1, max_length=100)
    assignment_role: Literal["primary", "assistant"] = Field(default="primary", alias="assignmentRole")

    model_config = {
        "populate_by_name": True,
        "from_attributes": True,
    }


class FacultyCourseSectionMappingOut(BaseModel):
    faculty_id: str = Field(min_length=1, max_length=36)
    faculty_name: str = Field(min_length=1, max_length=200)
    faculty_email: EmailStr
    total_assigned_hours: float = Field(ge=0.0, le=500.0)
    assignments: list[FacultyCourseSectionAssignment] = Field(default_factory=list)
