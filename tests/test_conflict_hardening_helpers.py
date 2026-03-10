from app.api.routes.generator import _build_reserved_slots_from_payload, _cross_term_resource_overlap_count
from app.api.routes.timetable import _resource_placement_conflicts
from app.schemas.generator import GeneratedCycleSolutionTerm
from app.schemas.timetable import OfficialTimetablePayload, parse_time_to_minutes


def _make_payload(*, assistants_for_slot_one: list[str] | None = None) -> OfficialTimetablePayload:
    return OfficialTimetablePayload.model_validate(
        {
            "programId": "prog-1",
            "termNumber": 2,
            "facultyData": [
                {
                    "id": "fac-1",
                    "name": "Faculty One",
                    "department": "CSE",
                    "workloadHours": 16,
                    "maxHours": 24,
                    "availability": ["Monday"],
                    "email": "faculty.one@example.edu",
                },
                {
                    "id": "fac-2",
                    "name": "Faculty Two",
                    "department": "CSE",
                    "workloadHours": 16,
                    "maxHours": 24,
                    "availability": ["Monday"],
                    "email": "faculty.two@example.edu",
                },
                {
                    "id": "fac-3",
                    "name": "Faculty Three",
                    "department": "CSE",
                    "workloadHours": 16,
                    "maxHours": 24,
                    "availability": ["Monday"],
                    "email": "faculty.three@example.edu",
                },
            ],
            "courseData": [
                {
                    "id": "course-1",
                    "code": "CSE101",
                    "name": "Course One",
                    "type": "theory",
                    "credits": 1,
                    "facultyId": "fac-1",
                    "duration": 1,
                    "hoursPerWeek": 1,
                    "semesterNumber": 2,
                    "theoryHours": 1,
                    "tutorialHours": 0,
                    "labHours": 0,
                    "assignFaculty": True,
                    "assignClassroom": True,
                },
                {
                    "id": "course-2",
                    "code": "CSE102",
                    "name": "Course Two",
                    "type": "theory",
                    "credits": 1,
                    "facultyId": "fac-2",
                    "duration": 1,
                    "hoursPerWeek": 1,
                    "semesterNumber": 2,
                    "theoryHours": 1,
                    "tutorialHours": 0,
                    "labHours": 0,
                    "assignFaculty": True,
                    "assignClassroom": True,
                },
            ],
            "roomData": [
                {"id": "room-1", "name": "R1", "capacity": 60, "type": "lecture", "building": "AB1"},
                {"id": "room-2", "name": "R2", "capacity": 60, "type": "lecture", "building": "AB1"},
            ],
            "timetableData": [
                {
                    "id": "slot-1",
                    "day": "Monday",
                    "startTime": "09:00",
                    "endTime": "10:00",
                    "courseId": "course-1",
                    "roomId": "room-1",
                    "facultyId": "fac-1",
                    "assistantFacultyIds": assistants_for_slot_one or [],
                    "section": "A",
                    "sessionType": "theory",
                },
                {
                    "id": "slot-2",
                    "day": "Monday",
                    "startTime": "09:00",
                    "endTime": "10:00",
                    "courseId": "course-2",
                    "roomId": "room-2",
                    "facultyId": "fac-2",
                    "section": "B",
                    "sessionType": "theory",
                },
            ],
        }
    )


def test_build_reserved_slots_from_payload_includes_assistant_faculty() -> None:
    payload = _make_payload(assistants_for_slot_one=["fac-2", "fac-2"])
    reserved = _build_reserved_slots_from_payload(payload)

    primary_entries = [item for item in reserved if item["faculty_id"] == "fac-1"]
    assistant_entries = [item for item in reserved if item["faculty_id"] == "fac-2" and item["room_id"] is None]

    assert primary_entries
    assert len(assistant_entries) == 1
    assert assistant_entries[0]["day"] == "Monday"
    assert assistant_entries[0]["start_time"] == "09:00"
    assert assistant_entries[0]["end_time"] == "10:00"


def test_cross_term_overlap_count_includes_assistant_faculty_usage() -> None:
    term_one_payload = _make_payload(assistants_for_slot_one=["fac-2"])
    term_two_payload = OfficialTimetablePayload.model_validate(
        {
            **term_one_payload.model_dump(by_alias=True),
            "termNumber": 4,
            "timetableData": [
                {
                    "id": "slot-3",
                    "day": "Monday",
                    "startTime": "09:00",
                    "endTime": "10:00",
                    "courseId": "course-1",
                    "roomId": "room-2",
                    "facultyId": "fac-3",
                    "assistantFacultyIds": ["fac-2"],
                    "section": "C",
                    "sessionType": "theory",
                }
            ],
        }
    )

    terms = [
        GeneratedCycleSolutionTerm(
            term_number=2,
            alternative_rank=1,
            fitness=1.0,
            hard_conflicts=0,
            soft_penalty=0.0,
            payload=term_one_payload,
        ),
        GeneratedCycleSolutionTerm(
            term_number=4,
            alternative_rank=1,
            fitness=1.0,
            hard_conflicts=0,
            soft_penalty=0.0,
            payload=term_two_payload,
        ),
    ]

    overlap_count = _cross_term_resource_overlap_count(terms)
    assert overlap_count >= 1


def test_resource_placement_conflicts_detect_assistant_overlap() -> None:
    payload = _make_payload(assistants_for_slot_one=["fac-2"])
    course_map = {course.id: course for course in payload.course_data}

    has_conflict = _resource_placement_conflicts(
        payload=payload,
        slot_id="slot-1",
        course_id="course-1",
        section="A",
        batch=None,
        day="Monday",
        start=parse_time_to_minutes("09:00"),
        end=parse_time_to_minutes("10:00"),
        room_id="room-1",
        faculty_id="fac-1",
        moving_assistant_ids=("fac-2",),
        course_map=course_map,
        elective_pairs=set(),
    )

    assert has_conflict is True
