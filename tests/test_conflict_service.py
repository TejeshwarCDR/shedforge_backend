import pytest
from app.services.conflict_service import ConflictService
from app.schemas.timetable import OfficialTimetablePayload, TimeSlotPayload, FacultyPayload, CoursePayload, RoomPayload

@pytest.fixture
def sample_payload():
    return OfficialTimetablePayload(
        versionId="v1",
        facultyData=[
            {"id": "f1", "name": "Prof A", "department": "Dept", "workloadHours": 0, "maxHours": 10, "email": "a@example.com"},
            {"id": "f2", "name": "Prof B", "department": "Dept", "workloadHours": 0, "maxHours": 10, "email": "b@example.com"},
        ],
        courseData=[
            {"id": "c1", "code": "C1", "name": "Course 1", "type": "theory", "credits": 3, "facultyId": "f1", "duration": 1, "hoursPerWeek": 3, "theoryHours": 3, "labHours": 0, "tutorialHours": 0},
            {"id": "c2", "code": "C2", "name": "Course 2", "type": "theory", "credits": 3, "facultyId": "f2", "duration": 1, "hoursPerWeek": 3, "theoryHours": 3, "labHours": 0, "tutorialHours": 0},
        ],
        roomData=[
            {"id": "r1", "name": "Room 1", "capacity": 100, "type": "lecture", "building": "B1"},
        ],
        timetableData=[
            TimeSlotPayload(
                id="s1", day="Monday", startTime="09:00", endTime="10:00",
                courseId="c1", roomId="r1", facultyId="f1", section="A",
                studentCount=50, sessionType="theory"
            ),
            TimeSlotPayload(
                id="s2", day="Monday", startTime="09:00", endTime="10:00",
                courseId="c2", roomId="r1", facultyId="f2", section="B",
                studentCount=50, sessionType="theory"
            )
        ]
    )

@pytest.fixture
def mock_resources():
    return {
        "rooms": {"r1": {"id": "r1", "name": "Room 1", "capacity": 100, "type": "lecture"}},
        "faculty": {"f1": {"id": "f1", "name": "Prof A"}, "f2": {"id": "f2", "name": "Prof B"}}
    }

def test_detect_room_conflict(sample_payload, mock_resources):
    service = ConflictService(sample_payload, mock_resources["rooms"], mock_resources["faculty"])
    report = service.detect_conflicts()
    
    assert len(report.conflicts) > 0
    conflict = report.conflicts[0]
    assert conflict.conflict_type == "room_conflict"
    assert "Room overlap in Room 1" in conflict.description
    assert conflict.conflict_type == "room_conflict"
    assert "Room overlap in Room 1" in conflict.description
    assert set(conflict.affected_slots) == {"s1", "s2"}

def test_detect_faculty_conflict(sample_payload, mock_resources):
    # Modify payload to create faculty conflict: same faculty, same time, diff rooms
    # s1: f1, r1, 09:00
    # s2: f1, r2, 09:00 (Need to change s2 in sample payload)
    
    # We create a specific payload for this test
    payload = OfficialTimetablePayload(
        versionId="v1",
        facultyData=[
            {"id": "f1", "name": "Prof A", "department": "Dept", "workloadHours": 0, "maxHours": 10, "email": "a@example.com"},
        ],
        courseData=[
            {"id": "c1", "code": "C1", "name": "Course 1", "type": "theory", "credits": 3, "facultyId": "f1", "duration": 1, "hoursPerWeek": 3, "theoryHours": 3, "labHours": 0, "tutorialHours": 0},
            {"id": "c2", "code": "C2", "name": "Course 2", "type": "theory", "credits": 3, "facultyId": "f1", "duration": 1, "hoursPerWeek": 3, "theoryHours": 3, "labHours": 0, "tutorialHours": 0},
        ],
        roomData=[
            {"id": "r1", "name": "Room 1", "capacity": 100, "type": "lecture", "building": "B1"},
            {"id": "r2", "name": "Room 2", "capacity": 100, "type": "lecture", "building": "B1"},
        ],
        timetableData=[
            TimeSlotPayload(
                id="s1", day="Monday", startTime="09:00", endTime="10:00",
                courseId="c1", roomId="r1", facultyId="f1", section="A", studentCount=50, sessionType="theory"
            ),
            TimeSlotPayload(
                id="s2", day="Monday", startTime="09:00", endTime="10:00",
                courseId="c2", roomId="r2", facultyId="f1", section="B", studentCount=50, sessionType="theory"
            )
        ]
    )
    resources = {
        "rooms": {"r1": {"name": "Room 1", "capacity": 100}, "r2": {"name": "Room 2", "capacity": 100}},
        "faculty": {"f1": {"name": "Prof A"}}
    }
    service = ConflictService(payload, resources["rooms"], resources["faculty"])
    report = service.detect_conflicts()
    
    assert len(report.conflicts) > 0
    conflict = report.conflicts[0]
    assert conflict.conflict_type == "faculty_conflict"
    assert "Faculty overlap for Prof A" in conflict.description

def test_detect_capacity_conflict(mock_resources):
    payload = OfficialTimetablePayload(
        versionId="v1",
        facultyData=[{"id": "f1", "name": "Prof A", "department": "Dept", "workloadHours": 0, "maxHours": 10, "email": "a@example.com"}],
        courseData=[{"id": "c3", "code": "C3", "name": "Course 3", "type": "theory", "credits": 3, "facultyId": "f1", "duration": 1, "hoursPerWeek": 3, "theoryHours": 3, "labHours": 0, "tutorialHours": 0}],
        roomData=[{"id": "r1", "name": "Room 1", "capacity": 100, "type": "lecture", "building": "B1"}],
        timetableData=[
            TimeSlotPayload(
                id="s3", day="Tuesday", startTime="10:00", endTime="11:00",
                courseId="c3", roomId="r1", facultyId="f1", section="A",
                studentCount=150, sessionType="theory" # Exceeds capacity 100
            )
        ]
    )
    service = ConflictService(payload, mock_resources["rooms"], mock_resources["faculty"])
    report = service.detect_conflicts()
    
    assert len(report.conflicts) == 1
    assert report.conflicts[0].conflict_type == "room_capacity"

def test_no_conflicts(mock_resources):
    payload = OfficialTimetablePayload(
        versionId="v1",
        facultyData=[{"id": "f1", "name": "Prof A", "department": "Dept", "workloadHours": 0, "maxHours": 10, "email": "a@example.com"}],
        courseData=[
            {"id": "c1", "code": "C1", "name": "Course 1", "type": "theory", "credits": 3, "facultyId": "f1", "duration": 1, "hoursPerWeek": 3, "theoryHours": 3, "labHours": 0, "tutorialHours": 0},
            {"id": "c2", "code": "C2", "name": "Course 2", "type": "theory", "credits": 3, "facultyId": "f1", "duration": 1, "hoursPerWeek": 3, "theoryHours": 3, "labHours": 0, "tutorialHours": 0},
        ],
        roomData=[{"id": "r1", "name": "Room 1", "capacity": 100, "type": "lecture", "building": "B1"}],
        timetableData=[
            TimeSlotPayload(
                id="s1", day="Monday", startTime="09:00", endTime="10:00",
                courseId="c1", roomId="r1", facultyId="f1", section="A",
                studentCount=50, sessionType="theory"
            ),
            TimeSlotPayload(
                id="s2", day="Monday", startTime="10:00", endTime="11:00", # Different time
                courseId="c2", roomId="r1", facultyId="f1", section="A",
                studentCount=50, sessionType="theory"
            )
        ]
    )
    service = ConflictService(payload, mock_resources["rooms"], mock_resources["faculty"])
    report = service.detect_conflicts()
    
    assert len(report.conflicts) == 0
