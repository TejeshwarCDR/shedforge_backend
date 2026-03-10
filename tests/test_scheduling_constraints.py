import pytest
from app.services.conflict_service import ConflictService
from app.schemas.timetable import OfficialTimetablePayload, TimeSlotPayload

@pytest.fixture
def resource_data():
    return {
        "rooms": {
            "r1": {"id": "r1", "name": "Room A102", "capacity": 100, "type": "lecture"},
            "r2": {"id": "r2", "name": "Lab L1", "capacity": 30, "type": "lab"}
        },
        "faculty": {
            "f1": {"id": "f1", "name": "Prof X"},
            "f2": {"id": "f2", "name": "Prof Y"}
        }
    }

def test_room_overlap_prevention(resource_data):
    """Verify that ConflictService detects room overlaps correctly."""
    payload = OfficialTimetablePayload(
        versionId="test_v1",
        facultyData=[
            {"id": "f1", "name": "Prof X", "department": "D1", "workloadHours": 0, "maxHours": 20, "email": "x@e.com"},
            {"id": "f2", "name": "Prof Y", "department": "D1", "workloadHours": 0, "maxHours": 20, "email": "y@e.com"},
        ],
        courseData=[
            {"id": "c1", "code": "C1", "name": "Course 1", "type": "theory", "credits": 3, "facultyId": "f1", "duration": 1, "hoursPerWeek": 3, "theoryHours": 3, "labHours": 0, "tutorialHours": 0},
            {"id": "c2", "code": "C2", "name": "Course 2", "type": "theory", "credits": 3, "facultyId": "f2", "duration": 1, "hoursPerWeek": 3, "theoryHours": 3, "labHours": 0, "tutorialHours": 0},
        ],
        roomData=[
            {"id": "r1", "name": "Room A102", "capacity": 100, "type": "lecture", "building": "B1"},
        ],
        timetableData=[
            TimeSlotPayload(
                id="s1", day="Monday", startTime="09:00", endTime="10:00",
                courseId="c1", roomId="r1", facultyId="f1", section="F",
                studentCount=50, sessionType="theory"
            ),
            TimeSlotPayload(
                id="s2", day="Monday", startTime="09:30", endTime="10:30", # Overlap
                courseId="c2", roomId="r1", facultyId="f2", section="G",
                studentCount=50, sessionType="theory"
            )
        ]
    )
    service = ConflictService(payload, resource_data["rooms"], resource_data["faculty"])
    report = service.detect_conflicts()
    
    room_conflicts = [c for c in report.conflicts if c.conflict_type == "room_conflict"]
    assert len(room_conflicts) > 0
    assert "Room A102" in room_conflicts[0].description

def test_section_overlap_prevention(resource_data):
    """Verify that ConflictService detects section overlaps correctly."""
    payload = OfficialTimetablePayload(
        versionId="test_v2",
        facultyData=[
            {"id": "f1", "name": "Prof X", "department": "D1", "workloadHours": 0, "maxHours": 20, "email": "x@e.com"},
            {"id": "f2", "name": "Prof Y", "department": "D1", "workloadHours": 0, "maxHours": 20, "email": "y@e.com"},
        ],
        courseData=[
            {"id": "c1", "code": "C1", "name": "Course 1", "type": "theory", "credits": 3, "facultyId": "f1", "duration": 1, "hoursPerWeek": 3, "theoryHours": 3, "labHours": 0, "tutorialHours": 0},
            {"id": "c2", "code": "C2", "name": "Course 2", "type": "theory", "credits": 3, "facultyId": "f2", "duration": 1, "hoursPerWeek": 3, "theoryHours": 3, "labHours": 0, "tutorialHours": 0},
        ],
        roomData=[
            {"id": "r1", "name": "Room A102", "capacity": 100, "type": "lecture", "building": "B1"},
            {"id": "r2", "name": "Lab L1", "capacity": 30, "type": "lab", "building": "B1"},
        ],
        timetableData=[
            TimeSlotPayload(
                id="s1", day="Friday", startTime="14:00", endTime="15:00",
                courseId="c1", roomId="r1", facultyId="f1", section="F",
                studentCount=40, sessionType="theory"
            ),
            TimeSlotPayload(
                id="s2", day="Friday", startTime="14:30", endTime="15:30", # Overlap
                courseId="c2", roomId="r2", facultyId="f2", section="F",
                studentCount=40, sessionType="theory"
            )
        ]
    )
    service = ConflictService(payload, resource_data["rooms"], resource_data["faculty"])
    report = service.detect_conflicts()
    
    section_conflicts = [c for c in report.conflicts if c.conflict_type == "section_conflict"]
    assert len(section_conflicts) > 0
    # Description currently is "Section overlap: c1 and c2"
    assert "Section overlap" in section_conflicts[0].description

def test_room_type_consistency(resource_data):
    """Verify that theory courses aren't scheduled in labs and vice versa."""
    payload = OfficialTimetablePayload(
        versionId="test_v3",
        facultyData=[
            {"id": "f1", "name": "Prof X", "department": "D1", "workloadHours": 0, "maxHours": 20, "email": "x@e.com"},
        ],
        courseData=[
            {"id": "c1", "code": "C1", "name": "Course 1", "type": "theory", "credits": 3, "facultyId": "f1", "duration": 1, "hoursPerWeek": 3, "theoryHours": 3, "labHours": 0, "tutorialHours": 0},
        ],
        roomData=[
            {"id": "r2", "name": "Lab L1", "capacity": 30, "type": "lab", "building": "B1"},
        ],
        timetableData=[
            TimeSlotPayload(
                id="s1", day="Monday", startTime="11:00", endTime="12:00",
                courseId="c1", roomId="r2", facultyId="f1", section="A",
                studentCount=20, sessionType="theory" # Scheduled in Lab r2
            )
        ]
    )
    service = ConflictService(payload, resource_data["rooms"], resource_data["faculty"])
    report = service.detect_conflicts()
    
    type_conflicts = [c for c in report.conflicts if c.conflict_type == "room_type"]
    assert len(type_conflicts) > 0
    # Description currently is "Theory session in lab room Lab L1"
    assert "session in" in type_conflicts[0].description
