import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from app.models.user import User, UserRole

# Mock bootstrap to prevent DB connection during startup
with patch("app.db.bootstrap.ensure_runtime_schema_compatibility"):
    from app.main import app

# Mock DB Session
def override_get_db():
    try:
        db = MagicMock()
        # Mock query results for conflicts endpoint
        # rooms = db.query(Room).all()
        # faculty = db.query(Faculty).all()
        
        mock_room = MagicMock()
        mock_room.id = "r1"
        mock_room.name = "Room 1"
        mock_room.capacity = 100
        mock_room.type = "lecture"
        
        mock_faculty = MagicMock()
        mock_faculty.id = "f1"
        mock_faculty.name = "Prof A"
        
        db.query.return_value.all.side_effect = [[mock_room], [mock_faculty]]
        yield db
    finally:
        pass

from app.api.deps import get_db, get_current_user, security

# Mock Auth
def override_get_current_user():
    return User(id="u1", email="admin@example.com", role=UserRole.admin, is_active=True)

def override_security():
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials="fake_token")

def test_health_check(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_detect_conflicts_valid_payload(client):
    # Override auth inside the test to avoid interference
    from app.main import app
    from app.api.deps import get_db, get_current_user, security
    
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_user] = override_get_current_user
    app.dependency_overrides[security] = override_security
    
    try:
        # We define payload
        payload = {
            "versionId": "v1",
            "facultyData": [{"id": "f1", "name": "Prof A", "department": "D", "workloadHours": 0, "maxHours": 10, "email": "a@e.com"}],
            "courseData": [{"id": "c1", "code": "C1", "name": "C1", "type": "theory", "credits": 3, "facultyId": "f1", "duration": 1, "hoursPerWeek": 3, "theoryHours": 3, "labHours": 0, "tutorialHours": 0}],
            "roomData": [{"id": "r1", "name": "Room 1", "capacity": 100, "type": "lecture", "building": "B"}],
            "timetableData": [
                {
                    "id": "s1", "day": "Monday", "startTime": "09:00", "endTime": "10:00",
                    "courseId": "c1", "roomId": "r1", "facultyId": "f1", "section": "A", "sessionType": "theory",
                    "studentCount": 50
                }
            ]
        }
        
        response = client.post("/api/conflicts/detect", json=payload)
        
        if response.status_code in (401, 403):
            pytest.fail(f"Auth failed: {response.text}")
            
        assert response.status_code == 200
        data = response.json()
        assert "conflicts" in data
        assert len(data["conflicts"]) == 0
    finally:
        # Clear specific overrides to be safe
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(security, None)

