from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.api.deps import get_db, require_roles
from app.models.user import User, UserRole
from app.schemas.conflict import ConflictReport
from app.schemas.timetable import OfficialTimetablePayload
from app.schemas.resolution import ResolveConflictRequest
from app.services.conflict_service import ConflictService
from app.models.room import Room
from app.models.faculty import Faculty

router = APIRouter()


def _prune_primary_from_slot_assistants(slot: object, primary_faculty_id: str) -> None:
    assistant_ids = []
    raw_assistants = getattr(slot, "assistantFacultyIds", None)
    if raw_assistants is None:
        raw_assistants = getattr(slot, "assistant_faculty_ids", None)
    if isinstance(raw_assistants, list):
        for item in raw_assistants:
            faculty_id = str(item or "").strip()
            if not faculty_id or faculty_id == primary_faculty_id or faculty_id in assistant_ids:
                continue
            assistant_ids.append(faculty_id)
    if hasattr(slot, "assistantFacultyIds"):
        setattr(slot, "assistantFacultyIds", assistant_ids)
    if hasattr(slot, "assistant_faculty_ids"):
        setattr(slot, "assistant_faculty_ids", assistant_ids)

@router.post("/detect", response_model=ConflictReport)
def detect_conflicts(
    payload: OfficialTimetablePayload,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
):
    # Load resources for validation
    rooms = db.query(Room).all()
    room_map = {r.id: {"id": r.id, "name": r.name, "capacity": r.capacity, "type": r.type} for r in rooms}
    
    faculty = db.query(Faculty).all()
    faculty_map = {f.id: {"id": f.id, "name": f.name} for f in faculty}
    
    service = ConflictService(payload, room_map, faculty_map)
    report = service.detect_conflicts()
    
    # Generate resolutions for each conflict
    for conflict in report.conflicts:
        resolutions = service.generate_resolutions(conflict)
        report.suggested_resolutions.extend(resolutions)
        
    return report

@router.post("/resolve", response_model=OfficialTimetablePayload)
def apply_resolution(
    request: ResolveConflictRequest,
    current_user: User = Depends(require_roles(UserRole.admin, UserRole.scheduler)),
):
    payload = request.payload
    action = request.action
    
    target_slot = None
    for slot in payload.timetableData:
        if slot.id == action.target_slot_id:
            target_slot = slot
            break
            
    if not target_slot:
        raise HTTPException(status_code=404, detail="Target slot not found")
        
    if action.action_type == "change_room":
        new_room_id = action.parameters.get("new_room_id") or action.parameters.get("roomId")
        if new_room_id:
            target_slot.roomId = new_room_id
            
    elif action.action_type == "move_slot":
        # Expect day, startTime, endTime, roomId
        if "day" in action.parameters:
            target_slot.day = action.parameters["day"]
        if "startTime" in action.parameters:
            target_slot.startTime = action.parameters["startTime"]
        if "endTime" in action.parameters:
            target_slot.endTime = action.parameters["endTime"]
        if "roomId" in action.parameters:
            target_slot.roomId = action.parameters["roomId"]
            
    elif action.action_type == "change_faculty":
        new_faculty_id = action.parameters.get("new_faculty_id") or action.parameters.get("facultyId")
        if new_faculty_id:
            target_slot.facultyId = new_faculty_id
            _prune_primary_from_slot_assistants(target_slot, new_faculty_id)
            
    elif action.action_type == "swap_slot":
        # Requires swapping heavily, usually easier structurally.
        # "other_slot_id" in params.
        other_slot_id = action.parameters.get("other_slot_id") or action.parameters.get("swapWithSlotId")
        if other_slot_id:
            other_slot = next((s for s in payload.timetableData if s.id == other_slot_id), None)
            if other_slot:
                # Swap essential timing/location
                target_slot.day, other_slot.day = other_slot.day, target_slot.day
                target_slot.startTime, other_slot.startTime = other_slot.startTime, target_slot.startTime
                target_slot.endTime, other_slot.endTime = other_slot.endTime, target_slot.endTime
                target_slot.roomId, other_slot.roomId = other_slot.roomId, target_slot.roomId

    return payload
