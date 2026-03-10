from pydantic import BaseModel
from typing import Literal, Optional, List

class ConflictDetail(BaseModel):
    id: str
    conflict_type: Literal[
        "room_conflict", 
        "faculty_conflict", 
        "section_conflict", 
        "room_capacity", 
        "room_type",
        "faculty_availability",
        "workload_overflow",
        "elective_mismatch"
    ]
    description: str
    severity: Literal["hard", "soft"]
    affected_slots: List[str]  # List of timetable slot IDs involved
    
class ResolutionAction(BaseModel):
    action_type: Literal["move_slot", "swap_slot", "change_room", "change_faculty"]
    description: str
    target_slot_id: str
    parameters: dict  # e.g. {"new_room_id": "r1"}

class ConflictReport(BaseModel):
    conflicts: List[ConflictDetail]
    suggested_resolutions: List[ResolutionAction]
