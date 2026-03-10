from app.schemas.conflict import ResolutionAction
from app.schemas.timetable import OfficialTimetablePayload
from pydantic import BaseModel

class ResolveConflictRequest(BaseModel):
    payload: OfficialTimetablePayload
    action: ResolutionAction
