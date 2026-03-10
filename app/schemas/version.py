from datetime import datetime

from pydantic import BaseModel


class TimetableVersionOut(BaseModel):
    id: str
    label: str
    summary: dict
    created_by_id: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class TimetableVersionCompare(BaseModel):
    from_version_id: str
    to_version_id: str
    added_slots: int
    removed_slots: int
    changed_slots: int
    from_label: str
    to_label: str


class TimetableTrendPoint(BaseModel):
    version_id: str
    label: str
    created_at: datetime
    constraint_satisfaction: float
    conflicts_detected: int
