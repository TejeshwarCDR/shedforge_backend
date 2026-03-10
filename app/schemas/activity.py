from datetime import datetime

from pydantic import BaseModel


class ActivityLogOut(BaseModel):
    id: str
    user_id: str | None
    action: str
    entity_type: str | None
    entity_id: str | None
    details: dict
    created_at: datetime

    model_config = {"from_attributes": True}
