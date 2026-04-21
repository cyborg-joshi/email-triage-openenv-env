from pydantic import BaseModel
from typing import List, Dict, Optional, Any


class WorldState(BaseModel):
    emails: List[str]
    calendar: Dict[str, str]
    pending_tasks: List[str]
    schema_version: str
    schema_name: str
    context: str


class ExecutiveAction(BaseModel):
    action: str
    reply: Optional[str] = ""
    step: int = 1


class ExecutiveObservation(BaseModel):
    world_state: Dict[str, Any]
    last_action: str
    message: str
    step_number: int
