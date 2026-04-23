from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from openenv.core import Observation, Action, State


class ExecutiveObservation(Observation):
    world_state: Dict[str, Any] = Field(default_factory=dict)
    last_action: str = ""
    message: str = ""
    step_number: int = 1


class ExecutiveAction(Action):
    action: str
    reply: Optional[str] = ""
    step: int = 1


class ExecutiveState(State):
    episode_count: int = 0
    current_schema: str = "v1"
    schema_name: str = "Corporate Mode"
    current_task: Optional[str] = None
    step_number: int = 1
    is_done: bool = False
