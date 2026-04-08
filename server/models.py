from pydantic import BaseModel, Field


class SymptomAction(BaseModel):
    """Action sent by the agent each step."""

    message: str = Field(..., description="One of: rest, medicine, doctor")


class SymptomObservation(BaseModel):
    """Observation returned by the environment."""

    task_id: str
    task_name: str
    patient_note: str
    severity: int = Field(..., ge=0, le=10)
    energy: int = Field(..., ge=0, le=10)
    immunity: int = Field(..., ge=0, le=10)
    day: int = Field(..., ge=0)
    ai_suggestion: str
    progress: float = Field(..., ge=0.0, le=1.0)
    last_action_error: str | None = None
    done: bool
    reward: float = Field(..., ge=0.0, le=1.0)
    score: float = Field(..., ge=0.0, le=1.0)


class SymptomState(BaseModel):
    """Environment state snapshot."""

    episode_id: str
    step_count: int
    task_id: str
    task_name: str
    severity: int = Field(..., ge=0, le=10)
    energy: int = Field(..., ge=0, le=10)
    immunity: int = Field(..., ge=0, le=10)
    done: bool
    total_reward: float = Field(..., ge=0.0)