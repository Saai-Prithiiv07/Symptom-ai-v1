from pydantic import BaseModel


class SymptomAction(BaseModel):
    message: str


class SymptomObservation(BaseModel):
    message: str
    severity: int
    energy: int
    immunity: int
    day: int
    ai_suggestion: str
    done: bool
    reward: float