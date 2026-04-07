from pydantic import BaseModel

# -------------------------------
# ACTION (input to /step)
# -------------------------------
class SymptomAction(BaseModel):
    action: str   # doctor / medicine / rest

# -------------------------------
# OBSERVATION (output from env)
# -------------------------------
class SymptomObservation(BaseModel):
    message: str
    severity: int
    energy: int
    immunity: int
    day: int
    ai_suggestion: str
    done: bool
    reward: float