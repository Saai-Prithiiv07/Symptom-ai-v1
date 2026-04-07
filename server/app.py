from fastapi import FastAPI
from server.symptom_env import SymptomEnvironment
from server.models import SymptomAction

app = FastAPI()

# Initialize environment
env = SymptomEnvironment()

# -------------------------------
# ROOT CHECK (for HF)
# -------------------------------
@app.get("/")
def home():
    return {"message": "Symptom AI Running 🚀"}

# -------------------------------
# RESET (IMPORTANT: MUST BE POST)
# -------------------------------
@app.post("/reset")
def reset():
    state = env.reset()
    return state

# -------------------------------
# STEP (IMPORTANT: MUST BE POST)
# -------------------------------
@app.post("/step")
def step(action: SymptomAction):
    obs, reward, done, info = env.step(action)

    return {
        "observation": obs,
        "reward": round(reward, 2),
        "done": done,
        "info": info
    }

# -------------------------------
# STATE (OPTIONAL BUT GOOD)
# -------------------------------
@app.get("/state")
def state():
    return env.state

# -------------------------------
# LEADERBOARD (OPTIONAL)
# -------------------------------
@app.get("/leaderboard")
def leaderboard():
    return {
        "top_scores": [0.92, 0.88, 0.85]
    }