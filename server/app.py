from fastapi import FastAPI
from server.symptom_env import SymptomEnvironment
from server.models import SymptomAction

app = FastAPI()
env = SymptomEnvironment()

@app.get("/")
def home():
    return {"message": "Symptom AI Running 🚀"}

@app.post("/reset")
def reset():
    return env.reset()

@app.post("/step")
def step(action: SymptomAction):
    return env.step(action.message)

# ✅ REQUIRED FOR OPENENV
def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

# ✅ VERY IMPORTANT (fixes your error)
if __name__ == "__main__":
    main()