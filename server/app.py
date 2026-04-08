from fastapi import FastAPI
from server.symptom_env import SymptomEnvironment
from server.models import SymptomAction

app = FastAPI()
env = SymptomEnvironment()

@app.get("/")
def home():
    return {"message": "Symptom AI Running", "status": "ok"}

@app.post("/reset")
def reset(task_id: str | None = None):
    obs = env.reset(task_id=task_id)
    return {
        "observation": obs.model_dump(),
        "reward": round(obs.reward, 2),
        "done": obs.done,
        "info": {"task_id": obs.task_id},
    }

@app.post("/step")
def step(action: SymptomAction):
    obs, reward, done, info = env.step(action.message)
    return {
        "observation": obs.model_dump(),
        "reward": round(reward, 2),
        "done": done,
        "info": info,
    }

@app.get("/state")
def state():
    return env.state.model_dump()

@app.get("/tasks")
def tasks():
    return {"tasks": env.list_tasks()}

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()