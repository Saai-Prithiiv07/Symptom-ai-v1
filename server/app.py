from fastapi import FastAPI
from server.symptom_env import SymptomEnvironment
from server.database import add_score, get_leaderboard
from server.auth import register_user, login_user
from server.models import SymptomAction
from fastapi.responses import HTMLResponse

app = FastAPI()
env = SymptomEnvironment()

current_user = None

@app.get("/")
def home():
    return {"message": "Symptom AI Running 🚀"}

@app.post("/register")
def register(data: dict):
    return register_user(data["username"], data["password"])

@app.post("/login")
def login(data: dict):
    global current_user
    res = login_user(data["username"], data["password"])
    if res["status"] == "success":
        current_user = data["username"]
    return res

@app.get("/reset")
def reset():
    return env.reset()

@app.post("/step")
def step(action: SymptomAction):
    result = env.step(action.message)

    if result["done"] and current_user:
        add_score(current_user, result["reward"])

    return result

@app.get("/leaderboard")
def leaderboard():
    return get_leaderboard()

@app.get("/ui", response_class=HTMLResponse)
def ui():
    with open("index.html") as f:
        return f.read()