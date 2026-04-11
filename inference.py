import os
import requests
from openai import OpenAI

# Required Hackathon Variables
LLM_API_BASE = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# Environment Variables
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://127.0.0.1:7860")
BENCHMARK = "symptom-ai"
MAX_STEPS = 8
SUCCESS_SCORE_THRESHOLD = 0.5

client = OpenAI(
    api_key=HF_TOKEN,
    base_url=LLM_API_BASE,
)

# =========================
# LOG FUNCTIONS
# =========================
def log_start(task_name):
    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)

def log_step(step, action, reward, done, error=None):
    error_val = "null" if error is None else error
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )

def log_end(success, steps, score, rewards):
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )

# =========================
# POLICY
# =========================
def choose_action(state):
    prompt = f"""You are an AI performing medical triage.
    Given the patient's symptoms, choose the most appropriate action: 'rest', 'medicine', or 'doctor'.
    Current State:
    Severity: {state.get('severity')}
    Energy: {state.get('energy')}
    Immunity: {state.get('immunity')}
    Patient Note: {state.get('patient_note')}
    Day: {state.get('day')}
    
    If severity >= 7, recommend 'doctor'.
    If severity >= 4, recommend 'medicine'.
    Otherwise, recommend 'rest'.
    
    Return exclusively the single word of the chosen action in lowercase."""
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0,
        )
        action = response.choices[0].message.content.strip().lower()
        if "doctor" in action: return "doctor"
        if "medicine" in action: return "medicine"
        return "rest"
    except Exception as e:
        print(f"LLM Call Error: {str(e)}", flush=True)
        return "rest"

# =========================
# MAIN LOOP
# =========================
def run_task(task_id, max_steps):
    log_start(task_id)

    rewards = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        res = requests.post(f"{ENV_BASE_URL}/reset?task_id={task_id}")
        res.raise_for_status()
        data = res.json()
        state = data["observation"]

        for step in range(1, max_steps + 1):
            action = choose_action(state)

            try:
                res = requests.post(
                    f"{ENV_BASE_URL}/step",
                    json={"message": action}
                )
                res.raise_for_status()
                data = res.json()

                reward = float(data.get("reward", 0.0))
                done = data.get("done", False)

                rewards.append(reward)
                steps_taken = step

                log_step(step, action, reward, done)

                state = data["observation"]

                if done:
                    # Final score provided by environment overrides raw average
                    score = state.get("score", 0.0)
                    break

            except Exception as e:
                log_step(step, action, 0.0, False, str(e))
                break

        if not done:
            score = state.get("score", 0.0)

        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        log_step(0, "error", 0.0, True, str(e))

    finally:
        log_end(success, steps_taken, score, rewards)

def main():
    try:
        res = requests.get(f"{ENV_BASE_URL}/tasks")
        res.raise_for_status()
        tasks = res.json().get("tasks", [])
    except Exception as e:
        print(f"Failed to fetch tasks: {e}. Falling back to default tasks.", flush=True)
        tasks = [
            {"id": "seasonal-cold", "max_steps": 6},
            {"id": "viral-fever", "max_steps": 7},
            {"id": "red-flag-breathing", "max_steps": 8}
        ]
        
    for t in tasks:
        run_task(t["id"], t.get("max_steps", MAX_STEPS))

if __name__ == "__main__":
    main()