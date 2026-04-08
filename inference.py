import os
import re
from typing import List, Optional

import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-120b:novita")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
TASK_NAME = os.getenv("TASK_NAME", "viral-fever")
BENCHMARK = os.getenv("BENCHMARK", "symptom-ai")
MAX_STEPS = int(os.getenv("MAX_STEPS", "8"))

SYSTEM_PROMPT = (
    "You are a triage assistant in a symptom management simulator. "
    "Choose one action each turn: rest, medicine, or doctor. "
    "Return only one token: rest OR medicine OR doctor."
)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


def safe_action(text: str) -> str:
    match = re.search(r"\b(rest|medicine|doctor)\b", text.lower())
    return match.group(1) if match else "rest"


def choose_action(client: OpenAI, observation: dict) -> str:
    note = observation.get("patient_note", "")
    severity = observation.get("severity", 0)
    day = observation.get("day", 0)
    ai_suggestion = observation.get("ai_suggestion", "rest")
    user_prompt = (
        f"Patient note: {note}\n"
        f"Severity: {severity}\n"
        f"Day: {day}\n"
        f"Suggested action: {ai_suggestion}\n"
        "Pick the next action."
    )
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=8,
        )
        content = completion.choices[0].message.content or "rest"
        return safe_action(content)
    except Exception:
        return safe_action(ai_suggestion)


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
    try:
        reset_resp = requests.post(f"{ENV_BASE_URL}/reset", params={"task_id": TASK_NAME}, timeout=30)
        reset_resp.raise_for_status()
        payload = reset_resp.json()
        observation = payload.get("observation", {})
        done = bool(payload.get("done", False))

        for step in range(1, MAX_STEPS + 1):
            if done:
                break
            action = choose_action(client, observation)
            error: Optional[str] = None
            try:
                step_resp = requests.post(f"{ENV_BASE_URL}/step", json={"message": action}, timeout=30)
                step_resp.raise_for_status()
                step_payload = step_resp.json()
                observation = step_payload.get("observation", {})
                reward = float(step_payload.get("reward", 0.0))
                done = bool(step_payload.get("done", False))
                error = observation.get("last_action_error")
            except Exception as exc:
                reward = 0.0
                done = True
                error = str(exc)

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action, reward=reward, done=done, error=error)

        score = float(observation.get("score", 0.0)) if isinstance(observation, dict) else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= 0.5
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()