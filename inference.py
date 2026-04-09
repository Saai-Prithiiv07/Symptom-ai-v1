import os
import requests

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "custom-model")

TASK_NAME = "symptom-simulation"
BENCHMARK = "medical-env"

MAX_STEPS = 7
SUCCESS_SCORE_THRESHOLD = 0.6


# =========================
# LOG FUNCTIONS
# =========================
def log_start():
    print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}", flush=True)


def log_step(step, action, reward, done, error=None):
    error_val = "null" if error is None else error
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success, steps, score, rewards):
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# =========================
# POLICY
# =========================
def choose_action(state):
    s = state.get("severity", 0)

    if s >= 5:
        return "doctor"
    elif s >= 2:
        return "medicine"
    else:
        return "rest"


# =========================
# MAIN LOOP
# =========================
def main():
    log_start()

    rewards = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        state = requests.get(f"{API_BASE_URL}/reset").json()

        for step in range(1, MAX_STEPS + 1):

            action = choose_action(state)

            try:
                res = requests.post(
                    f"{API_BASE_URL}/step",
                    json={"message": action}
                )

                data = res.json()

                reward = float(data.get("reward", 0.0))
                done = data.get("done", False)

                rewards.append(reward)
                steps_taken = step

                log_step(step, action, reward, done)

                state = data

                if done:
                    break

            except Exception as e:
                log_step(step, action, 0.0, False, str(e))
                break

        total_reward = sum(rewards)
        max_possible = steps_taken * 3.0 if steps_taken > 0 else 1

        score = total_reward / max_possible
        score = max(0.0, min(1.0, score))

        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        log_step(0, "error", 0.0, True, str(e))

    finally:
        log_end(success, steps_taken, score, rewards)


if __name__ == "__main__":
    main()
