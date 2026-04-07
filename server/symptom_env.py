import uuid
from server.models import SymptomObservation

class SymptomEnvironment:

    def __init__(self):
        self.reset()

    # -------------------------------
    # AI DECISION LOGIC
    # -------------------------------
    def ai_doctor(self):
        s = self.state["severity"]
        e = self.state["energy"]
        i = self.state["immunity"]

        if s >= 7:
            return "doctor"
        elif s >= 5:
            return "medicine"
        elif e < 4:
            return "rest"
        elif i < 4:
            return "medicine"
        else:
            return "rest"

    # -------------------------------
    # EXPLAIN ACTION
    # -------------------------------
    def explain_action(self, action):
        if action == "doctor":
            return "Condition critical - doctor needed immediately."
        elif action == "medicine":
            return "Moderate symptoms - medicine helps recovery."
        else:
            return "Low severity - rest improves energy."

    # -------------------------------
    # RESET ENVIRONMENT
    # -------------------------------
    def reset(self):
        self.state = {
            "episode_id": str(uuid.uuid4()),
            "severity": 5,
            "energy": 5,
            "immunity": 5,
            "day": 0
        }
        return self.state

    # -------------------------------
    # STEP FUNCTION (MAIN LOGIC)
    # -------------------------------
    def step(self, action_obj):
        action = action_obj.action

        self.state["day"] += 1

        # Apply action effects
        if action == "doctor":
            self.state["severity"] -= 3
            self.state["energy"] -= 1

        elif action == "medicine":
            self.state["severity"] -= 2
            self.state["immunity"] += 1

        elif action == "rest":
            self.state["energy"] += 2

        # Clamp values
        self.state["severity"] = max(0, min(10, self.state["severity"]))
        self.state["energy"] = max(0, min(10, self.state["energy"]))
        self.state["immunity"] = max(0, min(10, self.state["immunity"]))

        # AI suggestion
        ai_suggestion = self.ai_doctor()

        # Reward logic
        reward = 1.0 if action == ai_suggestion else 0.2

        # Done condition
        done = self.state["severity"] == 0 or self.state["day"] >= 7

        # Create observation
        observation = {
            "message": self.explain_action(action),
            "severity": self.state["severity"],
            "energy": self.state["energy"],
            "immunity": self.state["immunity"],
            "day": self.state["day"],
            "ai_suggestion": ai_suggestion,
            "done": done,
            "reward": reward
        }

        return observation, reward, done, {}