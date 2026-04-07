import uuid
from server.models import SymptomAction, SymptomObservation
class SymptomEnvironment:

    def __init__(self):
        self.reset()

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

    def explain_action(self, action):
        if action == "doctor":
            return "Condition critical — doctor needed immediately."
        elif action == "medicine":
            return "Moderate symptoms — medicine helps recovery."
        else:
            return "Low severity — rest improves energy."

    def reset(self):
        self.state = {
            "episode_id": str(uuid.uuid4()),
            "severity": 5,
            "energy": 5,
            "immunity": 5,
            "day": 0
        }
        return self.get_obs(False, 0)

    def step(self, action):
        self.state["day"] += 1

        if action == "medicine":
            self.state["severity"] -= 2
        elif action == "rest":
            self.state["energy"] += 1
        elif action == "doctor":
            self.state["severity"] -= 3
            self.state["immunity"] += 1

        # clamp
        self.state["severity"] = max(0, self.state["severity"])
        self.state["energy"] = min(10, self.state["energy"])
        self.state["immunity"] = min(10, self.state["immunity"])

        done = self.state["severity"] == 0 or self.state["day"] >= 7

        if self.state["severity"] == 0:
            reward = 3.0
        elif self.state["severity"] < 3:
            reward = 2.0
        elif action == "doctor":
            reward = 2.5
        else:
            reward = 1.0

        return self.get_obs(done, reward, action)

    def get_obs(self, done, reward, action=None):
        return {
            "message": "Simulation running",
            "severity": self.state["severity"],
            "energy": self.state["energy"],
            "immunity": self.state["immunity"],
            "day": self.state["day"],
            "ai_suggestion": self.ai_doctor(),
            "explanation": self.explain_action(action or self.ai_doctor()),
            "done": done,
            "reward": reward
        }