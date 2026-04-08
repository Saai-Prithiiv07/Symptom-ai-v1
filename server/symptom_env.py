import uuid
from dataclasses import dataclass
from typing import Dict, List

from server.models import SymptomObservation, SymptomState


@dataclass
class TaskConfig:
    task_id: str
    name: str
    difficulty: str
    patient_note: str
    initial_severity: int
    recommended_action: str
    max_steps: int


TASKS: List[TaskConfig] = [
    TaskConfig("seasonal-cold", "Seasonal Cold Triage", "easy", "Mild sore throat, low fever, can work normally.", 3, "rest", 6),
    TaskConfig("viral-fever", "Viral Fever Management", "medium", "Persistent fever, fatigue, appetite loss for 2 days.", 6, "medicine", 7),
    TaskConfig("red-flag-breathing", "Breathing Difficulty Escalation", "hard", "High fever with breathing discomfort and chest tightness.", 8, "doctor", 8),
]
TASK_LOOKUP: Dict[str, TaskConfig] = {task.task_id: task for task in TASKS}


class SymptomEnvironment:
    def __init__(self):
        self._task = TASKS[0]
        self._state: SymptomState | None = None
        self._last_error: str | None = None
        self.reset()

    @property
    def state(self) -> SymptomState:
        assert self._state is not None
        return self._state

    def list_tasks(self) -> List[dict]:
        return [{"id": t.task_id, "name": t.name, "difficulty": t.difficulty, "max_steps": t.max_steps, "reward_range": [0.0, 1.0]} for t in TASKS]

    def _ai_suggestion(self) -> str:
        if self.state.severity >= 7:
            return "doctor"
        if self.state.severity >= 4:
            return "medicine"
        return "rest"

    def _compute_progress(self) -> float:
        baseline = max(self._task.initial_severity, 1)
        return min(max((baseline - self.state.severity) / baseline, 0.0), 1.0)

    def _build_observation(self, reward: float, done: bool) -> SymptomObservation:
        score = min(max(self.state.total_reward / float(self._task.max_steps), 0.0), 1.0)
        return SymptomObservation(
            task_id=self._task.task_id,
            task_name=self._task.name,
            patient_note=self._task.patient_note,
            severity=self.state.severity,
            energy=self.state.energy,
            immunity=self.state.immunity,
            day=self.state.step_count,
            ai_suggestion=self._ai_suggestion(),
            progress=self._compute_progress(),
            last_action_error=self._last_error,
            done=done,
            reward=min(max(reward, 0.0), 1.0),
            score=score,
        )

    def reset(self, task_id: str | None = None) -> SymptomObservation:
        self._task = TASK_LOOKUP.get(task_id, TASKS[0]) if task_id else TASKS[0]
        self._state = SymptomState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            task_id=self._task.task_id,
            task_name=self._task.name,
            severity=self._task.initial_severity,
            energy=5,
            immunity=5,
            done=False,
            total_reward=0.0,
        )
        self._last_error = None
        return self._build_observation(0.0, False)

    def step(self, action_input: str) -> tuple[SymptomObservation, float, bool, dict]:
        if self.state.done:
            obs = self._build_observation(0.0, True)
            return obs, 0.0, True, {"warning": "Episode already finished."}

        action = (action_input or "").strip().lower()
        if action not in {"rest", "medicine", "doctor"}:
            self._last_error = f"invalid action: {action_input}"
            action = "rest"
        else:
            self._last_error = None

        self.state.step_count += 1
        if action == "doctor":
            self.state.severity -= 3
            self.state.energy -= 1
            self.state.immunity += 1
        elif action == "medicine":
            self.state.severity -= 2
            self.state.immunity += 1
        else:
            self.state.severity -= 1
            self.state.energy += 2
            self.state.immunity += 1

        self.state.severity = max(0, min(10, int(self.state.severity)))
        self.state.energy = max(0, min(10, int(self.state.energy)))
        self.state.immunity = max(0, min(10, int(self.state.immunity)))

        action_quality = 1.0 if action == self._task.recommended_action else 0.4
        improvement = 1.0 if self.state.severity < self._task.initial_severity else 0.2
        penalty = 0.0 if self._last_error is None else 0.2
        reward = min(max((0.5 * action_quality) + (0.5 * improvement) - penalty, 0.0), 1.0)

        self.state.total_reward += reward
        done = self.state.severity == 0 or self.state.step_count >= self._task.max_steps
        self.state.done = done
        obs = self._build_observation(reward, done)
        return obs, reward, done, {"task_id": self._task.task_id, "recommended_action": self._task.recommended_action, "difficulty": self._task.difficulty}