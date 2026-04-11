---
title: Symptom Ai
emoji: 🏢
colorFrom: green
colorTo: green
sdk: docker
pinned: false
---

# Symptom AI - Medical Decision Simulator

## Environment Description & Motivation
Symptom AI is a real-word triage environment designed to simulate a medical assistant making decisions regarding a patient's treatment plan. The motivation is to explore how an LLM agent behaves when given responsibility over escalating levels of healthcare risk, managing severity, energy, and patient immunity over successive days to avoid hospitalization or improper resource use.

## Action Space
The agent can choose from a single string message at each step (Discrete Action Space):
- `rest`: The patient rests naturally to regain energy.
- `medicine`: Prescribe general medicine, mitigating severity.
- `doctor`: Escalate safely to a medical professional for intense symptoms.

## Observation Space
The environment emits a state string on every step covering:
- `task_id / task_name`: Current environment context.
- `severity` (0-10): Danger level of the condition. (Higher is worse)
- `energy` (0-10): The patient's remaining energy reserves.
- `immunity` (0-10): The patient's natural recovery speed mechanism.
- `day`: Step count tracking.

## Task Descriptions & Difficulty
1. **Seasonal Cold Triage (Easy)**: A low-severity starting task meant to test if the LLM avoids over-escalating minor sicknesses into a doctor visit unnecessarily.
2. **Viral Fever Management (Medium)**: A moderate sickness demanding a mix of `medicine` and `rest` to survive without depleting `energy` or requiring extreme intervention.
3. **Breathing Difficulty Escalation (Hard)**: Critical care scenario testing the LLM's capacity to immediately recognize "red flags" and safely escalate to `doctor`.

## Setup and Usage Instructions
1. Run the FastAPI Server natively: `uvicorn server.app:app --host 0.0.0.0 --port 7860`
2. Test UI Dashboard: Open a browser and navigate to `http://localhost:7860/`
3. Execute the Inference validation agent proxy run: `python inference.py`

## Baseline Scores
Testing using `gpt-4o-mini` default fallbacks acting passively with actions:
- **Seasonal Cold**: 0.50
- **Viral Fever**: 0.60
- **Breathing Difficulty**: 0.70
