---
title: AI Executive Assistant Environment
emoji: 📬
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# AI Personal Executive Assistant — OpenEnv Environment

## The Problem

Real AI assistants are trained on fixed priority rules. But real humans switch contexts constantly — a startup founder, a corporate VP, and a board-level executive handle the same email completely differently. The correct action depends not just on the email, but on *who you are and what world you're in.*

This environment tests whether an LLM can detect and adapt to **silently changing rules** — without being told the rules changed. Every 10 episodes, the priority schema switches. The agent must figure out the new rules from reward signals alone.

**This is schema drift. It's what makes this environment genuinely hard.**

**Theme:** 3.2 — World Modeling (Personalized Tasks)

---

## Materials

| Resource | Link |
|----------|------|
| Live Environment | https://huggingface.co/spaces/kanishk22/email-triage-openenv-env |
| **Interactive Demo (Gradio)** | **https://huggingface.co/spaces/kanishk22/email-triage-demo** |
| Interactive API Docs | https://kanishk22-email-triage-openenv-env.hf.space/docs |
| HF Jobs Training Script | [train.py](./train.py) |
| Colab Training Notebook | https://colab.research.google.com/drive/1sqHn3AJB-PhwQ936fwWS7R4LSt_GPifC?usp=sharing |
| Demo Video | https://youtu.be/sfG-9tPbusc |
| Blog / Writeup | [blog.md](./blog.md) |
| WandB Training Logs | https://wandb.ai/kanishkjoshi22-cisco/email-triage-schema-drift/runs/xgiv7xo6 |
| GitHub | https://github.com/cyborg-joshi/email-triage-openenv-env |

---

## Results

### Training Progress — Reward Improved During Fine-Tuning

![train/rewards/reward_fn/mean — x: training step (0→4500), y: mean reward per batch](./wandb_reward_fn_mean.png)

*`train/rewards/reward_fn/mean` over 600 GRPO training steps. Reward trends upward from ~0.30 → ~0.40+. The environment's live reward function was the only training signal — no human labels, no reward model. [Full WandB run](https://wandb.ai/kanishkjoshi22-cisco/email-triage-schema-drift/runs/xgiv7xo6)*

![train/reward — x: training step, y: reward](./wandb_train_reward.png)

*`train/reward` over 600 steps — same upward trend confirming the learning signal was working throughout.*

### Before vs After — Episode Rewards Across 3 Schema Phases

![Before/After Fine-Tuning — x: episode (1→30), y: reward per episode](./before_after_finetuning.png)

*Episode-by-episode reward across all 3 schema phases. Red = Llama-3.2-3B base (no training). Green = GRPO fine-tuned Llama-3.2-3B. Reward drops at episodes 10 and 20 show schema drift kicking in — rules changed silently, both models take a hit.*

### Key Result — GRPO Fine-Tuning Improves 3B Model by +14%

| Model | v1 Corporate | v2 Startup | v3 Executive | **Overall** | **vs 3B Base** |
|-------|-------------|-----------|-------------|------------|----------------|
| Llama-3.2-3B (no training) | 0.301 | 0.303 | 0.415 | **0.340** | baseline |
| Llama-3.2-3B + GRPO fine-tuned | 0.346 | 0.385 | 0.435 | **0.389** | **+14%** |
| Llama-3.3-70B (no training, reference) | 0.41 | 0.58 | 0.50 | **0.496** | +46% |

*GRPO fine-tuning on the **same 3B model** improves overall reward by **+14%** (+15% v1, +27% v2, +5% v3). Trained purely from the live environment's reward signal — no human labels, no reward model. The fine-tuned 3B also achieves **78% of the 70B reference score at 1/23rd the model size.***

---

## What Makes This Environment Unique

### Schema Drift
Every 10 episodes the environment silently switches its active rule schema:

| Schema | Episodes | Priority | Tone | Max Reply |
|--------|----------|----------|------|-----------|
| v1 Corporate | 1–10 | Production alerts | Formal | 50 words |
| v2 Startup | 11–20 | Clients | Casual | 100 words |
| v3 Executive | 21+ | Legal / Revenue | Formal | 30 words |

The agent is never told the rules changed. It must figure it out from the rewards.

### Two-Step Episodes
Each episode has two steps — testing both decision-making and communication:
- **Step 1:** Choose an action — `reply`, `escalate`, `delegate`, `reschedule`, `ignore`
- **Step 2:** Write the actual reply (tone and length graded against current schema)

Reward is given only at the end of step 2 (delayed reward).

### Rich World State
The agent sees more than just an email:
```json
{
  "emails": ["ALERT: Checkout service is down. Revenue $8k/min."],
  "calendar": {"6pm": "Dad's retirement dinner — restaurant booked"},
  "pending_tasks": ["Reply to sales team by EOD"],
  "schema_version": "v1",
  "schema_name": "Corporate Mode",
  "context": "Friday evening. Production on fire. Dad's dinner in 2 hours."
}
```

---

## Reward Model

```
Total Reward = action_correctness (40%)
             + reply_quality      (40%)
             + conflict_awareness (20%)
```

- **Action correctness:** Did the agent pick the right action for the current schema? Partial credit for close actions.
- **Reply quality:** Keyword coverage + tone compliance + length within schema limit
- **Conflict awareness:** Did the reply acknowledge any scheduling conflict?

All rewards clamped to `[0.01, 0.99]`.

---

## Tasks (10 Scenarios)

| Task ID | Description | Schema Sensitive |
|---------|-------------|-----------------|
| `conflict_work` | Production emergency vs personal dinner | Yes |
| `conflict_calendar` | Double-booked appointments | Yes |
| `boss_pressure` | Angry boss demanding overdue report | Yes |
| `personal_event` | Friend's wedding with flight conflict | No |
| `spam_disguised` | Phishing email disguised as urgent | Yes |
| `legal_compliance` | Legal audit with strict deadline | Yes |
| `personal_family` | Family emergency, emotional response needed | No |
| `client_urgent` | Enterprise client threatening CEO escalation | Yes |
| `finance_pressure` | Overdue invoice needing delegation | No |
| `drift_detection` | Ideal action changes across all schemas | Yes |

---

## Training Pipeline

Fine-tuning uses **GRPO (Group Relative Policy Optimization)** from HuggingFace TRL + Unsloth:

- Base model: `unsloth/Llama-3.2-3B-Instruct` (4-bit quantized via LoRA)
- The environment's reward function IS the training signal — no human labels, no reward model
- Training loop connects directly to the live HuggingFace Space via HTTP
- 50 training episodes, 3 epochs, LoRA on q_proj + v_proj

See the [Colab training notebook](https://colab.research.google.com/drive/1sqHn3AJB-PhwQ936fwWS7R4LSt_GPifC?usp=sharing) to reproduce.

---

## Why This Matters

Most RL environments have fixed rules. Real-world tasks don't. A personal assistant deployed at a startup behaves differently than one deployed at a bank — and both behave differently from one working for an executive who switched industries last month.

Schema drift is a proxy for **distribution shift in deployed AI systems** — the thing that causes production models to silently degrade. This environment forces the agent to detect and adapt, which is closer to the real problem than any static benchmark.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/reset?task=<id>` | Start new episode |
| `POST` | `/step` | Send action or reply |
| `GET` | `/state` | Current episode state |
| `GET` | `/rubrics` | Per-component reward breakdown (action / reply / conflict) |
| `GET` | `/schema` | Active schema + drift schedule |
| `POST` | `/admin/reset_env` | Reset episode counter to 0 (back to v1) |
| `GET` | `/docs` | Interactive API documentation |

### Example: Full Episode

```bash
# Reset
curl -X POST "https://kanishk22-email-triage-openenv-env.hf.space/reset?task=conflict_work"

# Step 1 — send action
curl -X POST "https://kanishk22-email-triage-openenv-env.hf.space/step" \
  -H "Content-Type: application/json" \
  -d '{"action": "escalate", "reply": "", "step": 1}'

# Step 2 — send reply, receive reward
curl -X POST "https://kanishk22-email-triage-openenv-env.hf.space/step" \
  -H "Content-Type: application/json" \
  -d '{"action": "escalate", "reply": "On it. Looping in the on-call team. Will be 20 min late to dinner.", "step": 2}'
```

---

## Project Structure

```
email-triage-openenv/
├── env/
│   ├── environment.py   — ExecutiveAssistantEnv (reset, step, schema drift)
│   ├── graders.py       — compute_reward (action + reply + conflict)
│   ├── models.py        — WorldState, ExecutiveAction, ExecutiveObservation
│   └── scenarios.py     — All 10 email scenarios + 3 schema definitions
├── server/
│   └── app.py           — FastAPI server (7 endpoints + singleton env)
├── before_after_finetuning.png  — Reward curves (base vs fine-tuned)
├── inference.py         — Agent script (calls environment via HTTP)
├── openenv.yaml         — OpenEnv manifest
├── Dockerfile           — Container config for HuggingFace Spaces
└── requirements.txt     — Dependencies
```

---

## Interactive Demo

Try the environment yourself — no API knowledge needed:

**[https://huggingface.co/spaces/kanishk22/email-triage-demo](https://huggingface.co/spaces/kanishk22/email-triage-demo)**

1. Pick a scenario (e.g. `conflict_work`)
2. Click **Good Example** or **Bad Example** to pre-fill action + reply
3. Hit **Submit & Score** — see reward + rubric breakdown instantly
4. Switch scenarios to explore schema drift across Corporate / Startup / Executive modes

---

## Running Locally

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000
# Open http://localhost:8000/docs to test all endpoints
```

---

## Built With

- [OpenEnv](https://github.com/huggingface/openenv) — RL environment framework
- [FastAPI](https://fastapi.tiangolo.com) — API server
- [Pydantic](https://docs.pydantic.dev) — Data validation
- [TRL](https://github.com/huggingface/trl) + [Unsloth](https://github.com/unslothai/unsloth) — Fine-tuning
- [Llama-3.3-70B](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) — Agent LLM
- [HuggingFace Spaces](https://huggingface.co/spaces) — Deployment

---

**Author:** kanishk22 | Meta PyTorch OpenEnv × Scaler Hackathon Grand Finale 2026

