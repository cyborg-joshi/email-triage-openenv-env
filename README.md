---
title: AI Executive Assistant Environment
emoji: 📬
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# AI Personal Executive Assistant — OpenEnv Environment

An OpenEnv-compatible reinforcement learning environment that trains AI agents
to handle realistic email scenarios with **dynamic world state** and **schema drift**.

The agent receives emails, calendar conflicts, and pending tasks — then must choose
the correct action and write an appropriate reply. Every 10 episodes, the priority
rules change silently, forcing the agent to detect and adapt.

**Theme:** 3.2 — World Modeling (Personal Tasks) + Patronus AI Schema Drift Bonus

---

## Materials

| Resource | Link |
|----------|------|
| Live Environment | https://huggingface.co/spaces/kanishk22/email-triage-openenv-env |
| **Interactive Demo (Gradio)** | **https://huggingface.co/spaces/kanishk22/email-triage-demo** |
| Interactive API Docs | https://kanishk22-email-triage-openenv-env.hf.space/docs |
| HF Jobs Training Script | [train.py](./train.py) |
| Demo Video | https://youtu.be/b_DLdksyDlE |
| Blog / Writeup | [blog.md](./blog.md) |
| WandB Training Logs | https://wandb.ai/kanishkjoshi22-cisco/email-triage-schema-drift/runs/tsjx1n1p |
| GitHub | https://github.com/cyborg-joshi/email-triage-openenv-env |

---

## Results

### Base Model (Llama-3.3-70B, No Fine-Tuning)

30 episodes across all 3 schema phases, evaluated on the live environment:

| Schema Phase | Episodes | Avg Reward |
|-------------|----------|-----------|
| v1 Corporate | 1–10 | 0.41 |
| v2 Startup | 11–20 | 0.58 |
| v3 Executive | 21–30 | 0.50 |
| **Overall** | 30 | **0.496** |

### After GRPO Fine-Tuning (Llama-3.2-3B vs Llama-3.3-70B baseline)

| Schema Phase | 70B Base | 3B Fine-Tuned |
|-------------|----------|--------------|
| v1 Corporate | 0.41 | 0.38 |
| v2 Startup | 0.58 | 0.39 |
| v3 Executive | 0.50 | 0.45 |
| **Overall** | **0.496** | **0.407** |

*The 3B fine-tuned model achieves ~82% of the 70B baseline score at 1/23rd the size. The 3B model collapsed to always choosing `delegate` — a known GRPO action collapse failure mode where the model finds a safe local optimum and stops exploring. Fix: entropy bonus in the GRPO loss to force action diversity.*

*Results from hackathon Grand Finale, April 25–26 2026.*

### Reward Curves

![Before/After Fine-Tuning](https://media.githubusercontent.com/media/cyborg-joshi/email-triage-openenv-env/main/Before_after_finetuning.png)

*Episode-by-episode reward across 3 schema phases (v1 Corporate → v2 Startup → v3 Executive). Red line = base model (Llama-3.3-70B, no fine-tuning). Green line = GRPO fine-tuned model (Llama-3.2-3B, LoRA). Updated at hackathon Grand Finale April 25–26.*

### Training Logs

![train/rewards/reward_fn/mean](./wandb_reward_fn_mean.png)

![train/reward](./wandb_train_reward.png)

*GRPO fine-tuning run — `train/rewards/reward_fn/mean` trends from ~0.28 → ~0.42+ over 500 steps. The environment's reward function was the only training signal — no human labels. [Full WandB run](https://wandb.ai/kanishkjoshi22-cisco/email-triage-schema-drift/runs/5omalmor)*

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

See the [Colab training notebook](https://colab.research.google.com/drive/1gytu7Nlkm53UT1BN2_2fOFKcr-wNliQw?usp=sharing) to reproduce.

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

