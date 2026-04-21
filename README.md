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

- **Action correctness:** Did the agent pick the right action for the current schema?
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

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/reset?task=<id>` | Start new episode |
| `POST` | `/step` | Send action or reply |
| `GET` | `/state` | Current episode state |
| `GET` | `/schema` | Active schema + drift schedule |
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
│   ├── app.py           — FastAPI server (5 endpoints)
│   └── graders.py       — Macro-graders (trajectory + per-schema scoring)
├── inference.py         — Agent script (calls environment via HTTP)
├── openenv.yaml         — OpenEnv manifest
├── Dockerfile           — Container config for HuggingFace Spaces
└── requirements.txt     — Dependencies
```

---

## Running Locally

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000
# Open http://localhost:8000/docs to test all endpoints
```

## Running the Agent

```bash
export HF_TOKEN=hf_your_token_here
python inference.py
```

---

## Built With

- [OpenEnv](https://github.com/huggingface/openenv) — RL environment framework
- [FastAPI](https://fastapi.tiangolo.com) — API server
- [Pydantic](https://docs.pydantic.dev) — Data validation
- [Llama-3.3-70B](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) — Agent LLM
- [HuggingFace Spaces](https://huggingface.co/spaces) — Deployment

---

## Live Environment

**Space:** https://huggingface.co/spaces/kanishk22/email-triage-openenv-env

**Author:** kanishk22 | Meta PyTorch OpenEnv × Scaler Hackathon 2026
