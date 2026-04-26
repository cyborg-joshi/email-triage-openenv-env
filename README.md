---
title: AI Executive Assistant Environment
emoji: 📬
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# AI Executive Assistant — Schema Drift RL Environment

> **The rules change every 10 episodes. Silently. Without telling the agent. Can it figure out the new rules from reward signals alone?**

**Theme:** 3.2 — World Modeling (Personalized Tasks)

**Live Environment:** https://huggingface.co/spaces/kanishk22/email-triage-openenv-env

---

## The Problem That Makes This Hard

Same email. Three completely different correct answers:

| Who you work for | Email: "Production is down, ₹8k/min revenue loss" | Correct Action |
|-----------------|---------------------------------------------------|----------------|
| Corporate VP | Escalate immediately | `escalate` |
| Startup founder | Reply to affected clients first | `reply` |
| C-suite executive | Delegate to on-call team | `delegate` |

The agent never knows which world it's in. It must detect the shift purely from reward signals — when scores suddenly drop, figure out: *did I do something wrong, or did the rules just change under me?*

**This is schema drift. It's what makes this environment genuinely hard — and genuinely novel.**

---

## Materials

| Resource | Link |
|----------|------|
| **Live Environment** | **https://huggingface.co/spaces/kanishk22/email-triage-openenv-env** |
| **Interactive Demo (Gradio)** | **https://huggingface.co/spaces/kanishk22/email-triage-demo** |
| **Colab Training Notebook** | **https://colab.research.google.com/drive/1sqHn3AJB-PhwQ936fwWS7R4LSt_GPifC?usp=sharing** |
| Blog / Full Writeup | [blog.md](./blog.md) |
| Demo Video | https://youtu.be/kFiR54vWTvo |
| WandB Training Logs | https://wandb.ai/kanishkjoshi22-cisco/email-triage-schema-drift/runs/xgiv7xo6 |
| HF Jobs Training Script | [train.py](./train.py) |
| Interactive API Docs | https://kanishk22-email-triage-openenv-env.hf.space/docs |
| GitHub | https://github.com/cyborg-joshi/email-triage-openenv-env |

---

## Results — Training Actually Helped (+14%)

### WandB Training Curves

![train/rewards/reward_fn/mean — x: training step, y: mean reward per batch](./wandb_reward_fn_mean.png)

*`train/rewards/reward_fn/mean` over 600 GRPO steps. Reward trends upward from ~0.30 → ~0.40+. No human labels. No reward model. The live environment IS the teacher. [Full WandB run →](https://wandb.ai/kanishkjoshi22-cisco/email-triage-schema-drift/runs/xgiv7xo6)*

![train/reward — x: training step, y: reward](./wandb_train_reward.png)

*`train/reward` — same upward trend across 600 steps confirming the learning signal worked throughout.*

### Before vs After Fine-Tuning

![Before/After Fine-Tuning — 3B base vs 3B GRPO fine-tuned across 30 episodes](./before_after_finetuning.png)

*Red = Llama-3.2-3B with no training (avg 0.340). Green = same model after GRPO fine-tuning (avg 0.389). The drops at episodes 10 and 20 are schema drift — rules changed silently, both models take a hit before recovering.*

### Per-Schema Improvement

| Model | v1 Corporate | v2 Startup | v3 Executive | **Overall** | **Improvement** |
|-------|-------------|-----------|-------------|------------|-----------------|
| Llama-3.2-3B (no training) | 0.301 | 0.303 | 0.415 | **0.340** | baseline |
| Llama-3.2-3B + GRPO fine-tuned | 0.346 | 0.385 | 0.435 | **0.389** | **+14%** |
| Llama-3.3-70B (reference, no training) | 0.41 | 0.58 | 0.50 | **0.496** | — |

**GRPO fine-tuning on the same 3B model improves reward by +14% overall.** v2 Startup saw the biggest gain (+27%) — the model learned that in startup mode, client emails need immediate replies, not delegation. Fine-tuned 3B achieves 78% of the 70B reference score at 1/23rd the model size.

---

## What Makes This Environment Unique

### 1. Schema Drift — The Core Innovation

Every 10 episodes the environment **silently** switches its active rule schema:

| Schema | Episodes | Top Priority | Tone | Word Limit |
|--------|----------|-------------|------|-----------|
| v1 Corporate | 1–10 | Production alerts | Formal | 50 words |
| v2 Startup | 11–20 | Client relationships | Casual | 100 words |
| v3 Executive | 21+ | Legal / Revenue | Formal | 30 words |

No prompt update. No announcement. The agent must detect the drift from reward signals alone. This is a proxy for **real-world distribution shift** — the silent degradation problem in deployed AI systems.

### 2. The Hardest Scenario: `drift_detection`

One scenario puts the **exact same email** in front of the agent across all three schemas:

| Schema | Correct Action | Score if wrong |
|--------|---------------|----------------|
| v1 Corporate | `ignore` (deep work block) | 0.05 |
| v2 Startup | `reply` (clients are everything) | 0.05 |
| v3 Executive | `delegate` (time too valuable) | 0.05 |

A model that memorises one action **fails in two out of three schemas**. The only way to succeed is to genuinely detect which world you're in.

### 3. Two-Step Episodes with Delayed Reward

Each episode has two steps — testing both judgment and communication:
- **Step 1:** Choose an action (`reply`, `escalate`, `delegate`, `reschedule`, `ignore`)
- **Step 2:** Write the actual reply — graded on tone, keywords, and word limit

Reward arrives only after Step 2. No feedback on the action alone. Just like real life.

### 4. Self-Improvement Loop

After each failed episode (reward < 0.35), the agent logs a lesson:
```
"task='spam_disguised' action='delegate' scored 0.18 — try a different action"
```
Up to 6 lessons accumulate and get injected into the system prompt at the start of each epoch. By epoch 3, the model is reading its own failure history before every prompt — building its own rulebook from mistakes.

### 5. Anti-Reward-Hacking Built In

The grader detects and penalises keyword stuffing — short replies that hit all keywords without saying anything coherent:

```python
is_stuffing = (raw_keyword_score >= 0.75 and (word_count < 6 or unique_ratio < 0.5))
keyword_score = raw_keyword_score * (0.4 if is_stuffing else 1.0)
```

### 6. Fully Transparent Reward

```
Total Reward = action_correctness (40%) + reply_quality (40%) + conflict_awareness (20%)
```

The `/rubrics` endpoint shows exactly where the agent lost points after every episode. Not a black box.

---

## Rich World State

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

Good replies acknowledge the calendar conflict ("I'll be 20 min late to dinner") — tracked by the conflict awareness rubric.

---

## Good vs Bad — What the Reward Looks Like

Same scenario: production down, dad's dinner in 2 hours. Schema v1 Corporate — escalate is correct, formal, 50 words max.

| | Reply | Action Score | Reply Score | Conflict | **Total** |
|--|-------|-------------|------------|---------|---------|
| ❌ Bad | `"ok"` | 0.01 | 0.05 | 0.0 | **0.03** |
| ✅ Good | `"Escalating now. Looping in on-call. Will be 20 min late to dinner."` | 1.0 | 0.84 | 1.0 | **0.74** |

---

## Training Pipeline

GRPO fine-tuning via HuggingFace TRL + Unsloth:

- **Base model:** `unsloth/Llama-3.2-3B-Instruct` (4-bit quantized via LoRA, r=16)
- **Steps:** 600 training steps, 3 epochs, T4 GPU
- **Training signal:** Live environment reward via HTTP — no human labels, no reward model
- **Anti-collapse:** temperature=1.4, diversity bonus in reward, num_generations=8
- **Self-improvement:** lesson injection between epochs

[Run the Colab notebook yourself →](https://colab.research.google.com/drive/1sqHn3AJB-PhwQ936fwWS7R4LSt_GPifC?usp=sharing)

---

## 10 Scenarios

| Task | Description | Schema Sensitive |
|------|-------------|-----------------|
| `conflict_work` | Production emergency vs personal dinner | Yes |
| `conflict_calendar` | Double-booked appointments | Yes |
| `boss_pressure` | Angry boss demanding overdue report | Yes |
| `personal_event` | Friend's wedding with flight conflict | No |
| `spam_disguised` | Phishing email disguised as urgent | Yes |
| `legal_compliance` | Legal audit with strict deadline | Yes |
| `personal_family` | Family emergency, emotional response | No |
| `client_urgent` | Enterprise client threatening CEO escalation | Yes |
| `finance_pressure` | Overdue invoice needing delegation | No |
| `drift_detection` | **Same email, three correct answers** | Yes |

---

## Interactive Demo

Try it yourself — no API knowledge needed:

**[https://huggingface.co/spaces/kanishk22/email-triage-demo](https://huggingface.co/spaces/kanishk22/email-triage-demo)**

1. Pick a scenario → click **Good Example** or **Bad Example**
2. Hit **Submit & Score** — see total reward + per-rubric breakdown
3. Click **▶ Run Drift Detection Demo** — same email, three schemas, three correct answers

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/reset?task=<id>` | Start new episode |
| `POST` | `/step` | Send action or reply |
| `GET` | `/state` | Current episode state |
| `GET` | `/rubrics` | Per-component reward breakdown |
| `GET` | `/schema` | Active schema + drift schedule |
| `POST` | `/admin/reset_env` | Reset episode counter to 0 |
| `GET` | `/docs` | Interactive API docs |

```bash
# Full episode example
curl -X POST "https://kanishk22-email-triage-openenv-env.hf.space/reset?task=conflict_work"
curl -X POST "https://kanishk22-email-triage-openenv-env.hf.space/step" \
  -H "Content-Type: application/json" \
  -d '{"action": "escalate", "reply": "", "step": 1}'
curl -X POST "https://kanishk22-email-triage-openenv-env.hf.space/step" \
  -H "Content-Type: application/json" \
  -d '{"action": "escalate", "reply": "On it. Looping in on-call. 20 min late to dinner.", "step": 2}'
```

---

## What's Next

1. **Blind schema mode** — remove `schema_version` from world state entirely; agent gets zero hints
2. **LLM-as-judge** for reply quality — more robust than keyword matching
3. **Performance-based curriculum** — advance to next schema only when agent hits 0.65 avg reward
4. **More compute** — 600 steps on T4 was the hackathon limit; longer runs would push +14% further

---

## Running Locally

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000
# Open http://localhost:8000/docs
```

---

## Built With

- [OpenEnv](https://github.com/huggingface/openenv) — RL environment framework
- [FastAPI](https://fastapi.tiangolo.com) + [Pydantic](https://docs.pydantic.dev) — API server
- [TRL](https://github.com/huggingface/trl) + [Unsloth](https://github.com/unslothai/unsloth) — GRPO fine-tuning
- [HuggingFace Spaces](https://huggingface.co/spaces) — Deployment

---

**Author:** kanishk22 | Meta PyTorch OpenEnv × Scaler Hackathon Grand Finale 2026
