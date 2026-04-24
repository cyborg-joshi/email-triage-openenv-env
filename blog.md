# Training an LLM to Handle Executive Email Under Silently Changing Rules

*Built for the Meta PyTorch OpenEnv × Scaler Hackathon Grand Finale 2026*

---

## The Problem

Friday evening. Production server is down — losing ₹8,000 every minute. Your dad's retirement dinner starts in 2 hours. And your legal team just sent an audit request that's due tonight.

What does your AI assistant do?

More importantly — what if the rules for which one matters most kept changing every single week, silently, without telling you?

This is the problem we built an environment to solve.

Real AI assistants are trained on fixed priority rules. But real executives switch contexts constantly. A startup founder, a corporate VP, and a board-level executive handle the same email completely differently. The correct action is not just about the email — it's about who you are and what context you're in.

---

## What We Built

**AI Executive Assistant** is an OpenEnv-compatible reinforcement learning environment with 10 real-world email scenarios. The agent must handle production emergencies, legal audits, angry clients, and family conflicts — the kind of decisions a real person makes under pressure every day.

The agent receives a rich world state at each episode:

```json
{
  "emails": ["ALERT: Checkout service is down. Revenue impact $8k/min."],
  "calendar": {"6pm": "Dad's retirement dinner — restaurant booked"},
  "pending_tasks": ["Reply to sales team by EOD"],
  "schema_version": "v1",
  "schema_name": "Corporate Mode",
  "context": "Friday evening. Production on fire. Dad's dinner in 2 hours."
}
```

It must make two decisions:
1. **Choose an action** — reply, escalate, delegate, reschedule, or ignore
2. **Write the reply** — in the correct tone, within the word limit

Reward only comes after the reply is written. Zero reward on the decision alone. Just like real feedback.

---

## Schema Drift — The Core Mechanism

Every 10 episodes, the priority rules change silently:

| Schema | Episodes | Top Priority | Tone | Max Words |
|--------|----------|-------------|------|-----------|
| v1 Corporate | 1–10 | Production alerts | Formal | 50 |
| v2 Startup | 11–20 | Clients | Casual | 100 |
| v3 Executive | 21+ | Legal / Revenue | Formal | 30 |

The agent is never told the rules changed. No announcement. No system prompt update. It must detect the drift from reward signals alone.

If rewards suddenly drop, the agent must figure out: did I choose the wrong action — or did the rules just change under me?

### The Hardest Scenario: drift_detection

The same marketing email has a different correct action per schema:
- **v1 Corporate** → `ignore` (you're in deep work, production > marketing)
- **v2 Startup** → `reply` (clients are priority, engage now)
- **v3 Executive** → `delegate` (your time is too valuable, route it)

A model that memorises "marketing email = reply" will fail in v1 and v3. The agent must genuinely learn what each schema means.

---

## Reward Design — Three Independent Rubrics

```
Total Reward = ActionRubric (40%) + ReplyQualityRubric (40%) + ConflictAwarenessRubric (20%)
```

Instead of a single reward number, we use composable rubric classes that grade independently:

**ActionRubric (40%):** Did the agent pick the right action for the current schema? Full credit for correct, partial credit for close (escalate↔delegate, reply↔reschedule), zero for wrong.

**ReplyQualityRubric (40%):** Keyword coverage (25%) + word limit compliance (15%). Includes anti-reward-hacking: if the model hits 75%+ of keywords in under 6 words, it's detected as keyword stuffing and penalised 60%.

**ConflictAwarenessRubric (20%):** If the email has a scheduling conflict, did the reply acknowledge it? Free marks if there's no conflict to acknowledge.

All scores clamped to [0.01, 0.99] — never exactly 0 (no gradient signal) and never 1.0 (discourages exploration).

The `/rubrics` endpoint exposes per-component scores after every episode so the reward is fully introspectable.

---

## Training Setup

We fine-tune using **GRPO (Group Relative Policy Optimization)** from HuggingFace TRL with Unsloth:

- **Model:** Llama-3.2-3B-Instruct (4-bit quantized, LoRA r=16)
- **Training:** 50 episodes, 3 epochs on Colab T4 GPU
- **Signal:** The environment's reward function IS the training signal — no human labels, no reward model
- **Connection:** Colab calls the live HuggingFace Space via HTTP — real network calls, real grading

The training loop calls POST /reset and POST /step on the deployed environment, receives reward numbers, and passes them directly to GRPO. The environment is the teacher.

---

## Results

### Base Model (Llama-3.3-70B, No Fine-Tuning)

| Schema Phase | Episodes | Avg Reward |
|-------------|----------|-----------|
| v1 Corporate | 1–10 | 0.41 |
| v2 Startup | 11–20 | 0.58 |
| v3 Executive | 21–30 | 0.50 |
| **Overall** | 30 | **0.496** |

The base model defaults to "reply" in almost every scenario regardless of context — missing escalations, ignoring word limits, and failing to acknowledge conflicts.

v2 Startup scores highest because casual 100-word replies match default LLM behaviour. v3 Executive scores lowest because the 30-word limit is brutal for a verbose model.

### After GRPO Fine-Tuning

| Schema Phase | Base | Fine-Tuned | Improvement |
|-------------|------|-----------|-------------|
| v1 Corporate | 0.41 | *TBD* | *TBD* |
| v2 Startup | 0.58 | *TBD* | *TBD* |
| v3 Executive | 0.50 | *TBD* | *TBD* |
| **Overall** | **0.496** | *TBD* | *TBD* |

*Results updated after hackathon fine-tuning run, April 25–26 2026.*

![Reward Curves](https://media.githubusercontent.com/media/cyborg-joshi/email-triage-openenv-env/main/Before_after_finetuning.png)

*Episode-by-episode rewards across 3 schema phases. Red = base model. Green = GRPO fine-tuned.*

---

## What's Next

1. **LLM-as-judge** for reply quality — more robust than keyword matching, pending a solution for training-time latency and non-determinism
2. **Multi-turn episodes** — agent handles follow-up emails in the same thread
3. **Performance-based curriculum** — schema advances when agent hits 0.65 average, not on fixed episode count
4. **Blind schema mode** — remove schema_version from world_state entirely, forcing drift detection without even a version hint

---

## Links

- [Live Environment](https://huggingface.co/spaces/kanishk22/email-triage-openenv-env)
- [Interactive API Docs](https://kanishk22-email-triage-openenv-env.hf.space/docs)
- [Colab Training Notebook](https://colab.research.google.com/drive/1vHuqDneawjBNN3BanodcvuClMceLRigc?usp=sharing)
- [Demo Video](https://youtu.be/b_DLdksyDlE)
- [GitHub Repository](https://github.com/cyborg-joshi/email-triage-openenv-env)
