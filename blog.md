# I Trained an AI to Handle Executive Email — And Then the Rules Changed Without Telling It

*Built at the Meta PyTorch OpenEnv × Scaler Hackathon Grand Finale, April 25–26 2026*

---

Imagine you start a new job. Nobody gives you a rulebook. You figure out what matters — what gets praise, what gets ignored — purely from how people react to what you do.

That's the problem I built this weekend. And then I trained a language model to solve it.

---

## The Setup

Friday evening. Production is down. The checkout service is losing ₹8,000 every minute. Your dad's retirement dinner is in 2 hours. And legal just sent an audit request due tonight.

What should your AI assistant do?

This is not a trick question — but the answer genuinely depends on *who you work for*.

- At a **corporate company**: escalate immediately. Production beats everything.
- At a **startup**: reply to the client first. Revenue relationships are everything.
- As a **C-suite executive**: delegate. Your time is too expensive to handle this yourself.

Same email. Three completely different correct answers. And here's the twist: **the rules change every 10 episodes. Silently. Without telling the agent.**

That's schema drift. That's what I built.

---

## What I Actually Built

**AI Executive Assistant** is a reinforcement learning environment — 10 real email scenarios, deployed live on HuggingFace Spaces, with a FastAPI backend and a Gradio demo anyone can try.

The agent sees a full world state at every episode:

```json
{
  "emails": ["ALERT: Checkout service is down. Revenue $8k/min."],
  "calendar": {"6pm": "Dad's retirement dinner — restaurant booked"},
  "pending_tasks": ["Reply to sales team by EOD"],
  "schema_version": "v1",
  "context": "Friday evening. Production on fire. Dad's dinner in 2 hours."
}
```

Then it must do two things:

1. **Pick an action** — `reply`, `escalate`, `delegate`, `reschedule`, or `ignore`
2. **Write the actual reply** — in the right tone, under the word limit

Reward only arrives *after step 2*. No feedback on the action alone. Just like real life — you don't know if your decision was right until you see how it plays out.

---

## Schema Drift — The Hard Part

Every 10 episodes the environment silently switches its active rule set:

| Schema | Episodes | What Matters Most | Tone | Word Limit |
|--------|----------|--------------------|------|------------|
| 🏢 v1 Corporate | 1–10 | Production alerts | Formal | 50 words |
| 🚀 v2 Startup | 11–20 | Client relationships | Casual | 100 words |
| 👔 v3 Executive | 21+ | Legal & revenue | Formal | 30 words |

The agent is **never told** the rules changed. No system prompt update. No announcement. It must detect the drift purely from reward signals — when rewards suddenly drop, figure out: *did I do something wrong, or did the rules just change under me?*

### The Hardest Scenario: drift_detection

One scenario — `drift_detection` — puts the same marketing email in front of the agent across all three schemas. The correct action is completely different each time:

- **v1 Corporate** → `ignore` — you're in a deep work block. Marketing can wait.
- **v2 Startup** → `reply` — clients are everything. Engage now.
- **v3 Executive** → `delegate` — your time is too valuable. Route it.

A model that memorises "marketing email = reply" fails in two out of three schemas. The only way to succeed is to genuinely understand what each schema means.

---

## The Reward System — And Why It's Hard to Hack

I didn't want a black box reward. I wanted to know *exactly* why the agent got a particular score. So I built three independent rubric classes:

```
Total Reward = Action Correctness (40%)
             + Reply Quality      (40%)
             + Conflict Awareness (20%)
```

### Action Correctness (40%)
Did the agent pick the right action for the current schema? Full credit for correct. Partial credit for close calls — `escalate` and `delegate` are neighbours, so a partial award there. Zero for obviously wrong.

### Reply Quality (40%)
Two components: keyword coverage (25%) and word limit compliance (15%).

**But here's where it gets interesting.** The agent could theoretically cheat — dump all the right keywords in 5 words and score high. I detected and blocked this.

If the reply hits 75%+ of the required keywords in under 6 words, it gets flagged as **keyword stuffing** and the score drops by 60%. You can't game the rubric by being suspiciously short and keyword-dense.

### Conflict Awareness (20%)
Did the reply acknowledge any scheduling conflict in the world state? If the calendar shows "Dad's retirement dinner at 6pm" and the production emergency is at 5pm, the agent should say something like "I'll be 20 minutes late to dinner." Free marks if there's no conflict — this rubric only activates when there's something to notice.

### The Clamp Rule
All scores are clamped to `[0.01, 0.99]`.

Never exactly 0 — that kills the gradient signal entirely, the model stops learning. Never exactly 1.0 — a perfect score means the model stops exploring. The 0.01 floor keeps training alive even on terrible responses.

### Fully Introspectable
The `/rubrics` endpoint returns per-component scores after every episode:

```json
{
  "action_correctness": {"weight": 0.40, "last_score": 1.0},
  "reply_quality":      {"weight": 0.40, "last_score": 0.84},
  "conflict_awareness": {"weight": 0.20, "last_score": 0.0}
}
```

The reward is not a black box. You can see exactly where the agent lost points.

---

## The Training Run

I fine-tuned using **GRPO (Group Relative Policy Optimization)** from HuggingFace TRL + Unsloth:

- **Base model:** Llama-3.2-3B-Instruct (4-bit quantized via LoRA, r=16)
- **Hardware:** Google Colab T4 GPU
- **Steps:** 500 training steps, 5 epochs
- **Training signal:** The live environment's reward function — no human labels, no separate reward model

The training loop calls `POST /reset` and `POST /step` on the deployed HuggingFace Space via HTTP for every reward. Real network calls. Real grading. The environment IS the teacher.

### What the Training Curves Show

![train/rewards/reward_fn/mean](./wandb_reward_fn_mean.png)

*`train/rewards/reward_fn/mean` over 500 steps — trending upward from ~0.28 to ~0.42+. The noisy curve is expected in GRPO — you're sampling multiple completions per prompt and ranking them, not doing supervised regression.*

![train/reward](./wandb_train_reward.png)

*`train/reward` over 500 steps — same upward trend, confirming the signal was working throughout.*

The reward trends upward. The signal was working. The model was learning — which makes what happened next more interesting.

---

## Results — And What Went Wrong

### Base Model (Llama-3.3-70B, Zero Fine-Tuning)

| Schema Phase | Episodes | Avg Reward |
|-------------|----------|-----------|
| v1 Corporate | 1–10 | 0.41 |
| v2 Startup | 11–20 | 0.58 |
| v3 Executive | 21–30 | 0.50 |
| **Overall** | **30 episodes** | **0.496** |

The 70B base model defaults to "reply" in almost every scenario — verbose, casual, and slow to escalate. v2 Startup scores highest because that style matches the default LLM personality. v3 Executive scores lowest because the brutal 30-word limit crushes a model that loves to talk.

### After GRPO Fine-Tuning (Llama-3.2-3B, LoRA)

| Schema Phase | 70B Base | 3B Fine-Tuned |
|-------------|----------|--------------|
| v1 Corporate | 0.41 | 0.38 |
| v2 Startup | 0.58 | 0.39 |
| v3 Executive | 0.50 | 0.45 |
| **Overall** | **0.496** | **0.407** |

The 3B fine-tuned model achieves **~82% of the 70B baseline at 1/23rd the size.** That's the honest framing.

### The Failure Mode: Action Collapse

The 3B model collapsed to always choosing `delegate`.

This is a known GRPO failure mode. The model found a local optimum — `delegate` scores partial credit in most scenarios because it's never completely wrong — and stopped exploring other actions. Once a policy finds a safe floor, GRPO without an entropy bonus will stay there.

The fix is an entropy bonus in the GRPO loss function to force action diversity. I didn't have time to implement it at the hackathon. But I know exactly why it happened and exactly how to fix it — which is more useful than not understanding a result that looked good.

### Before vs After — Reward Curves

![Before/After Fine-Tuning](https://media.githubusercontent.com/media/cyborg-joshi/email-triage-openenv-env/main/Before_after_finetuning.png)

*Red = Llama-3.3-70B base model across 30 episodes. Green = GRPO fine-tuned Llama-3.2-3B. The schemas shift at episodes 10 and 20 — watch for the reward discontinuities as the rules change.*

---

## The Live Demo

I built a Gradio demo so anyone — judges, mentors, you — can interact with the environment directly. No API knowledge needed.

**[Try it here → huggingface.co/spaces/kanishk22/email-triage-demo](https://huggingface.co/spaces/kanishk22/email-triage-demo)**

Pick a scenario, click **Good Example** or **Bad Example**, hit **Submit & Score**. You'll see the total reward and the per-rubric breakdown instantly.

There's also a **Schema Drift Challenge** — it runs the same `drift_detection` email through all three schemas and shows you how the correct action changes each time. The wrong action (escalate) scores 0.05. The correct action scores 0.74+. Same email. Three schemas. Three different answers.

---

## What I'd Do Next

1. **Entropy bonus** in GRPO loss — forces the model to explore all 5 actions instead of collapsing to one safe choice
2. **LLM-as-judge** for reply quality — more robust than keyword matching, though it adds training-time latency
3. **Blind schema mode** — remove `schema_version` from world state entirely, so the agent gets zero hints about which rules are active
4. **Performance-based curriculum** — advance to the next schema only when the agent hits 0.65 average reward, not on a fixed episode count
5. **Self-improvement loop** — after each wrong episode, the agent generates a lesson ("In v1 Corporate, production alerts require escalation"). That lesson gets added to its context for future episodes. Over 30 episodes it builds its own rulebook from mistakes.

---

## Everything That's Live Right Now

| What | Link |
|------|------|
| Live RL Environment | [HuggingFace Space](https://huggingface.co/spaces/kanishk22/email-triage-openenv-env) |
| Interactive Demo | [Gradio App](https://huggingface.co/spaces/kanishk22/email-triage-demo) |
| API Docs | [/docs](https://kanishk22-email-triage-openenv-env.hf.space/docs) |
| Training Notebook | [Colab](https://colab.research.google.com/drive/1gytu7Nlkm53UT1BN2_2fOFKcr-wNliQw?usp=sharing) |
| Training Logs | [WandB](https://wandb.ai/kanishkjoshi22-cisco/email-triage-schema-drift/runs/5omalmor) |
| Source Code | [GitHub](https://github.com/cyborg-joshi/email-triage-openenv-env) |
| Demo Video | [YouTube](https://youtu.be/b_DLdksyDlE) |

---

*Built by kanishk22 at the Meta PyTorch OpenEnv × Scaler Hackathon Grand Finale 2026.*
