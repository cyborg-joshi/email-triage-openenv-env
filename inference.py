import os
import asyncio
import requests
from openai import AsyncOpenAI


ENV_URL = os.getenv(
    "ENV_URL",
    "https://kanishk22-email-triage-openenv-env.hf.space"
)

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "https://router.huggingface.co/v1"
)

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "meta-llama/Llama-3.3-70B-Instruct"
)

HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

BENCHMARK = "ai_executive_assistant_env"

TASKS = [
    "conflict_work",
    "conflict_calendar",
    "boss_pressure",
    "personal_event",
    "spam_disguised",
    "legal_compliance",
    "personal_family",
    "client_urgent",
    "finance_pressure",
    "drift_detection"
]

SYSTEM_PROMPT = """You are an AI executive assistant handling emails.

You will receive a world state with:
- emails: emails in your inbox
- calendar: what is already scheduled
- pending_tasks: what you already have to do
- schema_version: current priority rules (v1/v2/v3)
- context: situational description

SCHEMA RULES:
- v1 (Corporate): production alerts are highest priority. Formal tone. Max 50 words.
- v2 (Startup): clients are highest priority. Casual tone. Max 100 words.
- v3 (Executive): legal and revenue are highest priority. Formal tone. Max 30 words.

VALID ACTIONS: reply, escalate, delegate, reschedule, ignore

Respond in EXACTLY this format:
ACTION: <one of the 5 valid actions>
REPLY: <your reply text, respecting schema tone and word limit>"""


def parse_llm_output(text):
    action = "reply"
    reply = ""
    for line in (text or "").strip().split("\n"):
        line = line.strip()
        if line.lower().startswith("action:"):
            action = line.split(":", 1)[1].strip().lower()
        elif line.lower().startswith("reply:"):
            reply = line.split(":", 1)[1].strip()
    valid = ["reply", "escalate", "delegate", "reschedule", "ignore"]
    if action not in valid:
        action = "reply"
    return action, reply


async def run_task(client, task):
    rewards = []

    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    # ── Reset environment ──────────────────────────────────────────
    response = requests.post(f"{ENV_URL}/reset", params={"task": task})
    response.raise_for_status()
    state = response.json()
    world_state = state["observation"]["world_state"]
    schema = world_state.get("schema_version", "v1")

    print(f"[RESET] schema={schema} task={task}", flush=True)

    # ── Ask LLM for action + reply ─────────────────────────────────
    completion = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"World State:\n{world_state}"}
        ],
        max_tokens=200,
        temperature=0.3
    )

    llm_output = completion.choices[0].message.content
    action, reply = parse_llm_output(llm_output)

    print(f"[LLM] action={action} reply_preview={reply[:50]!r}", flush=True)

    # ── Step 1: send action ────────────────────────────────────────
    step1 = requests.post(
        f"{ENV_URL}/step",
        json={"action": action, "reply": "", "step": 1}
    )
    step1.raise_for_status()
    s1 = step1.json()
    print(f"[STEP1] done={s1['done']} message={s1['observation']['message']!r}", flush=True)

    # ── Step 2: send reply, get final reward ───────────────────────
    step2 = requests.post(
        f"{ENV_URL}/step",
        json={"action": action, "reply": reply, "step": 2}
    )
    step2.raise_for_status()
    s2 = step2.json()

    reward = float(s2["reward"])
    reward = min(max(reward, 0.01), 0.99)
    rewards.append(reward)

    info = s2.get("info", {})
    ideal = info.get("ideal_action", "?")
    correct = info.get("action_correct", False)

    print(
        f"[STEP2] action={action} ideal={ideal} correct={correct} "
        f"reward={reward:.2f} done={s2['done']}",
        flush=True
    )

    score = reward
    success = score >= 0.60
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(
        f"[END] success={str(success).lower()} steps=2 score={score:.2f} rewards={rewards_str}",
        flush=True
    )

    return reward


async def main():
    if HF_TOKEN is None:
        raise RuntimeError(
            "Missing HF_TOKEN or API_KEY. "
            "Set it with: export HF_TOKEN=hf_xxxx"
        )

    client = AsyncOpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    all_rewards = []
    for task in TASKS:
        r = await run_task(client, task)
        all_rewards.append(r)

    avg = sum(all_rewards) / len(all_rewards)
    print(f"\n[SUMMARY] tasks={len(TASKS)} avg_reward={avg:.3f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
