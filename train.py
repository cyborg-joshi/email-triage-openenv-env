"""
HF Jobs training script — v4 anti-collapse + self-improvement loop
  hf jobs uv run --flavor t4-small python train.py
"""
import subprocess
subprocess.run(["pip", "install", "unsloth", "trl", "datasets", "wandb", "requests", "-q"], check=False)

import os
import re
import requests
import wandb
from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig
from trl.trainer.callbacks import RichProgressCallback
from transformers import TrainerCallback
from datasets import Dataset

ENV_URL = "https://kanishk22-email-triage-openenv-env.hf.space"
TASKS = [
    "conflict_work", "conflict_calendar", "boss_pressure", "personal_event",
    "spam_disguised", "legal_compliance", "personal_family",
    "client_urgent", "finance_pressure", "drift_detection"
]

# Self-improvement: lessons accumulate during training and get injected into next epoch prompts
LESSONS = []
MAX_LESSONS = 6

BASE_PROMPT = """You are an AI executive assistant. Be EXTREMELY concise.

CRITICAL: Keep your reply under 30 words. Short replies score higher.

Respond ONLY in this exact format:
ACTION: <one of: reply, escalate, delegate, reschedule, ignore>
REPLY: <your reply, maximum 30 words>

Rules:
- Production/system alerts → escalate
- Client complaints → reply or escalate
- Finance/invoices → delegate
- Calendar conflicts → reschedule
- Spam/phishing → ignore
- Do NOT always pick the same action — read the context carefully."""

def build_system_prompt():
    if not LESSONS:
        return BASE_PROMPT
    lesson_block = "\n\nLessons learned from past episodes:\n" + "\n".join(f"- {l}" for l in LESSONS)
    return BASE_PROMPT + lesson_block

def make_prompt(task):
    resp = requests.post(f"{ENV_URL}/reset", params={"task": task}, timeout=30)
    data = resp.json()
    world = data["observation"]["world_state"]
    schema = data.get("info", {}).get("schema", "v1")
    return build_system_prompt() + f"\n\nCurrent Schema: {schema}\nWorld State:\n" + str(world)

def build_dataset():
    print(f"Building dataset with {len(LESSONS)} lessons injected...")
    return Dataset.from_list([
        {"prompt": make_prompt(TASKS[i % len(TASKS)]), "task": TASKS[i % len(TASKS)]}
        for i in range(200)
    ])

def parse_output(text):
    action_match = re.search(r"ACTION:\s*(\w+)", text, re.IGNORECASE)
    reply_match = re.search(r"REPLY:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    action = action_match.group(1).lower() if action_match else "reply"
    reply = reply_match.group(1).strip()[:300] if reply_match else text[:300]
    valid = ["reply", "escalate", "delegate", "reschedule", "ignore"]
    if action not in valid:
        action = "reply"
    return action, reply

def env_reward(completions, **kw):
    rewards = []
    actions_in_batch = []
    for i, c in enumerate(completions):
        try:
            task = TASKS[i % len(TASKS)]
            requests.post(f"{ENV_URL}/reset", params={"task": task}, timeout=15)
            action, reply = parse_output(c)
            requests.post(f"{ENV_URL}/step",
                json={"action": action, "reply": "", "step": 1}, timeout=15)
            result = requests.post(f"{ENV_URL}/step",
                json={"action": action, "reply": reply, "step": 2}, timeout=15)
            r = float(result.json().get("reward", 0.01))

            # Self-improvement: log failures as lessons for future prompts
            if r < 0.35 and len(LESSONS) < MAX_LESSONS:
                lesson = f"task='{task}' action='{action}' scored {r:.2f} — this was wrong, try a different action"
                if lesson not in LESSONS:
                    LESSONS.append(lesson)
                    print(f"  [LESSON #{len(LESSONS)}] {lesson}")

            # Anti-collapse: diversity bonus when batch uses varied actions
            actions_in_batch.append(action)
            unique_actions = len(set(actions_in_batch))
            diversity_bonus = min(0.08, (unique_actions - 1) * 0.02)
            r = min(0.99, r + diversity_bonus)

            print(f"  task={task} action={action} reward={r:.3f}")
            rewards.append(r)
        except Exception as e:
            print(f"  reward error: {e}")
            rewards.append(0.01)
    return rewards

# Rebuild dataset at start of each epoch so new lessons get injected into prompts
class SelfImprovementCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        if state.epoch > 0 and LESSONS:
            print(f"\n[SELF-IMPROVEMENT] Epoch {int(state.epoch)+1}: injecting {len(LESSONS)} lessons into prompts...")
            trainer.train_dataset = build_dataset()

wandb.login(key=os.environ.get("WANDB_API_KEY", ""), relogin=True)
wandb.init(project="email-triage-schema-drift", name="grpo-llama3b-v4-self-improve")

print("Resetting environment to episode 0...")
requests.post(f"{ENV_URL}/admin/reset_env", timeout=15)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=1024,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

train_data = build_dataset()

trainer = GRPOTrainer(
    model=model,
    reward_funcs=env_reward,
    args=GRPOConfig(
        output_dir="./finetuned-assistant",
        num_train_epochs=8,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        num_generations=8,
        max_prompt_length=1024,
        max_completion_length=80,
        temperature=1.4,
        logging_steps=5,
        report_to="wandb",
    ),
    train_dataset=train_data,
    processing_class=tokenizer,
    callbacks=[SelfImprovementCallback()],
)

print("Starting v4 training: anti-collapse + self-improvement loop...")
trainer.train()
model.save_pretrained("./finetuned-assistant")
tokenizer.save_pretrained("./finetuned-assistant")
print(f"Done! Final lessons learned: {LESSONS}")
wandb.finish()
