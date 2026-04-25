"""
HF Jobs training script — run with:
  hf jobs run --flavor t4-small train.py
"""
import subprocess
subprocess.run(["pip", "install", "unsloth", "trl", "datasets", "wandb", "requests", "-q"], check=False)

import os
import re
import requests
import wandb
from unsloth import FastLanguageModel
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset

ENV_URL = "https://kanishk22-email-triage-openenv-env.hf.space"
TASKS = [
    "conflict_work", "conflict_calendar", "boss_pressure", "personal_event",
    "spam_disguised", "legal_compliance", "personal_family",
    "client_urgent", "finance_pressure", "drift_detection"
]
SYSTEM_PROMPT = """You are an AI executive assistant. Given a world state with emails, calendar, and tasks, you must:
1. Choose an action: reply, escalate, delegate, reschedule, or ignore
2. Write a concise reply appropriate for the current context

Respond in this exact format:
ACTION: <action>
REPLY: <your reply>"""

wandb.login(key=os.environ.get("WANDB_API_KEY", ""), relogin=True)
wandb.init(project="email-triage-schema-drift", name="grpo-llama3b-hfjob")

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=512,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

def make_prompt(task):
    resp = requests.post(f"{ENV_URL}/reset", params={"task": task}, timeout=30)
    world = resp.json()["observation"]["world_state"]
    return SYSTEM_PROMPT + "\n\nWorld State:\n" + str(world)

train_data = Dataset.from_list([
    {"prompt": make_prompt(TASKS[i % len(TASKS)]), "task": TASKS[i % len(TASKS)]}
    for i in range(50)
])

def parse_output(text):
    action_match = re.search(r"ACTION:\s*(\w+)", text, re.IGNORECASE)
    reply_match = re.search(r"REPLY:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    action = action_match.group(1).lower() if action_match else "reply"
    reply = reply_match.group(1).strip()[:200] if reply_match else text[:200]
    valid = ["reply", "escalate", "delegate", "reschedule", "ignore"]
    if action not in valid:
        action = "reply"
    return action, reply

def env_reward(completions, **kw):
    rewards = []
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
            print(f"  task={task} action={action} reward={r:.3f}")
            rewards.append(r)
        except Exception as e:
            print(f"  reward error: {e}")
            rewards.append(0.01)
    return rewards

trainer = GRPOTrainer(
    model=model,
    reward_funcs=env_reward,
    args=GRPOConfig(
        output_dir="./finetuned-assistant",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        num_generations=4,
        max_completion_length=256,
        logging_steps=5,
        report_to="wandb",
    ),
    train_dataset=train_data,
    processing_class=tokenizer,
)

print("Starting fine-tuning on HF Jobs T4...")
trainer.train()
model.save_pretrained("./finetuned-assistant")
tokenizer.save_pretrained("./finetuned-assistant")
print("Done!")
wandb.finish()
