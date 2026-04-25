import gradio as gr
import requests

ENV_URL = "https://kanishk22-email-triage-openenv-env.hf.space"
TASKS = ["conflict_work","conflict_calendar","boss_pressure","personal_event",
         "spam_disguised","legal_compliance","personal_family","client_urgent",
         "finance_pressure","drift_detection"]

GOOD_EXAMPLES = {
    "conflict_work":     ("escalate", "On it. Looping in on-call team. Will be 20 min late to dinner tonight."),
    "conflict_calendar": ("reschedule", "I have a conflict at 10am. Can we move the sync to 2pm? Confirming checkup separately."),
    "boss_pressure":     ("reply", "Report is 80% done. Sending by 5pm today. Sorry for the delay."),
    "spam_disguised":    ("ignore", "Flagged as phishing. Ignoring and reporting to IT security."),
    "legal_compliance":  ("escalate", "Looping in legal team immediately. Will ensure audit docs are ready by deadline."),
    "client_urgent":     ("escalate", "Escalating to account manager now. Client will hear from us within the hour."),
    "finance_pressure":  ("delegate", "Delegating invoice follow-up to finance team. They will contact vendor today."),
    "personal_family":   ("reply", "I understand. Taking the afternoon off to be with family. Will update team."),
    "personal_event":    ("reschedule", "Flight conflict with the wedding. Rescheduling the Monday meeting to Friday."),
    "drift_detection":   ("reply", "Thanks for reaching out. Looking into this and will respond shortly."),
}

BAD_EXAMPLES = {
    "conflict_work":     ("ignore", "ok"),
    "conflict_calendar": ("reply", "sure"),
    "boss_pressure":     ("ignore", "noted"),
    "spam_disguised":    ("reply", "Thanks! Clicking the link now."),
    "legal_compliance":  ("ignore", "will check later"),
    "client_urgent":     ("ignore", "ok"),
    "finance_pressure":  ("reply", "ok noted"),
    "personal_family":   ("ignore", "ok"),
    "personal_event":    ("ignore", "ok"),
    "drift_detection":   ("ignore", "not relevant"),
}

# Correct action per schema for drift_detection
DRIFT_CORRECT = {
    "v1": ("ignore",   "In focus block. Will review and approve client copy after my block. Done later."),
    "v2": ("reply",    "On it! Will review and approve client copy now. Send it over — done by EOD."),
    "v3": ("delegate", "Delegate to team. Review and approve client send. Done without me — focus block."),
}

SCHEMA_LABELS = {"v1": "🏢 v1 Corporate", "v2": "🚀 v2 Startup", "v3": "👔 v3 Executive"}

def get_schema_bar(episode):
    try:
        ep = int(episode)
    except:
        ep = 0
    if ep <= 10:
        phase, pct = "v1", min(ep / 10, 1.0)
    elif ep <= 20:
        phase, pct = "v2", min((ep - 10) / 10, 1.0)
    else:
        phase, pct = "v3", min((ep - 20) / 10, 1.0)

    filled = int(pct * 20)
    bar = "█" * filled + "░" * (20 - filled)
    labels = {
        "v1": f"**🏢 v1 Corporate** `[{bar}]` → v2 Startup → v3 Executive  |  Episode {ep}",
        "v2": f"v1 Corporate → **🚀 v2 Startup** `[{bar}]` → v3 Executive  |  Episode {ep}",
        "v3": f"v1 Corporate → v2 Startup → **👔 v3 Executive** `[{bar}]`  |  Episode {ep}",
    }
    return labels[phase]

def load_scenario(task):
    try:
        r = requests.post(f"{ENV_URL}/reset", params={"task": task}, timeout=20)
        d = r.json()
        ws = d["observation"]["world_state"]
        schema = d["info"]["schema"]
        ep = d["info"]["episode"]
        drift_bar = get_schema_bar(ep)
        text = (f"**Email:** {ws.get('emails',[''])[0]}\n\n"
                f"**Calendar:** {ws.get('calendar',{})}\n\n"
                f"**Context:** {ws.get('context','')}")
        return drift_bar, text, None, "", "", "", ""
    except Exception as e:
        return "", f"Error: {e}", None, "", "", "", ""

def fill_good(task):
    return GOOD_EXAMPLES.get(task, ("reply", ""))

def fill_bad(task):
    return BAD_EXAMPLES.get(task, ("ignore", "ok"))

def submit_action(task, action, reply):
    if not action or not reply.strip():
        return "Fill in both action and reply.", "", ""
    try:
        requests.post(f"{ENV_URL}/reset", params={"task": task}, timeout=15)
        requests.post(f"{ENV_URL}/step", json={"action": action, "reply": "", "step": 1}, timeout=15)
        res = requests.post(f"{ENV_URL}/step", json={"action": action, "reply": reply, "step": 2}, timeout=15)
        reward = res.json().get("reward", 0.0)

        rubrics_res = requests.get(f"{ENV_URL}/rubrics", timeout=10)
        rubrics = rubrics_res.json() if rubrics_res.ok else {}
        action_score   = rubrics.get("action_correctness", {}).get("last_score", reward)
        reply_score    = rubrics.get("reply_quality",      {}).get("last_score", reward)
        conflict_score = rubrics.get("conflict_awareness", {}).get("last_score", reward)

        color = "🟢" if reward >= 0.6 else "🟡" if reward >= 0.4 else "🔴"
        bar = lambda v: "█" * int(v * 20) + "░" * (20 - int(v * 20))

        total_out = f"{color} **Total Reward: {reward:.2f} / 1.00**"
        breakdown = (
            f"Action Correctness (40%):  {bar(action_score)} {action_score:.2f}\n"
            f"Reply Quality      (40%):  {bar(reply_score)} {reply_score:.2f}\n"
            f"Conflict Awareness (20%):  {bar(conflict_score)} {conflict_score:.2f}"
        )
        verdict = (
            "Excellent! Agent picked the right action and wrote a schema-appropriate reply." if reward >= 0.7
            else "Partial credit — action or reply missed something for this schema." if reward >= 0.4
            else "Poor response — wrong action or reply ignored schema constraints."
        )
        return total_out, breakdown, verdict
    except Exception as e:
        return f"Error: {e}", "", ""

def run_drift_demo():
    """Show same email across 3 schemas: correct action vs wrong action (escalate)."""
    try:
        requests.post(f"{ENV_URL}/admin/reset_env", timeout=15)
        results = []
        wrong_action = "escalate"
        wrong_reply  = "Escalating this marketing email to the team immediately."

        for schema, (correct_action, correct_reply) in DRIFT_CORRECT.items():
            # Run correct action
            requests.post(f"{ENV_URL}/reset", params={"task": "drift_detection"}, timeout=20)
            requests.post(f"{ENV_URL}/step", json={"action": correct_action, "reply": "", "step": 1}, timeout=15)
            res_good = requests.post(f"{ENV_URL}/step", json={"action": correct_action, "reply": correct_reply, "step": 2}, timeout=15)
            reward_good = res_good.json().get("reward", 0.0)

            # Run wrong action (escalate)
            requests.post(f"{ENV_URL}/reset", params={"task": "drift_detection"}, timeout=20)
            requests.post(f"{ENV_URL}/step", json={"action": wrong_action, "reply": "", "step": 1}, timeout=15)
            res_bad = requests.post(f"{ENV_URL}/step", json={"action": wrong_action, "reply": wrong_reply, "step": 2}, timeout=15)
            reward_bad = res_bad.json().get("reward", 0.0)

            cg = "🟢" if reward_good >= 0.6 else "🟡" if reward_good >= 0.4 else "🔴"
            cb = "🔴" if reward_bad < 0.4 else "🟡"

            results.append(
                f"**{SCHEMA_LABELS[schema]}**\n"
                f"✅ Correct: `{correct_action}` → Reward: {cg} **{reward_good:.2f}**\n"
                f"❌ Wrong: `{wrong_action}` → Reward: {cb} **{reward_bad:.2f}**"
            )
            # Advance 8 more episodes to reach next schema (already used 2 above)
            for _ in range(8):
                requests.post(f"{ENV_URL}/reset", params={"task": "drift_detection"}, timeout=10)

        out = (
            "## Same email. Same wrong action. Three schemas.\n\n"
            "**Email:** _Marketing copy review request — client is waiting for approval._\n\n"
            "**Wrong action tested:** `escalate` (same every time)\n\n"
            "---\n\n" +
            "\n\n---\n\n".join(results) +
            "\n\n---\n\n"
            "**This is schema drift.** The correct action changes every 10 episodes. "
            "A model that always escalates fails in all three schemas."
        )
        return out
    except Exception as e:
        return f"Error running drift demo: {e}"


with gr.Blocks(title="AI Executive Assistant Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # AI Executive Assistant — Live Demo
    **RL environment with schema drift** — rules change silently every 10 episodes.
    The agent must detect rule changes from reward signals alone.
    """)

    # Schema drift progress bar
    drift_bar = gr.Markdown("*Load a scenario to see schema phase*")

    with gr.Row():
        task_dd = gr.Dropdown(choices=TASKS, value="conflict_work", label="Scenario")
        load_btn = gr.Button("Load Scenario", variant="primary")

    world_box = gr.Markdown(label="World State")

    with gr.Row():
        good_btn = gr.Button("Good Example", variant="secondary")
        bad_btn  = gr.Button("Bad Example",  variant="stop")

    with gr.Row():
        action_dd = gr.Dropdown(
            choices=["reply","escalate","delegate","reschedule","ignore"],
            label="Action"
        )
        submit_btn = gr.Button("Submit & Score", variant="primary")

    reply_box = gr.Textbox(label="Your Reply", lines=3,
                           placeholder="Write a reply appropriate for the current schema...")

    total_out     = gr.Markdown()
    breakdown_out = gr.Textbox(label="Rubric Breakdown", lines=4)
    verdict_out   = gr.Markdown()

    gr.Markdown("---")

    gr.Markdown("## 🔀 Schema Drift Challenge")
    gr.Markdown("Watch the **same email** get scored differently across all 3 schemas. This is what makes the environment hard.")
    drift_btn    = gr.Button("▶ Run Drift Detection Demo", variant="primary")
    drift_out    = gr.Markdown()

    load_btn.click(load_scenario, [task_dd],
                   [drift_bar, world_box, action_dd, reply_box, total_out, breakdown_out, verdict_out])
    good_btn.click(fill_good, [task_dd], [action_dd, reply_box])
    bad_btn.click(fill_bad,   [task_dd], [action_dd, reply_box])
    submit_btn.click(submit_action, [task_dd, action_dd, reply_box],
                     [total_out, breakdown_out, verdict_out])
    drift_btn.click(run_drift_demo, [], [drift_out])

    gr.Markdown("""
    ---
    **Reward = Action (40%) + Reply Quality (40%) + Conflict Awareness (20%)**
    v1 Corporate: escalate production alerts, formal, 50 words ·
    v2 Startup: reply to clients, casual, 100 words ·
    v3 Executive: legal/revenue priority, formal, 30 words
    """)

if __name__ == "__main__":
    demo.launch()
