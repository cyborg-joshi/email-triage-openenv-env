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

def load_scenario(task):
    try:
        r = requests.post(f"{ENV_URL}/reset", params={"task": task}, timeout=20)
        d = r.json(); ws = d["observation"]["world_state"]
        schema = d["info"]["schema"]; ep = d["info"]["episode"]
        schemas = {"v1":"🏢 Corporate","v2":"🚀 Startup","v3":"👔 Executive"}
        text = (f"**Schema:** {schemas.get(schema,schema)}  |  **Episode:** {ep}\n\n"
                f"**Email:** {ws.get('emails',[''])[0]}\n\n"
                f"**Calendar:** {ws.get('calendar',{})}\n\n"
                f"**Context:** {ws.get('context','')}")
        return text, None, "", "", "", ""
    except Exception as e:
        return f"Error: {e}", None, "", "", "", ""

def fill_good(task):
    action, reply = GOOD_EXAMPLES.get(task, ("reply", ""))
    return action, reply

def fill_bad(task):
    action, reply = BAD_EXAMPLES.get(task, ("ignore", "ok"))
    return action, reply

def submit_action(task, action, reply):
    if not action or not reply.strip():
        return "Fill in both action and reply.", "", ""
    try:
        requests.post(f"{ENV_URL}/reset", params={"task": task}, timeout=15)
        requests.post(f"{ENV_URL}/step", json={"action":action,"reply":"","step":1}, timeout=15)
        res = requests.post(f"{ENV_URL}/step", json={"action":action,"reply":reply,"step":2}, timeout=15)
        reward = res.json().get("reward", 0.0)

        # Get real per-component scores from /rubrics
        rubrics_res = requests.get(f"{ENV_URL}/rubrics", timeout=10)
        rubrics = rubrics_res.json() if rubrics_res.ok else {}

        action_score   = rubrics.get("action_correctness", {}).get("last_score", reward)
        reply_score    = rubrics.get("reply_quality",      {}).get("last_score", reward)
        conflict_score = rubrics.get("conflict_awareness", {}).get("last_score", reward)

        color = "🟢" if reward >= 0.6 else "🟡" if reward >= 0.4 else "🔴"
        bar = lambda v: "█"*int(v*20) + "░"*(20-int(v*20))

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

with gr.Blocks(title="AI Executive Assistant Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # AI Executive Assistant — Live Demo
    **RL environment with schema drift** — rules change silently every 10 episodes.
    The agent must detect rule changes from reward signals alone.
    """)

    with gr.Row():
        task_dd = gr.Dropdown(choices=TASKS, value="conflict_work", label="Scenario")
        load_btn = gr.Button("Load Scenario", variant="primary")

    world_box = gr.Markdown(label="World State")

    with gr.Row():
        good_btn = gr.Button("Good Example", variant="secondary")
        bad_btn  = gr.Button("Bad Example", variant="stop")

    with gr.Row():
        action_dd = gr.Dropdown(
            choices=["reply","escalate","delegate","reschedule","ignore"],
            label="Action"
        )
        submit_btn = gr.Button("Submit & Score", variant="primary")

    reply_box = gr.Textbox(label="Your Reply", lines=3,
                           placeholder="Write a reply appropriate for the current schema...")

    total_out    = gr.Markdown()
    breakdown_out = gr.Textbox(label="Rubric Breakdown", lines=5)
    verdict_out  = gr.Markdown()

    load_btn.click(load_scenario, [task_dd],
                   [world_box, action_dd, reply_box, total_out, breakdown_out, verdict_out])
    good_btn.click(fill_good, [task_dd], [action_dd, reply_box])
    bad_btn.click(fill_bad,  [task_dd], [action_dd, reply_box])
    submit_btn.click(submit_action, [task_dd, action_dd, reply_box],
                     [total_out, breakdown_out, verdict_out])

    gr.Markdown("""
    ---
    **Reward = Action (40%) + Reply Quality (40%) + Conflict Awareness (20%)**
    v1 Corporate: escalate production alerts, formal, 50 words ·
    v2 Startup: reply to clients, casual, 100 words ·
    v3 Executive: legal/revenue priority, formal, 30 words
    """)

if __name__ == "__main__":
    demo.launch()
