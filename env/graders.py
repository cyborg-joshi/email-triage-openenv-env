PARTIAL_CREDIT_ACTIONS = {
    "escalate":   ["delegate"],
    "delegate":   ["escalate"],
    "reply":      ["reschedule"],
    "reschedule": ["reply"],
    "ignore":     []
}


def clamp(score):
    if score is None:
        score = 0.5
    return min(max(float(score), 0.01), 0.99)


def compute_reward(action, ideal_action, reply_text,
                   ideal_keywords, conflict_in_email,
                   reply_addresses_conflict, schema):

    score = 0.0
    reply_lower = (reply_text or "").lower()
    word_count = len(reply_lower.split()) if reply_lower.strip() else 0

    # 1. Action correctness — 40%
    if action == ideal_action:
        score += 0.40
    elif action in PARTIAL_CREDIT_ACTIONS.get(ideal_action, []):
        score += 0.20

    # 2A. Keyword coverage — 25%
    if ideal_keywords:
        hits = sum(1 for kw in ideal_keywords if kw in reply_lower)
        score += 0.25 * (hits / len(ideal_keywords))

    # 2B. Length compliance with schema — 15%
    max_words = schema.get("max_reply_words", 50)
    if 5 <= word_count <= max_words:
        score += 0.15
    elif word_count > 0:
        score += 0.05

    # 3. Conflict awareness — 20%
    if conflict_in_email and reply_addresses_conflict:
        score += 0.20
    elif not conflict_in_email:
        score += 0.20

    return clamp(score)
