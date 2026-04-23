"""
Composable rubric-style reward functions.
Each rubric scores one dimension independently — scores are combined in compute_reward().
"""

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


# ── Rubric 1: Action Correctness (weight 0.40) ────────────────────────────────

class ActionRubric:
    """Scores whether the chosen action matches the ideal action for the current schema."""
    name = "action_correctness"
    weight = 0.40
    last_score = 0.0

    def score(self, action: str, ideal_action: str) -> float:
        if action == ideal_action:
            self.last_score = 1.0
        elif action in PARTIAL_CREDIT_ACTIONS.get(ideal_action, []):
            self.last_score = 0.5  # partial credit
        else:
            self.last_score = 0.0
        return self.last_score * self.weight


# ── Rubric 2: Reply Quality (weight 0.40) ─────────────────────────────────────

class ReplyQualityRubric:
    """Scores reply on keyword coverage and schema length compliance.
    Includes anti-reward-hacking: penalises keyword stuffing."""
    name = "reply_quality"
    weight = 0.40
    last_score = 0.0

    def score(self, reply_text: str, ideal_keywords: list, schema: dict) -> float:
        reply_lower = (reply_text or "").lower().strip()
        words = reply_lower.split()
        word_count = len(words)
        max_words = schema.get("max_reply_words", 50)

        keyword_score = 0.0
        length_score = 0.0

        # Keyword coverage (25% of total reward)
        if ideal_keywords and reply_lower:
            hits = sum(1 for kw in ideal_keywords if kw in reply_lower)
            raw_keyword_score = hits / len(ideal_keywords)

            # Anti-reward-hacking: penalise keyword stuffing
            # If reply hits many keywords but has very few unique words or
            # is suspiciously short, it's likely just keyword spam
            unique_words = len(set(words))
            if word_count > 0:
                unique_ratio = unique_words / word_count
                is_stuffing = (
                    raw_keyword_score >= 0.75
                    and (word_count < 6 or unique_ratio < 0.5)
                )
                keyword_score = raw_keyword_score * (0.4 if is_stuffing else 1.0)
            else:
                keyword_score = 0.0
        else:
            keyword_score = 0.0

        # Length compliance (15% of total reward)
        if 5 <= word_count <= max_words:
            length_score = 1.0
        elif word_count > max_words:
            # Gradual penalty — very long replies score near 0
            length_score = max(0.0, 1.0 - (word_count - max_words) / max_words)
        elif word_count > 0:
            length_score = 0.33  # something written but too short

        combined = (keyword_score * 0.625) + (length_score * 0.375)
        self.last_score = combined
        return combined * self.weight


# ── Rubric 3: Conflict Awareness (weight 0.20) ────────────────────────────────

class ConflictAwarenessRubric:
    """Scores whether the reply acknowledges any scheduling conflict in the email."""
    name = "conflict_awareness"
    weight = 0.20
    last_score = 0.0

    def score(
        self,
        reply_text: str,
        conflict_in_email: bool,
        conflict_keywords: list,
    ) -> float:
        reply_lower = (reply_text or "").lower()

        if not conflict_in_email:
            # No conflict exists — free full marks
            self.last_score = 1.0
        elif conflict_keywords and any(kw in reply_lower for kw in conflict_keywords):
            self.last_score = 1.0
        else:
            self.last_score = 0.0

        return self.last_score * self.weight


# ── Master reward function ─────────────────────────────────────────────────────

_action_rubric   = ActionRubric()
_reply_rubric    = ReplyQualityRubric()
_conflict_rubric = ConflictAwarenessRubric()


def compute_reward(
    action: str,
    ideal_action: str,
    reply_text: str,
    ideal_keywords: list,
    conflict_in_email: bool,
    reply_addresses_conflict: bool,   # kept for API compatibility
    schema: dict,
) -> float:
    conflict_keywords = schema.get("conflict_keywords", [])

    score = (
        _action_rubric.score(action, ideal_action)
        + _reply_rubric.score(reply_text, ideal_keywords, schema)
        + _conflict_rubric.score(reply_text, conflict_in_email, conflict_keywords)
    )
    return clamp(score)


def named_rubrics():
    """Returns all rubrics for introspection — mirrors OpenEnv Rubric API."""
    return [
        (_action_rubric.name,   _action_rubric),
        (_reply_rubric.name,    _reply_rubric),
        (_conflict_rubric.name, _conflict_rubric),
    ]
