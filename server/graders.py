def clamp(score):
    if score is None:
        score = 0.5
    return min(max(float(score), 0.01), 0.99)


def trajectory_grader(trajectory=None):
    """
    Macro-grader: scores the full training run.
    Receives a list of episode rewards collected across all episodes.
    Returns a single float representing overall agent performance.

    Example trajectory: [0.2, 0.3, 0.25, 0.6, 0.7, 0.85, 0.9]
    This shows an agent that started bad and improved — exactly what we want to see.
    """
    trajectory = trajectory or []

    if len(trajectory) == 0:
        return 0.50

    # Mean reward across all episodes
    mean_reward = sum(trajectory) / len(trajectory)

    return clamp(mean_reward)


def schema_drift_grader(trajectory=None):
    """
    Bonus macro-grader specific to schema drift.
    Checks if agent recovered quickly after each drift event.

    Episodes 1-10   = v1 (Corporate)
    Episodes 11-20  = v2 (Startup)    ← drift happens here
    Episodes 21-30  = v3 (Executive)  ← drift happens here

    A good agent:
      - Learns v1 well (high rewards by episode 10)
      - Dips at episode 11 (expected — new rules)
      - Recovers by episode 15 (adapts to v2)
      - Dips again at episode 21 (expected — new rules)
      - Recovers faster this time (agent learns to adapt)
    """
    trajectory = trajectory or []

    if len(trajectory) < 10:
        return clamp(sum(trajectory) / len(trajectory)) if trajectory else 0.50

    scores = {}

    # Score each schema phase separately
    v1_episodes = trajectory[:10]
    scores["v1_mean"] = sum(v1_episodes) / len(v1_episodes)

    if len(trajectory) >= 20:
        v2_episodes = trajectory[10:20]
        scores["v2_mean"] = sum(v2_episodes) / len(v2_episodes)

        # Recovery speed: did v2 mean beat the dip at episode 11?
        v2_recovery = trajectory[14:20]  # last 6 episodes of v2
        scores["v2_late_mean"] = sum(v2_recovery) / len(v2_recovery)

    if len(trajectory) >= 30:
        v3_episodes = trajectory[20:30]
        scores["v3_mean"] = sum(v3_episodes) / len(v3_episodes)

        v3_recovery = trajectory[24:30]
        scores["v3_late_mean"] = sum(v3_recovery) / len(v3_recovery)

    # Overall: mean of all phase means
    all_phase_means = [v for k, v in scores.items() if "mean" in k]
    overall = sum(all_phase_means) / len(all_phase_means) if all_phase_means else 0.50

    return clamp(overall)


def corporate_grader(trajectory=None):
    """Episodes 1-10 (v1 Corporate Mode) performance."""
    trajectory = trajectory or []
    relevant = trajectory[:10]
    if not relevant:
        return 0.50
    return clamp(sum(relevant) / len(relevant))


def startup_grader(trajectory=None):
    """Episodes 11-20 (v2 Startup Mode) performance."""
    trajectory = trajectory or []
    relevant = trajectory[10:20]
    if not relevant:
        return 0.50
    return clamp(sum(relevant) / len(relevant))


def executive_grader(trajectory=None):
    """Episodes 21-30 (v3 Executive Mode) performance."""
    trajectory = trajectory or []
    relevant = trajectory[20:30]
    if not relevant:
        return 0.50
    return clamp(sum(relevant) / len(relevant))
