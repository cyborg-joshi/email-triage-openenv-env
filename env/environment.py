import random
from env.models import ExecutiveObservation, ExecutiveAction
from env.graders import compute_reward
from env.scenarios import SCENARIOS, SCHEMAS


class ExecutiveAssistantEnv:

    def __init__(self):
        self.episode_count = 0
        self.current_schema_key = "v1"
        self.current_scenario_key = None
        self.current_scenario = None
        self.step_number = 0
        self.chosen_action = None
        self.done = False

    def _get_schema_key(self):
        if self.episode_count <= 10:
            return "v1"
        elif self.episode_count <= 20:
            return "v2"
        else:
            return "v3"

    def reset(self, task: str = None):
        self.episode_count += 1
        self.current_schema_key = self._get_schema_key()
        self.step_number = 1
        self.chosen_action = None
        self.done = False

        if task and task in SCENARIOS:
            self.current_scenario_key = task
        else:
            self.current_scenario_key = random.choice(list(SCENARIOS.keys()))

        self.current_scenario = SCENARIOS[self.current_scenario_key]
        schema = SCHEMAS[self.current_schema_key]

        return ExecutiveObservation(
            world_state={
                "emails": self.current_scenario["emails"],
                "calendar": self.current_scenario["calendar"],
                "pending_tasks": self.current_scenario["pending_tasks"],
                "schema_version": self.current_schema_key,
                "schema_name": schema["name"],
                "context": self.current_scenario["context"]
            },
            last_action="",
            message=(
                f"Episode {self.episode_count} started. "
                f"Schema: {schema['name']} ({self.current_schema_key}). "
                f"Step 1: Choose your action — reply, escalate, delegate, reschedule, ignore."
            ),
            step_number=1
        )

    def step(self, action: ExecutiveAction):
        if self.done:
            return (
                ExecutiveObservation(
                    world_state={},
                    last_action=action.action,
                    message="Episode already finished. Call /reset to start a new one.",
                    step_number=0
                ),
                0.01,
                True,
                {"episode": self.episode_count, "schema": self.current_schema_key}
            )

        schema = SCHEMAS[self.current_schema_key]

        if self.step_number == 1:
            valid_actions = ["reply", "escalate", "delegate", "reschedule", "ignore"]
            self.chosen_action = action.action.lower().strip()
            if self.chosen_action not in valid_actions:
                self.chosen_action = "reply"

            self.step_number = 2

            return (
                ExecutiveObservation(
                    world_state={
                        "emails": self.current_scenario["emails"],
                        "calendar": self.current_scenario["calendar"],
                        "pending_tasks": self.current_scenario["pending_tasks"],
                        "schema_version": self.current_schema_key,
                        "schema_name": schema["name"],
                        "context": self.current_scenario["context"],
                        "your_chosen_action": self.chosen_action
                    },
                    last_action=self.chosen_action,
                    message=(
                        f"Action '{self.chosen_action}' recorded. "
                        f"Step 2: Write your reply. "
                        f"Schema {self.current_schema_key} requires: "
                        f"{schema['tone']} tone, max {schema['max_reply_words']} words."
                    ),
                    step_number=2
                ),
                0.0,
                False,
                {
                    "step": 1,
                    "action_recorded": self.chosen_action,
                    "schema": self.current_schema_key
                }
            )

        elif self.step_number == 2:
            reply_text = (action.reply or "").strip()
            ideal_action = self.current_scenario["ideal_action"][self.current_schema_key]
            ideal_keywords = self.current_scenario["ideal_reply_keywords"]
            conflict = self.current_scenario.get("conflict", False)
            conflict_keywords = self.current_scenario.get("conflict_keywords", [])

            reply_lower = reply_text.lower()
            conflict_addressed = (
                any(kw in reply_lower for kw in conflict_keywords)
                if conflict_keywords else False
            )

            reward = compute_reward(
                action=self.chosen_action,
                ideal_action=ideal_action,
                reply_text=reply_text,
                ideal_keywords=ideal_keywords,
                conflict_in_email=conflict,
                reply_addresses_conflict=conflict_addressed,
                schema=schema
            )

            self.done = True

            return (
                ExecutiveObservation(
                    world_state={},
                    last_action=reply_text[:80] + ("..." if len(reply_text) > 80 else ""),
                    message="Episode complete.",
                    step_number=3
                ),
                reward,
                True,
                {
                    "episode": self.episode_count,
                    "schema": self.current_schema_key,
                    "schema_name": schema["name"],
                    "task": self.current_scenario_key,
                    "chosen_action": self.chosen_action,
                    "ideal_action": ideal_action,
                    "action_correct": self.chosen_action == ideal_action,
                    "reward": reward
                }
            )

        return (
            ExecutiveObservation(
                world_state={},
                last_action="",
                message="Unknown step. Call /reset.",
                step_number=0
            ),
            0.01,
            True,
            {}
        )

    def state(self):
        schema = SCHEMAS.get(self.current_schema_key, {})
        return {
            "episode_count": self.episode_count,
            "current_schema": self.current_schema_key,
            "schema_name": schema.get("name", ""),
            "current_task": self.current_scenario_key,
            "step_number": self.step_number,
            "done": self.done
        }
