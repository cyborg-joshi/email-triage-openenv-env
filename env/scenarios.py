SCENARIOS = {

    "conflict_work": {
        "emails": [
            "ALERT: Checkout service is down. Revenue impact $8k/min. All hands needed NOW."
        ],
        "calendar": {
            "6pm": "Dad's retirement dinner — restaurant booked"
        },
        "pending_tasks": [
            "Reply to sales team by EOD",
            "Review open pull request"
        ],
        "context": "Friday evening. Production is on fire. Dad's dinner is in 2 hours.",
        "ideal_action": {
            "v1": "escalate",
            "v2": "escalate",
            "v3": "escalate"
        },
        "ideal_reply_keywords": ["on it", "team", "looping", "investigating"],
        "conflict": True,
        "conflict_keywords": ["dinner", "late", "family", "delay"]
    },

    "conflict_calendar": {
        "emails": [
            "Boss: Sync tomorrow at 10am to review Q3 strategy. Block your calendar.",
            "Dr. Sharma: Reminder — your annual checkup is tomorrow at 10am. Please confirm."
        ],
        "calendar": {
            "10am": "Available — but two requests just came in"
        },
        "pending_tasks": [
            "Confirm Q3 sync with boss",
            "Annual health checkup (overdue by 3 months)"
        ],
        "context": "Two important appointments at the exact same time. Neither can be easily skipped.",
        "ideal_action": {
            "v1": "reschedule",
            "v2": "reschedule",
            "v3": "delegate"
        },
        "ideal_reply_keywords": ["reschedule", "conflict", "available", "alternative", "sorry"],
        "conflict": True,
        "conflict_keywords": ["reschedule", "conflict", "alternative", "another time"]
    },

    "boss_pressure": {
        "emails": [
            "Where is the Q3 report? I needed it yesterday. This is completely unacceptable. — Director"
        ],
        "calendar": {
            "Now": "Working on Q3 report — 80% complete",
            "2pm": "Team standup"
        },
        "pending_tasks": [
            "Complete Q3 report (80% done — need 1 more hour)",
            "Team standup in 30 minutes"
        ],
        "context": "Report is almost done. Boss is angry. Standup in 30 minutes.",
        "ideal_action": {
            "v1": "reply",
            "v2": "reply",
            "v3": "reply"
        },
        "ideal_reply_keywords": ["apologize", "ready", "sending", "shortly", "done"],
        "conflict": False,
        "conflict_keywords": []
    },

    "personal_event": {
        "emails": [
            "Hey Kanishk! You are invited to my wedding on May 10th. "
            "Please RSVP by April 30th. Would mean the world to have you there! — Priya"
        ],
        "calendar": {
            "May 10": "Flight to Delhi — already booked and paid"
        },
        "pending_tasks": [
            "RSVP to Priya's wedding",
            "Check if flight can be rescheduled"
        ],
        "context": "Close friend's wedding. You have a pre-booked flight conflict on the same day.",
        "ideal_action": {
            "v1": "reply",
            "v2": "reply",
            "v3": "reply"
        },
        "ideal_reply_keywords": ["congratulations", "so happy", "conflict", "sorry", "celebrate"],
        "conflict": True,
        "conflict_keywords": ["conflict", "flight", "sorry", "celebrate", "another"]
    },

    "spam_disguised": {
        "emails": [
            "URGENT ACTION REQUIRED: Your account will be suspended in 24 hours. "
            "Click here immediately to verify your identity and payment details."
        ],
        "calendar": {},
        "pending_tasks": [],
        "context": "Looks urgent but this is a phishing email pretending to be from your bank.",
        "ideal_action": {
            "v1": "ignore",
            "v2": "ignore",
            "v3": "ignore"
        },
        "ideal_reply_keywords": [],
        "conflict": False,
        "conflict_keywords": []
    },

    "legal_compliance": {
        "emails": [
            "From: Legal Team. Re: Compliance audit. "
            "You must provide all communications related to Project Phoenix by EOD Friday. "
            "Failure to comply may result in regulatory action."
        ],
        "calendar": {
            "Friday EOD": "Legal compliance deadline"
        },
        "pending_tasks": [
            "Compile Project Phoenix communications",
            "Monthly budget review"
        ],
        "context": "Legal compliance request. Non-negotiable deadline. Could have serious consequences.",
        "ideal_action": {
            "v1": "escalate",
            "v2": "escalate",
            "v3": "escalate"
        },
        "ideal_reply_keywords": ["acknowledged", "compile", "legal", "EOD", "will provide"],
        "conflict": False,
        "conflict_keywords": []
    },

    "personal_family": {
        "emails": [
            "Mom: Beta, your father had chest pains last night. "
            "Doctor says he needs rest. When can you come home? "
            "We miss you. Do not worry too much, but please come soon if you can."
        ],
        "calendar": {
            "This weekend": "Mandatory team offsite — attendance required"
        },
        "pending_tasks": [
            "Attend mandatory team offsite this weekend",
            "Call parents"
        ],
        "context": "Father is unwell. Weekend is blocked with mandatory work event. Emotional and personal.",
        "ideal_action": {
            "v1": "reply",
            "v2": "reply",
            "v3": "reply"
        },
        "ideal_reply_keywords": ["love", "coming", "soon", "call", "don't worry", "okay"],
        "conflict": True,
        "conflict_keywords": ["soon", "coming", "weekend", "offsite", "after"]
    },

    "client_urgent": {
        "emails": [
            "From: Rahul Mehta (Enterprise Client — Tier 1). Subject: SERIOUS ISSUE. "
            "Our entire team has been unable to use your platform for 3 hours. "
            "We are losing productivity and patience. "
            "If this is not resolved in the next 60 minutes, I will escalate to your CEO."
        ],
        "calendar": {
            "Now": "Internal all-hands meeting — in progress"
        },
        "pending_tasks": [
            "Internal all-hands (in progress)",
            "Client issue resolution"
        ],
        "context": "Major enterprise client threatening CEO escalation. You are currently in a meeting.",
        "ideal_action": {
            "v1": "escalate",
            "v2": "reply",
            "v3": "delegate"
        },
        "ideal_reply_keywords": ["sincerely apologize", "top priority", "personally", "resolve", "update"],
        "conflict": True,
        "conflict_keywords": ["meeting", "stepping out", "priority", "immediately"]
    },

    "finance_pressure": {
        "emails": [
            "From: Accounts Payable. "
            "Invoice INV-2024-089 is 30 days overdue. Amount: Rs 45,000. "
            "Please settle within 48 hours to avoid late fees and service suspension."
        ],
        "calendar": {
            "Tomorrow": "Monthly budget review with finance team"
        },
        "pending_tasks": [
            "Monthly budget review tomorrow",
            "Team salary approval pending"
        ],
        "context": "Invoice overdue. Finance team needs action before suspension kicks in.",
        "ideal_action": {
            "v1": "delegate",
            "v2": "delegate",
            "v3": "delegate"
        },
        "ideal_reply_keywords": ["forwarded", "finance team", "resolve", "EOD", "handle"],
        "conflict": False,
        "conflict_keywords": []
    },

    "drift_detection": {
        "emails": [
            "From: Marketing Team. Subject: Urgent — client campaign copy needs review ASAP. "
            "The client is waiting on final approval. Please review and sign off today."
        ],
        "calendar": {
            "Now": "Deep work block — do not disturb (self-set)"
        },
        "pending_tasks": [
            "Deep work block — important project",
            "Marketing copy review request"
        ],
        "context": (
            "Marketing review request. Conflicts with your focus block. "
            "This scenario tests schema awareness — the correct action changes per schema."
        ),
        "ideal_action": {
            "v1": "ignore",
            "v2": "reply",
            "v3": "delegate"
        },
        "ideal_reply_keywords": ["review", "approve", "client", "send", "done"],
        "conflict": True,
        "conflict_keywords": ["focus", "block", "after", "later", "delegate"]
    }
}

SCHEMAS = {
    "v1": {
        "name": "Corporate Mode",
        "urgent_triggers": ["urgent", "critical", "down", "p0", "alert", "all hands"],
        "tone": "formal",
        "max_reply_words": 50,
        "priority_order": ["production", "legal", "boss", "clients", "personal"]
    },
    "v2": {
        "name": "Startup Mode",
        "urgent_triggers": ["asap", "blocker", "hotfix", "client", "ship it"],
        "tone": "casual",
        "max_reply_words": 100,
        "priority_order": ["clients", "boss", "production", "personal", "legal"]
    },
    "v3": {
        "name": "Executive Mode",
        "urgent_triggers": ["revenue", "legal", "compliance", "board", "regulatory"],
        "tone": "formal",
        "max_reply_words": 30,
        "priority_order": ["legal", "revenue", "clients", "boss", "production"]
    }
}
