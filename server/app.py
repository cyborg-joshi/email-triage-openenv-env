from fastapi import FastAPI
from env.environment import ExecutiveAssistantEnv
from env.models import ExecutiveAction
from env.scenarios import SCHEMAS

app = FastAPI(
    title="AI Executive Assistant Environment",
    description="OpenEnv environment for training agents on email triage with schema drift.",
    version="2.0.0"
)

env = ExecutiveAssistantEnv()


@app.get("/")
def health():
    return {
        "status": "ok",
        "version": "2.0.0",
        "environment": "ai-executive-assistant",
        "theme": "3.2 World Modeling (Personal Tasks) + Patronus AI Schema Drift"
    }


@app.post("/reset")
def reset(task: str = None):
    obs = env.reset(task)
    return {
        "observation": obs.dict(),
        "reward": 0.0,
        "done": False,
        "info": {
            "task": env.current_scenario_key,
            "schema": env.current_schema_key,
            "episode": env.episode_count
        }
    }


@app.post("/step")
def step(action: ExecutiveAction):
    obs, reward, done, info = env.step(action)
    reward = min(max(float(reward), 0.01), 0.99) if done else 0.0
    return {
        "observation": obs.dict(),
        "reward": reward,
        "done": done,
        "info": info
    }


@app.get("/state")
def state():
    return env.state()


@app.post("/admin/reset_env")
def admin_reset_env():
    env.episode_count = 0
    env.current_schema_key = "v1"
    return {"status": "ok", "message": "Environment reset to episode 0 (v1 Corporate)"}


@app.get("/schema")
def schema():
    return {
        "current_schema_key": env.current_schema_key,
        "episode_count": env.episode_count,
        "schema_details": SCHEMAS.get(env.current_schema_key, {}),
        "drift_schedule": {
            "v1": "episodes 1-10  (Corporate Mode)",
            "v2": "episodes 11-20 (Startup Mode)",
            "v3": "episodes 21+   (Executive Mode)"
        }
    }


def main():
    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)
