from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.agent import QLearningAgent
from app.environment import LiveCommerceEnvironment

app = FastAPI(title="Edith-sama — Live Commerce AI Agent",
              description="Local live-commerce simulation using tabular Q-Learning.",
              version="0.1.0")
agent = QLearningAgent()
agent.load()
environment = LiveCommerceEnvironment()
session_active = False


class TrainRequest(BaseModel):
    episodes: int = Field(default=10_000, ge=1, le=100_000)


class StepRequest(BaseModel):
    action: str | None = None


@app.get("/")
def root():
    return {"app": "Edith-sama — Live Commerce AI Agent", "status": "running", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": bool(agent.q_table), "session_active": session_active}


@app.post("/train")
def train_model(request: TrainRequest):
    history = agent.train(episodes=request.episodes)
    agent.save()
    recent = history[-min(100, len(history)):]
    return {
        "message": "Training selesai", "episodes": len(history),
        "q_states": len(agent.q_table), "epsilon": round(agent.epsilon, 4),
        "last_100_average_reward": round(sum(x["reward"] for x in recent) / len(recent), 2),
        "model_path": "models/q_table.json",
    }


@app.post("/session/start")
def start_session():
    global session_active
    environment.reset()
    session_active = True
    return {"message": "Sesi simulasi dimulai", "state": environment.get_state(),
            "stats": {"stock": environment.stock, "traffic": environment.traffic,
                      "revenue": environment.revenue, "sales": environment.sales}}


@app.post("/session/step")
def session_step(request: StepRequest | None = None):
    if not session_active:
        raise HTTPException(status_code=400, detail="Mulai sesi terlebih dahulu.")
    if environment.stock <= 0 or environment.turn >= 100:
        raise HTTPException(status_code=400, detail="Sesi selesai. Mulai sesi baru.")
    manual_action = request.action if request else None
    if manual_action is not None and manual_action not in environment.ACTIONS:
        raise HTTPException(status_code=400, detail="Aksi tidak valid.")
    action = manual_action or agent.choose_action(environment.state_key(), training=False)
    result = environment.step(action)
    return {**result, "session": {"turn": environment.turn, "max_turns": 100,
                                  "finished": result["done"]}}


@app.get("/session/state")
def session_state():
    return {"active": session_active, "state": environment.get_state(),
            "stats": {"stock": environment.stock, "traffic": environment.traffic,
                      "interested": environment.interested, "revenue": environment.revenue,
                      "sales": environment.sales, "discount_count": environment.discount_count,
                      "turn": environment.turn}}


@app.get("/model")
def model_info():
    return {"loaded": bool(agent.q_table), "states_learned": len(agent.q_table),
            "epsilon": round(agent.epsilon, 4), "actions": environment.ACTIONS}
