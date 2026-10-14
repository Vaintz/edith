from pathlib import Path
import random
import statistics

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.agent import QLearningAgent
from app.environment import LiveCommerceEnvironment

app = FastAPI(title="Edith-sama — Live Commerce AI Agent",
              description="Local live-commerce simulation using tabular Q-Learning.",
              version="0.5.0")
agent = QLearningAgent()
agent.load()
environment = LiveCommerceEnvironment()
session_active = False
DASHBOARD_PATH = Path(__file__).parent / "static" / "index.html"


class TrainRequest(BaseModel):
    episodes: int = Field(default=10_000, ge=1, le=100_000)


class StepRequest(BaseModel):
    action: str | None = None


class EvaluationRequest(BaseModel):
    episodes: int = Field(default=500, ge=10, le=5000)
    seed: int = Field(default=2026, ge=0, le=2_147_483_647)


def evaluate_policy(policy: str, episodes: int, seed: int):
    """Evaluate a policy and collect revenue and gross-profit metrics."""
    rewards, revenues, gross_profits, sales, turns, discounts = [], [], [], [], [], []
    discount_costs, full_price_values = [], []
    action_stats = {
        action: {"selections": 0, "sales": 0, "revenue": 0, "gross_profit": 0,
                 "discounts": 0, "realized_discount_cost": 0, "full_price_sales_value": 0}
        for action in LiveCommerceEnvironment.ACTIONS
    }
    for episode in range(episodes):
        episode_seed = seed + episode
        env = LiveCommerceEnvironment(seed=episode_seed)
        rng = random.Random(episode_seed + 9_999_991)
        total_reward = 0
        episode_discount_cost = 0
        episode_full_price_value = 0
        while True:
            state_key = env.state_key()
            if policy == "trained":
                values = agent.q_table.get(state_key)
                if values:
                    best_value = max(values.values())
                    action = next(a for a in env.ACTIONS if values.get(a, 0.0) == best_value)
                else:
                    action = env.ACTIONS[0]
            elif policy == "random":
                action = rng.choice(env.ACTIONS)
            else:
                action = "JELASKAN_PRODUK"

            previous_revenue = env.revenue
            previous_profit = env.gross_profit
            result = env.step(action)
            total_reward += result["reward"]
            info = result["info"]
            metrics = action_stats[action]
            metrics["selections"] += 1
            if info["sold"]:
                transaction_revenue = env.revenue - previous_revenue
                transaction_profit = env.gross_profit - previous_profit
                discount_cost = env.price - transaction_revenue
                metrics["sales"] += 1
                metrics["revenue"] += transaction_revenue
                metrics["gross_profit"] += transaction_profit
                metrics["full_price_sales_value"] += env.price
                metrics["realized_discount_cost"] += discount_cost
                episode_full_price_value += env.price
                episode_discount_cost += discount_cost
            if info["discount"]:
                metrics["discounts"] += 1
            if result["done"]:
                break
        rewards.append(total_reward)
        revenues.append(env.revenue)
        gross_profits.append(env.gross_profit)
        sales.append(env.sales)
        turns.append(env.turn)
        discounts.append(env.discount_count)
        discount_costs.append(episode_discount_cost)
        full_price_values.append(episode_full_price_value)

    for metrics in action_stats.values():
        metrics["conversion_rate_percent"] = round(metrics["sales"] / metrics["selections"] * 100, 2) if metrics["selections"] else 0
        metrics["average_revenue_per_selection"] = round(metrics["revenue"] / metrics["selections"], 2) if metrics["selections"] else 0
        metrics["average_gross_profit_per_selection"] = round(metrics["gross_profit"] / metrics["selections"], 2) if metrics["selections"] else 0

    average_revenue = statistics.mean(revenues)
    average_gross_profit = statistics.mean(gross_profits)
    average_discount_cost = statistics.mean(discount_costs)
    average_full_price_value = statistics.mean(full_price_values)
    return {
        "policy": policy,
        "episodes": episodes,
        "average_reward": round(statistics.mean(rewards), 2),
        "average_revenue": round(average_revenue, 2),
        "average_gross_profit": round(average_gross_profit, 2),
        "average_sales": round(statistics.mean(sales), 2),
        "average_turns": round(statistics.mean(turns), 2),
        "average_discounts": round(statistics.mean(discounts), 2),
        "average_realized_discount_cost": round(average_discount_cost, 2),
        "average_full_price_sales_value": round(average_full_price_value, 2),
        "revenue_realization_percent": round(average_revenue / average_full_price_value * 100, 2) if average_full_price_value else 0,
        "sell_through_rate": round(statistics.mean(sales) / 20 * 100, 2),
        "reward_stddev": round(statistics.pstdev(rewards), 2),
        "action_stats": action_stats,
        "rewards": rewards,
    }


@app.get("/", include_in_schema=False)
def root():
    return FileResponse(DASHBOARD_PATH)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": bool(agent.q_table), "session_active": session_active}


@app.post("/train")
def train_model(request: TrainRequest):
    history = agent.train(episodes=request.episodes)
    agent.save()
    recent = history[-min(100, len(history)):]
    return {
        "message": "Training selesai dengan reward laba kotor.",
        "episodes": len(history),
        "q_states": len(agent.q_table),
        "epsilon": round(agent.epsilon, 4),
        "last_100_average_reward": round(sum(x["reward"] for x in recent) / len(recent), 2),
        "model_path": "models/q_table.json",
        "reward_objective": "gross_profit",
        "assumed_unit_cost": environment.unit_cost,
    }


@app.post("/evaluate")
def evaluate_model(request: EvaluationRequest):
    if not agent.q_table:
        raise HTTPException(status_code=400, detail="Model belum dilatih. Jalankan POST /train terlebih dahulu.")
    episodes, seed = request.episodes, request.seed
    trained = evaluate_policy("trained", episodes, seed)
    random_policy = evaluate_policy("random", episodes, seed)
    fixed = evaluate_policy("fixed", episodes, seed)

    def summary(result):
        return {k: v for k, v in result.items() if k != "rewards"}

    baseline_reward = random_policy["average_reward"]
    reward_delta = trained["average_reward"] - baseline_reward
    reward_pct = round((reward_delta / abs(baseline_reward)) * 100, 2) if baseline_reward else None
    return {
        "episodes": episodes,
        "seed": seed,
        "assumptions": {
            "list_price_per_unit": environment.price,
            "assumed_unit_cost": environment.unit_cost,
            "discount_rate": 0.10,
            "reward_objective": "gross_profit = transaction_price - assumed_unit_cost",
            "warning": "HPP Rp60.000 adalah asumsi demo, bukan data produk nyata. Laba kotor belum mengurangi biaya operasional, pajak, platform, atau biaya lain."
        },
        "note": "Evaluasi pada lingkungan sintetis. Ketiga kebijakan memakai seed episode yang sama. Hasil bukan bukti performa transaksi nyata. Reward mengoptimalkan laba kotor dengan asumsi HPP Rp60.000 per unit.",
        "trained_agent": summary(trained),
        "random_baseline": summary(random_policy),
        "fixed_baseline": summary(fixed),
        "comparison_vs_random": {
            "average_reward_difference": round(reward_delta, 2),
            "average_reward_percent": reward_pct,
            "average_revenue_difference": round(trained["average_revenue"] - random_policy["average_revenue"], 2),
            "average_gross_profit_difference": round(trained["average_gross_profit"] - random_policy["average_gross_profit"], 2),
            "average_sales_difference": round(trained["average_sales"] - random_policy["average_sales"], 2),
            "average_discount_difference": round(trained["average_discounts"] - random_policy["average_discounts"], 2),
            "average_realized_discount_cost_difference": round(trained["average_realized_discount_cost"] - random_policy["average_realized_discount_cost"], 2),
            "reward_better_than_random": trained["average_reward"] > random_policy["average_reward"],
        },
        "comparison_vs_fixed": {
            "average_reward_difference": round(trained["average_reward"] - fixed["average_reward"], 2),
            "average_revenue_difference": round(trained["average_revenue"] - fixed["average_revenue"], 2),
            "average_gross_profit_difference": round(trained["average_gross_profit"] - fixed["average_gross_profit"], 2),
            "average_realized_discount_cost_difference": round(trained["average_realized_discount_cost"] - fixed["average_realized_discount_cost"], 2),
            "reward_better_than_fixed": trained["average_reward"] > fixed["average_reward"],
        },
    }


@app.post("/session/start")
def start_session():
    global session_active
    environment.reset()
    session_active = True
    return {"message": "Sesi simulasi dimulai", "state": environment.get_state(),
            "stats": {"stock": environment.stock, "traffic": environment.traffic,
                      "revenue": environment.revenue, "sales": environment.sales,
                      "gross_profit": environment.gross_profit,
                      "assumed_unit_cost": environment.unit_cost}}


@app.post("/session/step")
def session_step(request: StepRequest | None = None):
    global session_active
    if not session_active:
        raise HTTPException(status_code=400, detail="Mulai sesi terlebih dahulu.")
    if environment.stock <= 0 or environment.turn >= environment.MAX_TURNS:
        raise HTTPException(status_code=400, detail="Sesi selesai. Mulai sesi baru.")
    manual_action = request.action if request else None
    if manual_action is not None and manual_action not in environment.ACTIONS:
        raise HTTPException(status_code=400, detail="Aksi tidak valid.")
    state_key = environment.state_key()
    q_values = agent.q_table.get(state_key, {a: 0.0 for a in environment.ACTIONS})
    action = manual_action or agent.choose_action(state_key, training=False)
    result = environment.step(action)
    if result["done"]:
        session_active = False
    intent_labels = {
        "TANYA_PRODUK": "penonton membutuhkan informasi produk",
        "SENSITIF_HARGA": "penonton sensitif terhadap harga",
        "PASIF": "penonton perlu dibangun engagement-nya",
    }
    action_labels = {
        "JELASKAN_PRODUK": "menjawab kebutuhan informasi",
        "BERI_DISKON": "menanggapi sensitivitas harga",
        "INTERAKSI_SANTAI": "membangun engagement",
    }
    chosen_action = action
    intent = result["info"].get("intent_before", environment.intent)
    reason = (
        f"State menunjukkan {intent_labels.get(intent, intent)}; "
        f"aksi {action_labels.get(chosen_action, chosen_action)}."
    )
    if manual_action:
        reason = "Aksi dipilih operator. " + reason
    return {**result, "decision": {
                "state_key": state_key, "q_values": q_values,
                "mode": "manual" if manual_action else "agent",
                "reason": reason,
                "audience_intent_before_action": result["info"].get("intent_before"),
                "audience_message": result["info"].get("audience_message"),
                "host_response": result["info"].get("host_response"),
                "action_context_fit": result["info"].get("intent_fit")
            },
            "session": {"turn": environment.turn, "max_turns": environment.MAX_TURNS,
                        "finished": result["done"]}}


@app.get("/session/state")
def session_state():
    return {"active": session_active, "state": environment.get_state(),
            "stats": {"stock": environment.stock, "traffic": environment.traffic,
                      "interested": environment.interested, "revenue": environment.revenue,
                      "gross_profit": environment.gross_profit,
                      "assumed_unit_cost": environment.unit_cost,
                      "sales": environment.sales, "discount_count": environment.discount_count,
                      "turn": environment.turn}}


@app.get("/model")
def model_info():
    return {"loaded": bool(agent.q_table), "states_learned": len(agent.q_table),
            "epsilon": round(agent.epsilon, 4), "actions": environment.ACTIONS}
