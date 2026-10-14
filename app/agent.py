import json
import random
from pathlib import Path

from app.environment import LiveCommerceEnvironment

MODEL_PATH = Path("models/q_table.json")


class QLearningAgent:
    def __init__(self, learning_rate=0.10, discount_factor=0.95, epsilon=1.0,
                 epsilon_min=0.05, epsilon_decay=0.995, seed=None):
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = random.Random(seed)
        self.q_table = {}

    def _ensure_state(self, state_key):
        if state_key not in self.q_table:
            self.q_table[state_key] = {a: 0.0 for a in LiveCommerceEnvironment.ACTIONS}

    def choose_action(self, state_key, training=False):
        self._ensure_state(state_key)
        if training and self.rng.random() < self.epsilon:
            return self.rng.choice(LiveCommerceEnvironment.ACTIONS)
        values = self.q_table[state_key]
        best = max(values.values())
        return self.rng.choice([a for a, v in values.items() if v == best])

    def learn(self, state_key, action, reward, next_state_key, done):
        self._ensure_state(state_key)
        self._ensure_state(next_state_key)
        current = self.q_table[state_key][action]
        future = 0.0 if done else max(self.q_table[next_state_key].values())
        target = reward + self.gamma * future
        self.q_table[state_key][action] = current + self.alpha * (target - current)

    def train(self, episodes=10_000, seed=42, reset_model=True):
        # Reward semantics changed to prioritize revenue. Do not mix old Q-values
        # with the new reward scale when starting a fresh training run.
        if reset_model:
            self.q_table = {}
            self.epsilon = 1.0
        env = LiveCommerceEnvironment(seed=seed)
        history = []
        for episode in range(episodes):
            env.reset()
            total_reward = 0
            while True:
                state = env.state_key()
                action = self.choose_action(state, training=True)
                result = env.step(action)
                self.learn(state, action, result["reward"], env.state_key(), result["done"])
                total_reward += result["reward"]
                if result["done"]:
                    break
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            history.append({"episode": episode + 1, "reward": total_reward,
                            "sales": env.sales, "revenue": env.revenue})
        return history

    def save(self, path=MODEL_PATH):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"config": {"learning_rate": self.alpha,
                              "discount_factor": self.gamma, "epsilon": self.epsilon},
                   "q_table": self.q_table}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self, path=MODEL_PATH):
        path = Path(path)
        if not path.exists():
            return False
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.q_table = payload["q_table"]
        self.epsilon = payload.get("config", {}).get("epsilon", self.epsilon)
        return True
