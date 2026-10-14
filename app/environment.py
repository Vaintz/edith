import random


class LiveCommerceEnvironment:
    """Context-aware synthetic live-commerce environment for tabular Q-learning."""

    ACTIONS = ["JELASKAN_PRODUK", "BERI_DISKON", "INTERAKSI_SANTAI"]
    MAX_TURNS = 30
    DEFAULT_UNIT_COST = 60_000  # Demo assumption, not a real product cost.
    INTENTS = ["TANYA_PRODUK", "SENSITIF_HARGA", "PASIF"]

    def __init__(self, seed=None, unit_cost=DEFAULT_UNIT_COST):
        if not isinstance(unit_cost, int) or unit_cost < 0:
            raise ValueError("unit_cost harus berupa integer >= 0")
        self.rng = random.Random(seed)
        self.price = 100_000
        self.unit_cost = unit_cost
        self.reset()

    def reset(self):
        self.stock = 20
        self.traffic = self.rng.randint(3, 12)
        self.interested = self.rng.random() < 0.5
        self.intent = self.rng.choice(self.INTENTS)
        self.revenue = self.sales = self.discount_count = self.turn = 0
        self.gross_profit = 0
        return self.get_state()

    def get_state(self):
        stock_level = 0 if self.stock == 0 else (1 if self.stock <= 5 else 2)
        return {
            "stock_level": stock_level,
            "traffic_level": 1 if self.traffic >= 8 else 0,
            "sentiment_level": 1 if self.interested else 0,
            "audience_intent": self.intent,
        }

    def state_key(self):
        state = self.get_state()
        return f"{state['stock_level']}_{state['traffic_level']}_{state['sentiment_level']}_{state['audience_intent']}"

    def step(self, action):
        if action not in self.ACTIONS:
            raise ValueError(f"Aksi tidak dikenal: {action}")
        if self.stock <= 0 or self.turn >= self.MAX_TURNS:
            raise RuntimeError("Sesi selesai. Mulai sesi baru dengan reset().")

        old_traffic, old_interest, old_stock, old_intent = self.traffic, self.interested, self.stock, self.intent
        messages, sold = [], False
        discount = 0.10 if action == "BERI_DISKON" else 0.0
        intent_fit = 0.0

        if action == "JELASKAN_PRODUK":
            messages.append("Host menjelaskan manfaat dan menjawab kebutuhan produk.")
            intent_fit = 0.18 if self.intent == "TANYA_PRODUK" else 0.04
            if self.rng.random() < (0.72 if self.intent == "TANYA_PRODUK" else 0.30):
                self.interested = True
                if self.intent == "TANYA_PRODUK":
                    self.intent = "PASIF" if self.rng.random() < 0.35 else "TANYA_PRODUK"
        elif action == "BERI_DISKON":
            self.discount_count += 1
            messages.append("Host menawarkan diskon 10%.")
            intent_fit = 0.20 if self.intent == "SENSITIF_HARGA" else -0.04
            if self.rng.random() < (0.72 if old_intent == "SENSITIF_HARGA" else 0.28):
                self.interested = True
                if old_intent == "SENSITIF_HARGA":
                    self.intent = "PASIF" if self.rng.random() < 0.55 else "SENSITIF_HARGA"
        else:
            messages.append("Host menyapa penonton dan membangun interaksi.")
            intent_fit = 0.17 if self.intent == "PASIF" else 0.03
            if self.rng.random() < (0.68 if self.intent == "PASIF" else 0.35):
                self.traffic += 2
                self.interested = True
                if self.intent == "PASIF":
                    self.intent = self.rng.choice(["TANYA_PRODUK", "SENSITIF_HARGA", "PASIF"])

        # Baseline demand + current context + action fit. Discount is not universally best.
        probability = 0.04
        if self.interested:
            probability += 0.16
        if self.traffic >= 8:
            probability += 0.12
        probability += {
            "JELASKAN_PRODUK": 0.10,
            "BERI_DISKON": 0.08,
            "INTERAKSI_SANTAI": 0.05,
        }[action]
        probability += intent_fit
        if self.stock <= 5:
            probability -= 0.03  # Scarcity is not a free conversion boost; fewer units remain.
        probability = max(0.01, min(probability, 0.90))

        transaction_price = 0
        transaction_profit = 0
        reward = 0.0
        if self.stock > 0 and self.rng.random() < probability:
            self.stock -= 1
            self.sales += 1
            sold = True
            transaction_price = int(self.price * (1 - discount))
            transaction_profit = transaction_price - self.unit_cost
            self.revenue += transaction_price
            self.gross_profit += transaction_profit
            reward = transaction_profit / 10_000
            messages.append(f"Pembelian berhasil: Rp{transaction_price:,}.")
        else:
            messages.append("Belum ada pembelian pada giliran ini.")

        if self.rng.random() < 0.25:
            self.traffic = max(0, self.traffic + self.rng.choice([-3, -1, 1, 3]))
        if self.traffic == 0:
            messages.append("Penonton sedang sepi.")

        # Audience intent can change over time, even when no purchase occurs.
        if self.rng.random() < 0.12:
            self.intent = self.rng.choice(self.INTENTS)

        self.turn += 1
        done = self.stock == 0 or self.turn >= self.MAX_TURNS
        return {
            "state": self.get_state(),
            "reward": reward,
            "done": done,
            "info": {
                "action": action, "messages": messages, "sold": sold,
                "discount": discount, "stock_before": old_stock, "stock": self.stock,
                "traffic_before": old_traffic, "traffic": self.traffic,
                "interested_before": old_interest, "interested": self.interested,
                "intent_before": old_intent, "audience_intent": self.intent,
                "revenue": self.revenue, "sales": self.sales,
                "discount_count": self.discount_count,
                "purchase_probability": round(probability, 3),
                "transaction_price": transaction_price,
                "transaction_profit": transaction_profit,
                "gross_profit": self.gross_profit, "unit_cost": self.unit_cost,
                "intent_fit": round(intent_fit, 2),
            },
        }
