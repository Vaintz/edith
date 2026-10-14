import random


class LiveCommerceEnvironment:
    """Synthetic live-commerce environment for reinforcement learning."""

    ACTIONS = ["JELASKAN_PRODUK", "BERI_DISKON", "INTERAKSI_SANTAI"]

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.price = 100_000
        self.reset()

    def reset(self):
        self.stock = 20
        self.traffic = self.rng.randint(3, 12)
        self.interested = self.rng.random() < 0.5
        self.revenue = self.sales = self.discount_count = self.turn = 0
        return self.get_state()

    def get_state(self):
        stock_level = 0 if self.stock == 0 else (1 if self.stock <= 5 else 2)
        return {
            "stock_level": stock_level,
            "traffic_level": 1 if self.traffic >= 8 else 0,
            "sentiment_level": 1 if self.interested else 0,
        }

    def state_key(self):
        state = self.get_state()
        return f"{state['stock_level']}_{state['traffic_level']}_{state['sentiment_level']}"

    def step(self, action):
        if action not in self.ACTIONS:
            raise ValueError(f"Aksi tidak dikenal: {action}")
        if self.stock <= 0 or self.turn >= 100:
            raise RuntimeError("Sesi selesai. Mulai sesi baru dengan reset().")

        old_traffic, old_interest, old_stock = self.traffic, self.interested, self.stock
        reward, messages, sold = 0, [], False
        discount = 0.10 if action == "BERI_DISKON" else 0.0

        if action == "JELASKAN_PRODUK":
            messages.append("Host menjelaskan manfaat dan keunggulan produk.")
            if self.rng.random() < 0.55:
                self.interested = True
        elif action == "BERI_DISKON":
            self.discount_count += 1
            messages.append("Host menawarkan diskon 10% kepada penonton.")
            if self.rng.random() < 0.70:
                self.interested = True
        else:
            messages.append("Host menyapa penonton dan berinteraksi santai.")
            if self.rng.random() < 0.60:
                self.traffic += 2
                self.interested = True

        probability = 0.05
        if self.interested:
            probability += 0.20
        if self.traffic >= 8:
            probability += 0.15
        probability += {"JELASKAN_PRODUK": 0.10, "BERI_DISKON": 0.25, "INTERAKSI_SANTAI": 0.05}[action]
        probability = min(probability, 0.95)

        if self.stock > 0 and self.rng.random() < probability:
            self.stock -= 1
            self.sales += 1
            sold = True
            transaction_price = int(self.price * (1 - discount))
            self.revenue += transaction_price
            # Reward hanya merepresentasikan pendapatan simulasi setelah diskon.
            # Rp10.000 pendapatan = 1 poin; interaksi tanpa transaksi tidak diberi bonus.
            reward += transaction_price / 10_000
            messages.append(f"Pembelian berhasil: Rp{transaction_price:,}.")
        else:
            messages.append("Belum ada pembelian.")

        if self.rng.random() < 0.25:
            self.traffic = max(0, self.traffic + self.rng.choice([-3, -1, 1, 3]))
        if self.traffic == 0:
            messages.append("Penonton sedang sepi.")
        self.turn += 1
        done = self.stock == 0 or self.turn >= 100
        return {
            "state": self.get_state(),
            "reward": reward,
            "done": done,
            "info": {
                "action": action, "messages": messages, "sold": sold,
                "discount": discount, "stock_before": old_stock,
                "stock": self.stock, "traffic_before": old_traffic,
                "traffic": self.traffic, "interested_before": old_interest,
                "interested": self.interested, "revenue": self.revenue,
                "sales": self.sales, "discount_count": self.discount_count,
                "purchase_probability": round(probability, 3),
            },
        }
