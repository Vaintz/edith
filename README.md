# Edith-sama — Live Commerce AI Agent

Local prototype of a simulated live-commerce host that learns to choose sales strategies using tabular Q-Learning.

## Features

- Python simulator for inventory, viewer traffic, audience interest, purchases, sales, and revenue.
- Tabular Q-Learning with epsilon-greedy exploration.
- JSON persistence for the learned Q-table.
- FastAPI endpoints for training, running simulation sessions, and evaluating the learned policy against random and fixed baselines.
- No GPU or external LLM API required.

> This is a synthetic environment. Reward values and buyer behavior are assumptions for software experimentation, not evidence of real-world consumer behavior.

## Requirements

- Python 3.11+
- Windows PowerShell (commands below also adapt to other platforms)

## Setup on Windows

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation for this terminal session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Run the API

From the repository root:

```powershell
python -m uvicorn app.main:app --reload
```

Open:
- API root: http://127.0.0.1:8000
- Interactive API docs: http://127.0.0.1:8000/docs

## Suggested first run

1. Call `POST /train` with `{"episodes": 10000}`.
2. Call `POST /session/start`.
3. Call `POST /session/step` with an empty JSON object `{}` to let the agent choose an action.
4. Call `GET /session/state` to inspect current metrics.
5. Call `GET /model` to inspect the learned Q-table.

You can pass an explicit action to `POST /session/step`:
- `JELASKAN_PRODUK`
- `BERI_DISKON`
- `INTERAKSI_SANTAI`

## Reward and evaluation

- The learning reward is tied directly to simulated revenue after discounts: Rp10,000 revenue equals 1 reward point.
- Audience interaction can change the simulated audience state, but it does not earn reward unless a sale occurs.
- POST `/evaluate` compares the trained policy with random and fixed policies over seeded synthetic episodes. Results are simulation evidence only, not proof of real-world performance.
- Retraining starts a fresh Q-table because the reward definition changed; it does not mix old values with the new reward scale.

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Service information |
| GET | `/health` | Health check |
| POST | `/train` | Train and save the Q-learning agent from a fresh Q-table |
| POST | `/session/start` | Start a fresh simulation session |
| POST | `/session/step` | Execute one agent or manually selected action |
| GET | `/session/state` | Current session state and metrics |
| GET | `/model` | Model metadata |
| POST | `/evaluate` | Compare trained, random, and fixed policies |

## Project structure

```text
edith-sama/
├── app/
│   ├── __init__.py
│   ├── agent.py
│   ├── environment.py
│   └── main.py
├── models/              # Q-table is generated here
├── .gitignore
├── README.md
└── requirements.txt
```

## Current scope and limitations

- The audience and purchase process are stochastic synthetic rules.
- Revenue is gross simulated sales revenue; operating costs and product margins are not modeled.
- Q-learning performance must be evaluated against a baseline before drawing conclusions.
- The web dashboard, avatar, live chat UI, and comparative evaluation are planned next.
