# AFL Multi Builder

A fully local AFL prediction and multi-building application focused on **real predictive accuracy, honest validation, and edge detection**.

> This system **does not place bets**. It predicts, prices, ranks, and explains AFL betting legs and multis.

---

## Features

- **Calibrated probability models** (Elo + Logistic Regression + XGBoost ensemble)
- **Post-hoc isotonic calibration** with Brier score and log loss reporting
- **Expected value and edge detection** per market (head-to-head, player disposals)
- **Correlation-aware multi construction** — never assumes leg independence
- **Walk-forward backtesting** with CLV-style market comparison
- **FastAPI backend** + **Streamlit UI** (or HTML/JS static UI)
- **SQLite database** (PostgreSQL-compatible schema)
- **Full demo dataset** — works out of the box with no external APIs

---

## Quick Start

### 1. Install dependencies

```bash
cd afl_multi_builder
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env if you want to change database path or thresholds
```

### 3. Seed demo data

```bash
python scripts/seed_demo_data.py
```

### 4. Train models

```bash
python scripts/run_training.py
```

### 5. Run backtest

```bash
python scripts/run_backtest.py
```

### 6. Run prediction pipeline

```bash
python scripts/run_pipeline.py
```

### 7. Start the API

```bash
uvicorn app.main:app --reload
# Open http://localhost:8000 for UI
# Open http://localhost:8000/docs for API docs
```

### 8. (Alternative) Streamlit UI

```bash
streamlit run streamlit_app.py
# Open http://localhost:8501
```

---

## Running Tests

```bash
pytest tests/ -v
# or
python scripts/run_tests.py
```

---

## Docker

```bash
# Build and run
docker-compose up --build

# API available at http://localhost:8000
# Streamlit at http://localhost:8501
```

---

## Project Structure

```
afl_multi_builder/
├── app/
│   ├── api/routes.py            # FastAPI endpoints
│   ├── core/                    # Config, logging, metrics, utils
│   ├── data_ingestion/          # Provider interfaces + demo loader
│   ├── db/                      # SQLAlchemy models + seed
│   ├── features/pipeline.py     # Elo, rolling features, player features
│   ├── pricing/                 # Models, calibration, odds processing
│   ├── correlation/engine.py    # Pairwise correlation + conflict detection
│   ├── optimizer/multi_builder.py  # Multi generation + leg ranking
│   ├── services/                # Pipeline, training, backtest, reports
│   ├── schemas/                 # Pydantic API schemas
│   └── ui/                      # Static HTML/CSS/JS frontend
├── data/demo/                   # Demo CSV files
├── scripts/                     # CLI scripts
├── tests/                       # pytest test suite
├── streamlit_app.py             # Streamlit alternative UI
├── requirements.txt
├── .env.example
├── Dockerfile
└── docker-compose.yml
```

---

## Model Architecture

### Match Win Probability
- **Elo ratings** (K=32, home advantage=70 pts, season carryover=75%)
- **Logistic Regression** over engineered features
- **XGBoost classifier** (100 trees, max_depth=4, learning_rate=0.05)
- **Weighted ensemble** (25% Elo, 35% LR, 40% XGB)
- **Isotonic regression calibration** fitted on held-out validation data

### Player Disposals Over/Under
- **XGBoost regressor** for expected disposals (5-game rolling features)
- **Normal distribution** conversion to over/under probability
- **Isotonic calibration** applied post-hoc

### Features
- Pre-game Elo ratings and differential
- Rolling 5-game: score, disposals, clearances, contested possessions, inside 50s
- Weather: temperature, wind speed, rain flag
- Player: rolling mean/std (3/5/10 games), form trend, consistency CV, opponent allowance

---

## Correlation Engine

AFL multis are **not independent**. The engine handles:

| Leg Combination | Base Correlation |
|---|---|
| H2H + Line (same game) | 0.85 |
| Total + Player Props (same game) | 0.40 |
| Player + Player (same team, same game) | 0.60 |
| Player + Player (same player) | 0.85 |
| Cross-game | 0.00 |

Adjusted probability uses a **Gaussian copula approximation**:
```
adj_prob = naive_prob × penalty_factor
penalty_factor = 1 - (avg_correlation × 0.3 × (n_legs - 1))
```

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/v1/health` | GET | Health check |
| `/api/v1/pipeline/run` | POST | Run full prediction pipeline |
| `/api/v1/legs` | GET | Get ranked legs (`?mode=value\|safe`) |
| `/api/v1/multis` | GET | Get ranked multis (`?mode=value\|safe\|same_game`) |
| `/api/v1/multis/generate` | POST | Generate multis from custom legs |
| `/api/v1/training/run` | POST | Train all models |
| `/api/v1/backtest/run` | POST | Run walk-forward backtest |
| `/api/v1/reports/summary` | GET | Data and model summary |

Full interactive docs: http://localhost:8000/docs

---

## Calibration Metrics

The system tracks calibration quality via:
- **Brier Score** (0=perfect, 0.25=no-skill baseline)
- **Log Loss** (lower is better)
- **Probability bin analysis** (10 bins)
- **Calibration curve** comparison (raw vs calibrated)

---

## Important Notes

- **No real bets are placed** — this is a research and analysis tool
- Demo data covers 2021-2024 seasons with ~450 player stat records
- Accuracy improves significantly with more historical data (real AFL APIs)
- All edge/EV calculations assume you can get the stated odds at time of prediction
- Past performance does not guarantee future results
