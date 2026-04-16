# AFL Prediction Intelligence

A production AFL prediction system for calibrated probability estimates, edge detection, and correlation-aware multi recommendations.

> **Live-data only.** This system requires real API credentials and will not start without them. There is no demo mode, no CSV fallback, and no partial-data path. If data is unavailable, the system fails clearly and tells you why.

> This system **does not place bets**. It predicts, prices, ranks, and explains AFL betting legs and multis for research purposes.

---

## Requirements

| Service | Required? | Purpose |
|---|---|---|
| [Sportradar AFL API](https://developer.sportradar.com/) | **Required** | Fixtures, schedules, match data |
| [The Odds API](https://the-odds-api.com/) | Optional (recommended) | Live bookmaker odds, edge calculation |

Without Sportradar credentials the system will refuse to start.

Without Odds API credentials the system runs model-only predictions (no market edge comparison).

---

## Quick Start

### 1. Install dependencies

```bash
cd afl_multi_builder
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env — set at minimum:
#   SPORTRADAR_API_KEY=<your_key>
#   SPORTRADAR_AFL_SEASON_ID=sr:season:<id>
```

To find the current season ID:
```bash
curl "https://api.sportradar.com/australianrules/trial/v3/en/competitions/sr:competition:656/seasons.json?api_key=<your_key>"
```

### 3. Run preflight validation

Verify all required API sources are reachable before touching the pipeline:

```bash
python -c "
from app.services.preflight import PreflightService
svc = PreflightService()
report = svc.run(raise_on_failure=False)
for c in report.checks:
    icon = '✓' if c.passed else ('✗' if c.required else '⚠')
    print(f'  {icon} {c.name}: {c.detail}')
print('Passed:', report.passed)
"
```

Or via the API (after starting the server):
```bash
curl http://localhost:8000/api/v1/preflight
```

### 4. Train models

```bash
python scripts/run_training.py
```

### 5. Run the prediction pipeline

```bash
python scripts/run_pipeline.py
```

### 6. Start the API + UI

```bash
uvicorn app.main:app --reload
# UI:      http://localhost:8000
# API docs: http://localhost:8000/docs
# Preflight: http://localhost:8000/api/v1/preflight
```

---

## Running Tests

```bash
pytest tests/ -v
```

Tests that require live API credentials are skipped (not faked) if keys are absent.
There are no demo-data-backed tests.

---

## Docker

```bash
# Copy and configure .env first
cp .env.example .env
# Edit .env with your API keys

docker-compose up --build
# API: http://localhost:8000
# Streamlit: http://localhost:8501
```

---

## Project Structure

```
afl_multi_builder/
├── app/
│   ├── api/routes.py            # FastAPI endpoints
│   ├── core/                    # Config, logging, metrics
│   ├── data_ingestion/          # Sportradar, Odds API, edge intelligence
│   ├── db/                      # SQLAlchemy models
│   ├── features/pipeline.py     # Elo, rolling features, player features
│   ├── pricing/                 # Models, calibration, odds processing
│   ├── correlation/engine.py    # Pairwise correlation + conflict detection
│   ├── optimizer/multi_builder.py  # Multi generation + leg ranking
│   ├── services/                # Pipeline, training, backtest, preflight
│   ├── schemas/                 # Pydantic API schemas
│   └── ui/                      # Static HTML/CSS/JS frontend
├── scripts/                     # CLI scripts
├── tests/                       # pytest test suite (live-data tests)
├── streamlit_app.py             # Streamlit alternative UI
├── requirements.txt
├── .env.example
├── Dockerfile
└── docker-compose.yml
```

---

## Data Flow

```
.env (API keys)
    ↓
PreflightService (validates all sources before any prediction)
    ↓
SportradarLoader (fixtures, schedules, match summaries)
    ↓ optionally merged with
OddsAPIProvider (live bookmaker odds from The Odds API)
    ↓
FeaturePipeline (Elo ratings, rolling stats, player features)
    ↓
EnsembleModel (Elo + LR + XGBoost, isotonic calibration)
    ↓
LegRanker (EV, edge, confidence scoring)
    ↓
MultiBuilder (correlation-aware combinations)
    ↓
FastAPI / UI (structured human-readable output)
```

If **any required step fails** → hard stop with clear error message.

---

## Model Architecture

### Match Win Probability
- **Elo ratings** (K=32, home advantage=70 pts, season carryover=75%)
- **Logistic Regression** over engineered features
- **XGBoost classifier** (100 trees, max_depth=4, learning_rate=0.05)
- **Weighted ensemble** (25% Elo, 35% LR, 40% XGB)
- **Isotonic regression calibration** fitted on held-out validation data

### Player Disposals Over/Under
- **Disabled** by default — player roster and stats endpoints are not available on the Sportradar trial plan
- When player data becomes available: XGBoost regressor + Normal distribution conversion

### Features (pre-game only, no look-ahead)
- Pre-game Elo ratings and differential
- Rolling 5-game: score, disposals, clearances, contested possessions, inside 50s
- Weather: temperature, wind speed, rain flag (when available)

---

## Supported Markets

| Market | Status | Requirement |
|---|---|---|
| Head-to-Head | **Active** | Sportradar fixtures |
| Line (handicap) | **Active** | Sportradar fixtures |
| Total score | **Active** | Sportradar fixtures |
| Player Disposals | **Disabled** | Requires player stats (not on trial plan) |

Markets that cannot be backed by real data are **disabled**, not faked.

---

## Correlation Engine

AFL multis are not independent. The engine handles:

| Leg Combination | Base Correlation |
|---|---|
| H2H + Line (same game) | 0.85 |
| Total + Player Props (same game) | 0.40 |
| Player + Player (same team, same game) | 0.60 |
| Player + Player (same player) | 0.85 |
| Cross-game | 0.00 |

Adjusted probability uses a Gaussian copula approximation:
```
adj_prob = naive_prob × penalty_factor
penalty_factor = 1 − (avg_correlation × 0.3 × (n_legs − 1))
```

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/v1/health` | GET | Health check |
| `/api/v1/preflight` | GET | Preflight validation (API keys, fixtures, models) |
| `/api/v1/system/status` | GET | Full system status with market availability |
| `/api/v1/pipeline/run` | POST | Run full prediction pipeline |
| `/api/v1/legs` | GET | Get ranked legs (from last pipeline run) |
| `/api/v1/multis` | GET | Get ranked multis (from last pipeline run) |
| `/api/v1/multis/generate` | POST | Generate multis from custom legs |
| `/api/v1/training/run` | POST | Train all models |
| `/api/v1/backtest/run` | POST | Run walk-forward backtest |
| `/api/v1/reports/summary` | GET | Data and model summary |
| `/api/v1/sync/upcoming` | POST | Sync upcoming fixtures from Sportradar |
| `/api/v1/quota/status` | GET | Sportradar API quota usage |

Full interactive docs: `http://localhost:8000/docs`

---

## Preflight Checks

The system runs 8 checks before every pipeline run:

| Check | Required? | Description |
|---|---|---|
| `data_mode` | Required | Must be 'live' or 'cache' |
| `sportradar_api_key` | Required | Key must be present |
| `sportradar_season_id` | Required | Current season ID must be set |
| `sportradar_connectivity` | Required | API must be reachable |
| `upcoming_fixtures` | Required | At least 1 upcoming fixture must exist |
| `odds_api_key` | Optional | Without this, no market edge calculation |
| `model_artifacts` | Optional | Auto-trains if missing |
| `data_freshness` | Optional | Warns if cache is stale |

If any **required** check fails → pipeline blocked, clear error returned.

---

## Error Taxonomy

| Error | Cause | Fix |
|---|---|---|
| `PREFLIGHT_FAILED` | Required preflight check failed | See check details and fix instructions |
| `SPORTRADAR_NOT_CONFIGURED` | Missing API key | Set `SPORTRADAR_API_KEY` in .env |
| `NoUpcomingFixturesError` | No upcoming AFL fixtures in season | Check `SPORTRADAR_AFL_SEASON_ID` |
| `RuntimeError: season ID` | `SPORTRADAR_AFL_SEASON_ID` not set | Add season ID to .env |
| `ImportError: demo_loader` | Old code importing demo module | Remove demo imports; use real providers |

---

## Important Notes

- **No real bets are placed** — research and analysis tool only
- **No demo mode** — live API credentials required at all times
- **No partial predictions** — if required data is missing, the market is disabled
- Accuracy improves with more historical match data
- All edge/EV calculations assume you can get the stated odds at prediction time
- Past performance does not guarantee future results
