"""FastAPI route handlers — live-data system only."""
from datetime import datetime
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from loguru import logger

from app.schemas.models import (
    HealthResponse, PipelineRunRequest, PipelineResponse,
    MultiGenerateRequest, MultiResponse, LegResponse,
    TrainingRunResponse, BacktestRunResponse, SummaryResponse,
)
from app.services.pipeline import PredictionPipeline
from app.services.training import TrainingService
from app.services.backtest import WalkForwardBacktester
from app.services.reports import ReportsService
from app.services.preflight import PreflightService, PreflightError
from app.correlation.engine import Leg
from app.optimizer.multi_builder import MultiBuilder
from app.core.config import settings

router = APIRouter()

# In-memory cache for last pipeline/training/backtest results
_last_pipeline_result: Optional[Dict] = None
_last_training_result: Optional[Dict] = None
_last_backtest_result: Optional[Dict] = None


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.utcnow().isoformat(),
    )


@router.get("/preflight", response_model=Dict, tags=["System"])
async def preflight_check():
    """
    Run preflight validation checks.

    Verifies API keys, connectivity, fixtures, odds, and model artifacts
    before committing to a pipeline run. Returns a detailed report of what
    passed, what failed, and how to fix each issue.
    """
    svc = PreflightService()
    report = svc.run(raise_on_failure=False)
    return report.to_dict()


@router.get("/system/status", response_model=Dict, tags=["System"])
async def system_status():
    """
    Full system health status: data sources, API quotas, model registry,
    data freshness, and supported markets.
    """
    status: Dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat(),
        "data_mode": settings.effective_data_mode,
        "afl_data": {
            "configured": settings.is_afl_data_configured,
            "competition_id": settings.afl_data_competition_id,
        },
        "odds_api": {
            "configured": settings.is_odds_api_configured,
        },
        "supported_markets": [],
        "disabled_markets": [],
    }

    # Determine supported markets based on what's configured
    if settings.is_afl_data_configured:
        status["supported_markets"].append({
            "market": "head_to_head",
            "reason": "AFL Data Sports Group API fixtures available",
        })
        status["supported_markets"].append({
            "market": "player_disposals",
            "reason": "AFL Advanced Pack player stats available",
        })
    else:
        status["disabled_markets"].append({
            "market": "head_to_head",
            "reason": "AFL_DATA_AUTHKEY not configured",
        })
        status["disabled_markets"].append({
            "market": "player_disposals",
            "reason": "AFL_DATA_AUTHKEY not configured",
        })

    if settings.is_odds_api_configured:
        status["supported_markets"].append({
            "market": "bookmaker_odds",
            "reason": "Odds API configured — live bookmaker odds available",
        })
    else:
        status["disabled_markets"].append({
            "market": "bookmaker_odds",
            "reason": "ODDS_API_KEY not configured — edge vs market unavailable",
        })

    if settings.is_odds_api_configured:
        try:
            from app.data_ingestion.odds_api_client import OddsAPIClient
            client = OddsAPIClient()
            remaining = client.get_remaining_requests()
            status["odds_api"]["remaining_requests"] = remaining
        except Exception:
            pass

    # Model registry
    try:
        from app.pricing.models import ModelRegistry
        registry = ModelRegistry()
        status["models"] = registry.list_models()
    except Exception:
        status["models"] = []

    # Data freshness
    cache_dir = settings.raw_cache_dir
    if cache_dir.exists():
        cache_files = list(cache_dir.glob("*.json")) + list(cache_dir.glob("*.pickle"))
        if cache_files:
            most_recent = max(cache_files, key=lambda f: f.stat().st_mtime)
            age_hours = (datetime.utcnow().timestamp() - most_recent.stat().st_mtime) / 3600
            status["data_freshness"] = {
                "cache_files": len(cache_files),
                "most_recent_age_hours": round(age_hours, 1),
                "ttl_hours": settings.cache_ttl_hours,
                "fresh": age_hours <= settings.cache_ttl_hours,
            }
        else:
            status["data_freshness"] = {"cache_files": 0, "fresh": True}
    else:
        status["data_freshness"] = {"cache_files": 0, "fresh": True}

    return status


@router.post("/pipeline/run", response_model=Dict, tags=["Pipeline"])
async def run_pipeline(request: PipelineRunRequest = None):
    """
    Run the full prediction pipeline.

    Runs preflight validation first — fails immediately if required data
    sources are unavailable. Generates candidate legs and multis for
    upcoming fixtures using live AFL Data Sports Group API + Odds API data only.
    """
    global _last_pipeline_result
    logger.info("API: Running prediction pipeline...")
    try:
        pipeline = PredictionPipeline()
        result = pipeline.run()
        _last_pipeline_result = result
        return result
    except PreflightError as e:
        logger.error("Preflight failed: {}", e)
        raise HTTPException(
            status_code=503,
            detail={
                "error": "PREFLIGHT_FAILED",
                "message": str(e),
                "failed_checks": e.report.to_dict()["failed_required"],
            },
        )
    except Exception as e:
        logger.exception(f"Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/legs", response_model=List[Dict], tags=["Legs"])
async def get_legs(mode: str = "value"):
    """
    Get ranked candidate legs from the last pipeline run.
    mode: 'value' (highest EV) or 'safe' (highest probability with minimum edge).
    Requires a prior POST /pipeline/run call.
    """
    global _last_pipeline_result
    if _last_pipeline_result is None:
        raise HTTPException(
            status_code=404,
            detail="No pipeline results available. POST /pipeline/run first.",
        )

    if mode == "safe":
        return _last_pipeline_result.get("safe_legs", [])
    return _last_pipeline_result.get("value_legs", [])


@router.get("/multis", response_model=List[Dict], tags=["Multis"])
async def get_multis(mode: str = "value"):
    """
    Get ranked multis from the last pipeline run.
    mode: 'value', 'safe', or 'same_game'.
    Requires a prior POST /pipeline/run call.
    """
    global _last_pipeline_result
    if _last_pipeline_result is None:
        raise HTTPException(
            status_code=404,
            detail="No pipeline results available. POST /pipeline/run first.",
        )

    if mode == "safe":
        return _last_pipeline_result.get("safe_multis", [])
    elif mode == "same_game":
        return _last_pipeline_result.get("same_game_multis", [])
    return _last_pipeline_result.get("value_multis", [])


@router.post("/multis/generate", response_model=List[Dict], tags=["Multis"])
async def generate_multis(request: MultiGenerateRequest):
    """
    Generate multis from a custom set of legs.
    Allows the UI to request specific combinations.
    """
    try:
        builder = MultiBuilder(
            max_correlation=request.max_correlation,
            min_ev=request.min_ev,
        )
        legs = []
        for l in request.legs:
            legs.append(Leg(
                leg_id=l.leg_id,
                fixture_id=l.fixture_id,
                player_id=l.player_id,
                team_id=l.team_id,
                market_type=l.market_type,
                selection=l.selection,
                decimal_odds=l.decimal_odds,
                calibrated_probability=l.model_probability,
                ev=l.ev,
                confidence_score=l.confidence_score,
                explanation=l.explanation,
            ))
        multis = builder.build(
            legs,
            n_legs=request.n_legs,
            max_results=request.max_results,
            mode=request.mode,
        )
        return [
            {
                "multi_id": m.multi_id,
                "n_legs": m.n_legs,
                "multi_type": m.multi_type,
                "combined_odds": m.combined_odds,
                "adjusted_probability": m.adjusted_probability,
                "ev": m.ev,
                "correlation_score": m.correlation_score,
                "correlation_label": m.correlation_label,
                "risk_score": m.risk_score,
                "explanation": m.explanation,
                "leg_ids": m.leg_ids,
            }
            for m in multis
        ]
    except Exception as e:
        logger.exception(f"Multi generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/training/run", response_model=Dict, tags=["Training"])
async def run_training():
    """Train all prediction models using live data."""
    global _last_training_result
    logger.info("API: Running model training...")
    try:
        svc = TrainingService()
        result = svc.run_all()
        _last_training_result = result
        return result
    except Exception as e:
        logger.exception(f"Training error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/backtest/run", response_model=Dict, tags=["Backtesting"])
async def run_backtest():
    """Run walk-forward backtesting on historical data."""
    global _last_backtest_result
    logger.info("API: Running backtest...")
    try:
        backtester = WalkForwardBacktester()
        result = backtester.run()
        _last_backtest_result = result
        return result
    except Exception as e:
        logger.exception(f"Backtest error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reports/summary", response_model=Dict, tags=["Reports"])
async def get_summary():
    """Get summary of data, models, and settings."""
    try:
        svc = ReportsService()
        return svc.get_summary()
    except Exception as e:
        logger.warning(f"Summary failed: {e}")
        # Return a minimal summary rather than crashing the UI health load
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "data_summary": {
                "total_fixtures": 0,
                "completed_fixtures": 0,
                "upcoming_fixtures": 0,
                "total_players": 0,
                "total_odds_records": 0,
                "seasons": [],
            },
            "models_available": [],
            "settings": {
                "min_edge_threshold": settings.min_edge_threshold,
                "min_ev_threshold": settings.min_ev_threshold,
                "max_correlation_score": settings.max_correlation_score,
                "max_legs_per_game": settings.max_legs_per_game,
                "max_multi_legs": settings.max_multi_legs,
            },
            "error": str(e),
        }


@router.get("/reports/backtest", response_model=Dict, tags=["Reports"])
async def get_backtest_report():
    """Get last backtest results."""
    global _last_backtest_result
    if _last_backtest_result is None:
        raise HTTPException(status_code=404, detail="No backtest results yet. POST /backtest/run first.")
    return _last_backtest_result


@router.get("/reports/training", response_model=Dict, tags=["Reports"])
async def get_training_report():
    """Get last training results."""
    global _last_training_result
    if _last_training_result is None:
        raise HTTPException(status_code=404, detail="No training results yet. POST /training/run first.")
    return _last_training_result


# ---------------------------------------------------------------------------
# Sync, Bootstrap, Quota endpoints
# ---------------------------------------------------------------------------

@router.post("/sync/upcoming", response_model=Dict, tags=["Sync"])
async def sync_upcoming(lookahead_days: int = 14):
    """Pull upcoming fixtures from AFL Data Sports Group API and upsert to DB."""
    try:
        from app.services.sync import SyncService
        svc = SyncService()
        return svc.sync_upcoming(lookahead_days=lookahead_days)
    except Exception as exc:
        logger.exception("sync_upcoming error: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/sync/settle_recent", response_model=Dict, tags=["Sync"])
async def sync_settle_recent(lookback_days: int = 7):
    """Fetch completed fixtures and settle open predictions."""
    try:
        from app.services.sync import SyncService
        svc = SyncService()
        return svc.sync_settle_recent(lookback_days=lookback_days)
    except Exception as exc:
        logger.exception("sync_settle_recent error: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/sync/status", response_model=Dict, tags=["Sync"])
async def sync_status():
    """Return current data-source and sync status."""
    try:
        from app.services.sync import SyncService
        svc = SyncService()
        return svc.get_sync_status()
    except Exception as exc:
        logger.exception("sync_status error: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/bootstrap/run", response_model=Dict, tags=["Bootstrap"])
async def run_bootstrap(force_retrain: bool = False, background_tasks: BackgroundTasks = None):
    """Run the full daily bootstrap research cycle.

    Steps: sync → settle → evaluate → (retrain) → (promote) → predict.
    Set force_retrain=true to skip criteria checks and always retrain.
    """
    try:
        from app.services.bootstrap import BootstrapCycleService
        svc = BootstrapCycleService()
        return svc.run(force_retrain=force_retrain)
    except Exception as exc:
        logger.exception("bootstrap/run error: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/quota/status", response_model=Dict, tags=["Quota"])
async def quota_status():
    """Return AFL Data API status (unlimited calls — no quota)."""
    return {
        "configured": settings.is_afl_data_configured,
        "provider": "AFL Data Sports Group",
        "call_rate": "unlimited",
        "message": "AFL Data Sports Group API has unlimited call rate — no quota tracking needed.",
    }


@router.get("/reports/bootstrap", response_model=Dict, tags=["Reports"])
async def get_bootstrap_report(limit: int = 10):
    """Return the last N bootstrap cycle logs."""
    try:
        from app.db.database import SessionLocal
        from app.db.models import ResearchCycleLog
        with SessionLocal() as db:
            rows = (
                db.query(ResearchCycleLog)
                .order_by(ResearchCycleLog.started_at.desc())
                .limit(limit)
                .all()
            )
            return {
                "cycles": [
                    {
                        "cycle_id": r.cycle_id,
                        "started_at": r.started_at.isoformat() if r.started_at else None,
                        "elapsed_seconds": r.elapsed_seconds,
                        "n_fixtures_synced": r.n_fixtures_synced,
                        "n_legs_settled": r.n_legs_settled,
                        "brier_score": r.brier_score,
                        "roi": r.roi,
                        "retrained": r.retrained,
                        "promoted": r.promoted,
                        "summary": r.summary,
                    }
                    for r in rows
                ]
            }
    except Exception as exc:
        logger.exception("reports/bootstrap error: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/reports/evaluation", response_model=Dict, tags=["Reports"])
async def get_evaluation_report(lookback_days: int = 30):
    """Run and return a self-evaluation report over recent settled predictions."""
    try:
        from app.services.evaluation import EvaluationService
        svc = EvaluationService()
        return svc.evaluate(lookback_days=lookback_days)
    except Exception as exc:
        logger.exception("reports/evaluation error: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc))
