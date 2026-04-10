"""FastAPI route handlers."""
from datetime import datetime
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from loguru import logger

from app.schemas.models import (
    HealthResponse, PipelineRunRequest, PipelineResponse,
    MultiGenerateRequest, MultiResponse, LegResponse,
    TrainingRunResponse, BacktestRunResponse, SummaryResponse,
    GenerateRequest,
)
from app.services.pipeline import PredictionPipeline
from app.services.training import TrainingService
from app.services.backtest import WalkForwardBacktester
from app.services.reports import ReportsService
from app.services.generate_service import GenerateService, STYLE_PROFILES
from app.correlation.engine import Leg
from app.optimizer.multi_builder import MultiBuilder
from app.core.config import settings

router = APIRouter()

# In-memory cache for last pipeline results (for demo simplicity)
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


@router.post("/pipeline/run", response_model=Dict, tags=["Pipeline"])
async def run_pipeline(request: PipelineRunRequest = None):
    """
    Run the full prediction pipeline.
    Generates candidate legs and multis for upcoming fixtures.
    """
    global _last_pipeline_result
    logger.info("API: Running prediction pipeline...")
    try:
        pipeline = PredictionPipeline()
        result = pipeline.run()
        _last_pipeline_result = result
        return result
    except Exception as e:
        logger.exception(f"Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/legs", response_model=List[Dict], tags=["Legs"])
async def get_legs(mode: str = "value"):
    """
    Get ranked candidate legs.
    mode: 'value' (highest EV) or 'safe' (highest probability with minimum edge).
    """
    global _last_pipeline_result
    if _last_pipeline_result is None:
        # Auto-run pipeline
        pipeline = PredictionPipeline()
        _last_pipeline_result = pipeline.run()

    if mode == "safe":
        return _last_pipeline_result.get("safe_legs", [])
    return _last_pipeline_result.get("value_legs", [])


@router.get("/multis", response_model=List[Dict], tags=["Multis"])
async def get_multis(mode: str = "value"):
    """
    Get ranked multis.
    mode: 'value', 'safe', or 'same_game'.
    """
    global _last_pipeline_result
    if _last_pipeline_result is None:
        pipeline = PredictionPipeline()
        _last_pipeline_result = pipeline.run()

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
        # Convert request legs to Leg objects
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
    """Train all prediction models."""
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
    svc = ReportsService()
    return svc.get_summary()


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
# Phase-2: Sync, Bootstrap, Quota endpoints
# ---------------------------------------------------------------------------

@router.post("/sync/upcoming", response_model=Dict, tags=["Sync"])
async def sync_upcoming(lookahead_days: int = 14):
    """Pull upcoming fixtures from Sportradar (or demo) and upsert to DB."""
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
    Set ``force_retrain=true`` to skip criteria checks and always retrain.
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
    """Return Sportradar API quota usage."""
    if not settings.is_sportradar_configured:
        return {
            "configured": False,
            "message": "Sportradar API key not configured. Running in demo mode.",
            "data_mode": settings.effective_data_mode,
        }
    try:
        from app.data_ingestion.quota_manager import QuotaManager
        qm = QuotaManager()
        return {"configured": True, **qm.get_status()}
    except Exception as exc:
        logger.exception("quota/status error: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc))


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


# ---------------------------------------------------------------------------
# Selection-driven generation workflow  (new dashboard flow)
# ---------------------------------------------------------------------------

@router.get("/fixtures/weekend", response_model=Dict, tags=["Generate"])
async def get_weekend_fixtures():
    """
    Return AFL fixtures for the current or next upcoming round.
    Used by the dashboard to populate the fixture selection list.
    """
    try:
        fixtures = GenerateService.get_weekend_fixtures()
        styles = GenerateService.get_style_profiles()

        # Round metadata
        round_no = fixtures[0]["round"] if fixtures else 0
        season = fixtures[0]["season"] if fixtures else 0

        return {
            "round": round_no,
            "season": season,
            "n_fixtures": len(fixtures),
            "fixtures": fixtures,
            "styles": styles,
        }
    except Exception as exc:
        logger.exception("fixtures/weekend error: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/generate", response_model=Dict, tags=["Generate"])
async def start_generate(request: GenerateRequest):
    """
    Start an async multi-generation job for the selected fixtures.

    Returns a job_id immediately. Poll /generate/{job_id}/status for progress
    and results.

    Body:
        fixture_ids: list of fixture IDs to include
        multi_style: one of safest | balanced | value | aggressive | longshot
        n_multis:    exact number of multis to generate (1–20)
    """
    try:
        job_id = GenerateService.start_job(
            fixture_ids=request.fixture_ids,
            multi_style=request.multi_style,
            n_multis=request.n_multis,
        )
        logger.info(
            "Started generate job %s: fixtures=%s style=%s n=%d",
            job_id, request.fixture_ids, request.multi_style, request.n_multis,
        )
        return {"job_id": job_id, "status": "pending"}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("generate start error: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/generate/{job_id}/status", response_model=Dict, tags=["Generate"])
async def get_generate_status(job_id: str):
    """
    Poll the status of a generate job.

    Returns:
        status:   pending | running | complete | error
        stages:   list of pipeline stage progress objects
        result:   populated when status == complete
        error:    populated when status == error
    """
    data = GenerateService.get_job_status(job_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return data
