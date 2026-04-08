"""FastAPI route handlers."""
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
from app.correlation.engine import Leg
from app.optimizer.multi_builder import MultiBuilder

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
