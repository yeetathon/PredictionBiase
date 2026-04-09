"""Smoke tests for the full pipeline, training, and backtest services.

All tests inject DemoAFLDataProvider so they never need a live Sportradar
API key and never touch the network.
"""
import pytest
from pathlib import Path

# ── Demo data path ────────────────────────────────────────────────────────────
DEMO_DATA = Path(__file__).parent / "demo_data"


def _make_loader():
    """Return a DataLoader backed by local demo CSVs."""
    from app.data_ingestion.demo_loader import (
        DemoAFLDataProvider,
        DemoOddsProvider,
        DemoWeatherProvider,
        DemoInjuryProvider,
    )
    from app.data_ingestion.loader import DataLoader

    return DataLoader(
        afl_provider=DemoAFLDataProvider(DEMO_DATA),
        odds_provider=DemoOddsProvider(DEMO_DATA),
        weather_provider=DemoWeatherProvider(DEMO_DATA),
        injury_provider=DemoInjuryProvider(DEMO_DATA),
    )


class TestPipelineSmoke:
    def test_pipeline_runs_without_error(self):
        """Full pipeline should complete without raising."""
        from app.services.pipeline import PredictionPipeline
        pipeline = PredictionPipeline(loader=_make_loader())
        result = pipeline.run()
        assert "run_id" in result
        assert "n_fixtures" in result
        assert "value_legs" in result
        assert "value_multis" in result
        assert isinstance(result["value_legs"], list)
        assert isinstance(result["value_multis"], list)

    def test_pipeline_has_legs_or_explains_why_not(self):
        """Pipeline either returns legs or reports zero candidates."""
        from app.services.pipeline import PredictionPipeline
        pipeline = PredictionPipeline(loader=_make_loader())
        result = pipeline.run()
        assert isinstance(result["n_candidate_legs"], int)
        assert result["n_candidate_legs"] >= 0

    def test_pipeline_elapsed_is_positive(self):
        from app.services.pipeline import PredictionPipeline
        pipeline = PredictionPipeline(loader=_make_loader())
        result = pipeline.run()
        assert result["elapsed_seconds"] > 0


class TestTrainingSmoke:
    def test_training_runs(self):
        """Training service should complete."""
        from app.services.training import TrainingService
        svc = TrainingService(loader=_make_loader())
        result = svc.run_match_model_training()
        assert "status" in result
        assert result["status"] in ("success", "insufficient_data")

    def test_player_training_runs(self):
        from app.services.training import TrainingService
        svc = TrainingService(loader=_make_loader())
        result = svc.run_player_disposals_training()
        assert "status" in result

    def test_full_training_returns_run_id(self):
        from app.services.training import TrainingService
        svc = TrainingService(loader=_make_loader())
        result = svc.run_all()
        assert "run_id" in result
        assert len(result["run_id"]) == 8


class TestBacktestSmoke:
    def test_backtest_runs(self):
        """Backtest should complete and return structured results."""
        from app.services.backtest import WalkForwardBacktester
        backtester = WalkForwardBacktester(
            loader=_make_loader(), min_train_samples=20, step_size=5
        )
        result = backtester.run()
        assert "run_id" in result
        assert "status" in result
        assert result["status"] in ("success", "insufficient_data", "no_predictions")

    def test_backtest_metrics_in_range(self):
        """If backtest succeeds, metrics should be in valid ranges."""
        from app.services.backtest import WalkForwardBacktester
        backtester = WalkForwardBacktester(
            loader=_make_loader(), min_train_samples=20, step_size=5
        )
        result = backtester.run()
        if result.get("status") == "success":
            metrics = result["overall_metrics"]
            assert 0 <= metrics["brier_score"] <= 1
            assert metrics["log_loss"] > 0
            assert 0 <= metrics["accuracy"] <= 1
