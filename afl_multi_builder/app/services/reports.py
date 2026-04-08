"""Report generation service: summarises pipeline, backtest, and training results."""
from datetime import datetime
from typing import Dict, List, Optional
from loguru import logger

from app.data_ingestion.loader import DataLoader
from app.pricing.models import ModelRegistry
from app.core.config import settings


class ReportsService:
    """Generates summary reports for the UI and API."""

    def __init__(self, loader: Optional[DataLoader] = None):
        self.loader = loader or DataLoader()
        self.registry = ModelRegistry()

    def get_summary(self) -> Dict:
        """Return a high-level summary of data and model status."""
        fixtures = self.loader.load_fixtures_df()
        players = self.loader.load_players_df()
        odds = self.loader.load_odds_df()

        completed = fixtures[fixtures["status"] == "completed"] if not fixtures.empty else fixtures
        upcoming = fixtures[fixtures["status"] == "upcoming"] if not fixtures.empty else fixtures

        models_available = self.registry.list_models()

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "data_summary": {
                "total_fixtures": int(len(fixtures)),
                "completed_fixtures": int(len(completed)),
                "upcoming_fixtures": int(len(upcoming)),
                "total_players": int(len(players)),
                "total_odds_records": int(len(odds)),
                "seasons": sorted(fixtures["season"].unique().tolist()) if not fixtures.empty else [],
            },
            "models_available": models_available,
            "settings": {
                "min_edge_threshold": settings.min_edge_threshold,
                "min_ev_threshold": settings.min_ev_threshold,
                "max_correlation_score": settings.max_correlation_score,
                "max_legs_per_game": settings.max_legs_per_game,
                "max_multi_legs": settings.max_multi_legs,
            },
        }

    def format_backtest_report(self, backtest_results: Dict) -> str:
        """Format backtest results as readable text report."""
        if backtest_results.get("status") != "success":
            return f"Backtest failed: {backtest_results.get('status')}"

        lines = [
            "=" * 60,
            "AFL PREDICTION MODEL - BACKTEST REPORT",
            "=" * 60,
            f"Run ID: {backtest_results.get('run_id')}",
            f"Timestamp: {backtest_results.get('timestamp')}",
            f"Total predictions: {backtest_results.get('n_total_predictions')}",
            f"Walk-forward folds: {backtest_results.get('n_folds')}",
            "",
            "── Overall Metrics ──────────────────────────────────",
        ]

        metrics = backtest_results.get("overall_metrics", {})
        lines.append(f"  Brier Score:  {metrics.get('brier_score', 'N/A'):.4f}  (0=perfect, 0.25=baseline)")
        lines.append(f"  Log Loss:     {metrics.get('log_loss', 'N/A'):.4f}")
        lines.append(f"  Accuracy:     {metrics.get('accuracy', 'N/A'):.1%}")

        cal = backtest_results.get("calibration_report", {})
        lines += [
            "",
            "── Calibration ──────────────────────────────────────",
            f"  Raw Brier:         {cal.get('raw_brier_score', 'N/A')}",
            f"  Calibrated Brier:  {cal.get('calibrated_brier_score', 'N/A')}",
            f"  Brier Improvement: {cal.get('brier_improvement', 'N/A')}",
        ]

        roi_results = backtest_results.get("roi_by_edge_threshold", [])
        if roi_results:
            lines += ["", "── Simulated ROI by Edge Threshold ─────────────────"]
            lines.append(f"  {'Edge':>8}  {'N Bets':>7}  {'Hit Rate':>9}  {'ROI':>8}  {'Avg Odds':>9}")
            lines.append("  " + "-" * 50)
            for r in roi_results:
                lines.append(
                    f"  {r['edge_threshold']:>7.0%}  {r['n_bets']:>7}  "
                    f"{r['hit_rate']:>9.1%}  {r['roi']:>7.1%}  {r['avg_odds']:>9.2f}"
                )

        clv = backtest_results.get("clv_analysis", {})
        if clv.get("status") != "insufficient_market_data":
            lines += [
                "",
                "── CLV Analysis ─────────────────────────────────────",
                f"  Fixtures with market data: {clv.get('n_with_market_data', 0)}",
                f"  Avg model vs market:       {clv.get('avg_model_vs_market', 0):+.1%}",
                f"  % with positive edge:      {clv.get('pct_positive_edge', 0):.1%}",
                f"  {clv.get('interpretation', '')}",
            ]

        lines.append("=" * 60)
        return "\n".join(lines)

    def format_training_report(self, training_results: Dict) -> str:
        """Format training results as text report."""
        lines = [
            "=" * 60,
            "AFL PREDICTION MODEL - TRAINING REPORT",
            "=" * 60,
            f"Run ID: {training_results.get('run_id')}",
            f"Timestamp: {training_results.get('timestamp')}",
            "",
        ]

        for model_key in ["match_model", "player_disposals_model"]:
            r = training_results.get(model_key, {})
            if not r:
                continue
            lines.append(f"── {r.get('model_name', model_key)} ──")
            lines.append(f"  Type:       {r.get('model_type', 'unknown')}")
            lines.append(f"  Train/Val:  {r.get('n_train', 0)}/{r.get('n_val', 0)}")
            metrics = r.get("test_metrics", {})
            if metrics:
                lines.append(f"  Test Brier (raw):  {metrics.get('raw_brier', 'N/A')}")
                lines.append(f"  Test Brier (cal):  {metrics.get('calibrated_brier', 'N/A')}")
            fi = r.get("feature_importance", {})
            if fi:
                lines.append("  Top features:")
                for feat, score in list(fi.items())[:5]:
                    lines.append(f"    {feat}: {score:.4f}")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)
