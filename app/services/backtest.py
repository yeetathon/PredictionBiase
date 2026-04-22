"""
Walk-forward backtesting service.

Approach:
  - Sort all completed fixtures chronologically
  - Use an expanding window: train on data up to window, test on next batch
  - No leakage: test-window data is never in the training set
  - Measure: Brier score, log loss, hit rate, ROI at various edge thresholds
  - CLV-style evaluation: compare model probability to pre-game market odds
"""
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from loguru import logger

from app.features.pipeline import FeaturePipeline
from app.pricing.models import EnsembleModel, CalibratedModel, ModelRegistry
from app.pricing.calibration import IsotonicCalibrator, evaluate_calibration
from app.core.metrics import (
    compute_brier_score, compute_log_loss, compute_roi, compute_ev
)
from app.core.config import settings
from app.data_ingestion.loader import DataLoader


class WalkForwardBacktester:
    """
    Walk-forward backtester.
    Trains iteratively on expanding windows and records out-of-sample metrics.
    """

    def __init__(
        self,
        loader: Optional[DataLoader] = None,
        min_train_samples: int = 30,
        step_size: int = 9,  # games per step (one round)
        edge_threshold: float = None,
    ):
        self.loader = loader or DataLoader()
        self.pipeline = FeaturePipeline(self.loader)
        self.registry = ModelRegistry()
        self.min_train_samples = min_train_samples
        self.step_size = step_size
        self.edge_threshold = edge_threshold or settings.min_edge_threshold
        self.run_id = str(uuid.uuid4())[:8]

    def run(self) -> Dict:
        """
        Execute walk-forward backtest.
        Returns a comprehensive results dictionary.
        """
        logger.info(f"[{self.run_id}] Starting walk-forward backtest...")

        X, y, feature_cols = self.pipeline.get_model_ready_match_data()
        if X.empty or len(y) < self.min_train_samples + self.step_size:
            logger.warning("Insufficient data for backtesting.")
            return {
                "status": "insufficient_data",
                "n_samples": len(y),
                "run_id": self.run_id,
            }

        n = len(X)
        logger.info(f"Backtest on {n} samples with {len(feature_cols)} features.")

        # Get market odds for CLV comparison
        odds_df = self.loader.load_odds_df()
        fixtures = self.loader.load_fixtures_df()

        # Build fixture -> market prob mapping
        market_probs = self._build_market_prob_map(odds_df, fixtures)

        # Walk-forward loop
        all_predictions = []
        fold_metrics = []

        train_end = self.min_train_samples
        while train_end < n:
            test_end = min(train_end + self.step_size, n)

            X_train = X.iloc[:train_end]
            y_train = y.iloc[:train_end]
            X_test = X.iloc[train_end:test_end]
            y_test = y.iloc[train_end:test_end]

            if len(X_test) == 0:
                break

            # Train model on expanding window
            model = EnsembleModel()
            try:
                if len(X_train) >= 20:
                    val_size = max(5, int(len(X_train) * 0.15))
                    X_tr = X_train.iloc[:-val_size]
                    y_tr = y_train.iloc[:-val_size]
                    X_v = X_train.iloc[-val_size:]
                    y_v = y_train.iloc[-val_size:]
                    calibrated = CalibratedModel(model)
                    calibrated.fit(X_tr, y_tr, X_v, y_v)
                else:
                    model.fit(X_train, y_train)
                    calibrator = IsotonicCalibrator()
                    raw = model.predict_proba(X_train)[:, 1]
                    calibrator.fit(raw, y_train)
                    calibrated = CalibratedModel(model, calibrator)
            except Exception as e:
                logger.warning(f"Training failed at fold {train_end}: {e}")
                train_end += self.step_size
                continue

            # Predict
            try:
                raw_probs = calibrated.predict_proba_raw(X_test)[:, 1]
                cal_probs = calibrated.predict_proba(X_test)[:, 1]
            except Exception as e:
                logger.warning(f"Prediction failed: {e}")
                train_end += self.step_size
                continue

            # Record predictions
            test_indices = X_test.index.tolist()
            for i, idx in enumerate(range(train_end, test_end)):
                fixture_id = None
                try:
                    # Try to retrieve fixture_id from the features DataFrame
                    full_features = self.pipeline.get_team_features()
                    completed = full_features[full_features["status"] == "completed"]
                    if idx < len(completed):
                        fixture_id = int(completed.iloc[idx]["fixture_id"])
                except Exception:
                    pass

                market_prob = market_probs.get(fixture_id, None)

                all_predictions.append({
                    "idx": idx,
                    "fixture_id": fixture_id,
                    "y_true": int(y_test.iloc[i]),
                    "raw_prob": float(raw_probs[i]),
                    "cal_prob": float(cal_probs[i]),
                    "market_prob": market_prob,
                    "edge": float(cal_probs[i] - market_prob) if market_prob else None,
                    "train_size": train_end,
                })

            # Fold metrics (skip if only one class in test set)
            if len(y_test) >= 3 and len(np.unique(y_test.values)) > 1:
                try:
                    fold_metrics.append({
                        "fold": train_end,
                        "n_train": train_end,
                        "n_test": len(y_test),
                        "brier": round(compute_brier_score(y_test.values, cal_probs), 5),
                        "logloss": round(compute_log_loss(y_test.values, cal_probs), 5),
                        "accuracy": round(float(np.mean((cal_probs >= 0.5) == y_test.values)), 4),
                    })
                except Exception:
                    pass

            train_end += self.step_size

        if not all_predictions:
            return {"status": "no_predictions", "run_id": self.run_id}

        pred_df = pd.DataFrame(all_predictions)

        # Overall metrics (only on rows with both classes present)
        y_all = pred_df["y_true"].values
        cal_all = pred_df["cal_prob"].values
        raw_all = pred_df["raw_prob"].values
        overall_brier = round(compute_brier_score(y_all, cal_all), 5)
        # log_loss requires both classes
        if len(np.unique(y_all)) > 1:
            overall_logloss = round(compute_log_loss(y_all, cal_all), 5)
        else:
            overall_logloss = None
        overall_accuracy = round(float(np.mean((cal_all >= 0.5) == y_all)), 4)
        try:
            calibration_report = evaluate_calibration(y_all, raw_all, cal_all)
        except Exception:
            calibration_report = {"status": "insufficient_classes"}

        # Simulated betting ROI at various edge thresholds
        roi_results = self._compute_roi_by_edge(pred_df)

        # CLV analysis
        clv_results = self._compute_clv_analysis(pred_df)

        result = {
            "run_id": self.run_id,
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            "n_total_predictions": len(pred_df),
            "n_folds": len(fold_metrics),
            "overall_metrics": {
                "brier_score": overall_brier,
                "log_loss": overall_logloss if overall_logloss is not None else "N/A",
                "accuracy": overall_accuracy,
            },
            "calibration_report": calibration_report,
            "roi_by_edge_threshold": roi_results,
            "clv_analysis": clv_results,
            "fold_metrics": fold_metrics,
            "feature_count": len(feature_cols),
        }

        logger.info(
            f"[{self.run_id}] Backtest complete. "
            f"Brier: {overall_brier:.4f}, LogLoss: {overall_logloss:.4f}, "
            f"Accuracy: {overall_accuracy:.1%}"
        )

        return result

    def _build_market_prob_map(
        self, odds_df: pd.DataFrame, fixtures: pd.DataFrame
    ) -> Dict[int, float]:
        """Build fixture_id -> vig-adjusted home win market probability."""
        if odds_df.empty:
            return {}

        h2h = odds_df[odds_df["market_type"] == "head_to_head"]
        result = {}
        for fid in h2h["fixture_id"].unique():
            fx_odds = h2h[h2h["fixture_id"] == fid]
            home_odds_rows = fx_odds[fx_odds["selection"] == "home_win"]
            away_odds_rows = fx_odds[fx_odds["selection"] == "away_win"]
            if home_odds_rows.empty or away_odds_rows.empty:
                continue
            best_home = float(home_odds_rows["decimal_odds"].max())
            best_away = float(away_odds_rows["decimal_odds"].max())
            raw_home = 1.0 / best_home
            raw_away = 1.0 / best_away
            total = raw_home + raw_away
            if total > 0:
                result[int(fid)] = round(raw_home / total, 5)
        return result

    def _compute_roi_by_edge(self, pred_df: pd.DataFrame) -> List[Dict]:
        """Compute ROI assuming flat stake at each edge threshold."""
        results = []
        # Get market implied probs
        edge_col = pred_df["edge"].dropna()
        if edge_col.empty:
            return []

        thresholds = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
        for thresh in thresholds:
            bets = pred_df[pred_df["edge"].notna() & (pred_df["edge"] >= thresh)]
            if len(bets) < 3:
                continue
            # Simulate: bet on home team when edge >= thresh
            # Use 1/market_prob as proxy for odds
            outcomes = bets["y_true"].values
            # approximate decimal odds from market prob
            market_probs = bets["market_prob"].values
            decimal_odds = np.where(market_probs > 0, 1.0 / market_probs, 2.0)
            roi_dict = compute_roi(outcomes, decimal_odds)
            results.append({
                "edge_threshold": thresh,
                **roi_dict,
            })

        return results

    def _compute_clv_analysis(self, pred_df: pd.DataFrame) -> Dict:
        """
        Closing line value analysis.
        Compare model probability to market probability.
        Positive CLV = model is consistently finding underpriced outcomes.
        """
        has_market = pred_df[pred_df["market_prob"].notna()].copy()
        if len(has_market) < 5:
            return {"status": "insufficient_market_data"}

        has_market["model_vs_market"] = has_market["cal_prob"] - has_market["market_prob"]
        avg_clv = float(has_market["model_vs_market"].mean())
        pct_positive = float((has_market["model_vs_market"] > 0).mean())

        return {
            "n_with_market_data": int(len(has_market)),
            "avg_model_vs_market": round(avg_clv, 5),
            "pct_positive_edge": round(pct_positive, 4),
            "interpretation": (
                "Model consistently finds positive edge vs market"
                if avg_clv > 0.02 else
                "Model marginally differs from market" if avg_clv > 0
                else "Model underperforms vs market on average"
            ),
        }
