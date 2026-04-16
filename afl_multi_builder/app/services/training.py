"""
Model training service.
Orchestrates feature engineering, train/val split, model fitting,
calibration, and artifact saving.
"""
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from loguru import logger

from sklearn.model_selection import train_test_split

from app.features.pipeline import FeaturePipeline
from app.pricing.models import (
    EnsembleModel, PlayerDisposalsModel, CalibratedModel, ModelRegistry
)
from app.pricing.calibration import IsotonicCalibrator, evaluate_calibration
from app.core.metrics import compute_brier_score, compute_log_loss
from app.core.config import settings
from app.data_ingestion.loader import DataLoader


class TrainingService:
    """
    Trains match-level and player-level models.
    Saves artifacts and training metadata.
    """

    def __init__(self, loader: Optional[DataLoader] = None):
        self.loader = loader or DataLoader()
        self.pipeline = FeaturePipeline(self.loader)
        self.registry = ModelRegistry()
        self.run_id = str(uuid.uuid4())[:8]

    def run_match_model_training(
        self,
        test_size: float = 0.2,
        val_size: float = 0.15,
        force_retrain: bool = False,
    ) -> Dict:
        """
        Train the match win probability ensemble model.
        Returns training metadata dict.
        """
        logger.info(f"[{self.run_id}] Starting match model training...")

        X, y, feature_cols = self.pipeline.get_model_ready_match_data()

        if X.empty or len(y) < 20:
            logger.warning("Insufficient match data for training.")
            return {"status": "insufficient_data", "n_samples": len(y)}

        logger.info(f"Training with {len(y)} samples, {len(feature_cols)} features.")

        # Chronological split (no shuffle — preserve time order)
        n = len(X)
        val_end = int(n * (1 - test_size))
        val_start = int(val_end * (1 - val_size / (1 - test_size)))

        X_train = X.iloc[:val_start]
        y_train = y.iloc[:val_start]
        X_val = X.iloc[val_start:val_end]
        y_val = y.iloc[val_start:val_end]
        X_test = X.iloc[val_end:]
        y_test = y.iloc[val_end:]

        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        # Train calibrated ensemble
        ensemble = EnsembleModel()
        calibrated = CalibratedModel(ensemble)

        if len(X_val) < 5:
            # Not enough held-out data — train base model only, skip calibration.
            # Calibrating on training data causes data leakage and overconfident probs.
            logger.warning(
                "Validation set too small ({} samples) — calibration skipped to avoid leakage. "
                "Collect more data before relying on calibrated probabilities.",
                len(X_val),
            )
            ensemble.fit(X_train, y_train)
            calibrated.base_model = ensemble
            # Leave calibrator unfitted — it will pass through raw probs unchanged,
            # which is honest about the lack of calibration data.
        else:
            calibrated.fit(X_train, y_train, X_val, y_val)

        # Test set evaluation
        test_metrics = {}
        if len(X_test) >= 5:
            raw_test = calibrated.predict_proba_raw(X_test)[:, 1]
            cal_test = calibrated.predict_proba(X_test)[:, 1]
            test_metrics = {
                "raw_brier": round(compute_brier_score(y_test.values, raw_test), 5),
                "calibrated_brier": round(compute_brier_score(y_test.values, cal_test), 5),
                "raw_logloss": round(compute_log_loss(y_test.values, raw_test), 5),
                "calibrated_logloss": round(compute_log_loss(y_test.values, cal_test), 5),
            }
            logger.info(f"Test metrics: {test_metrics}")

        # Save
        model_path = self.registry.save("match_win_ensemble", calibrated)
        fi = calibrated.feature_importance()

        metadata = {
            "run_id": self.run_id,
            "model_name": "match_win_ensemble",
            "model_type": "CalibratedEnsemble",
            "market_type": "head_to_head",
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_test": len(X_test),
            "feature_names": feature_cols,
            "test_metrics": test_metrics,
            "feature_importance": {k: round(v, 5) for k, v in list(fi.items())[:10]},
            "calibration_report": calibrated.calibration_report,
            "artifact_path": str(model_path),
            "timestamp": datetime.utcnow().isoformat(),
            "status": "success",
        }

        logger.info(f"[{self.run_id}] Match model training complete.")
        return metadata

    def run_player_disposals_training(
        self,
        default_line: float = 22.5,
        test_size: float = 0.2,
    ) -> Dict:
        """
        Train player disposals over/under model.
        Returns training metadata.
        """
        logger.info(f"[{self.run_id}] Starting player disposals model training...")

        X, y, feature_cols = self.pipeline.get_model_ready_player_data(
            line=default_line, stat_col="disposals"
        )

        if X.empty or len(y) < 15:
            logger.warning("Insufficient player data for training.")
            return {"status": "insufficient_data", "n_samples": len(y)}

        logger.info(f"Player model training with {len(y)} samples.")

        n = len(X)
        split_idx = int(n * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Train regression model
        player_features_full = self.pipeline.get_player_features("disposals")
        valid = player_features_full[player_features_full["target"].notna()]

        regression_model = PlayerDisposalsModel()
        y_reg = valid.iloc[:split_idx]["target"]
        regression_model.fit(X_train, y_reg)

        # For over/under calibration, wrap with calibrator
        calibrator = IsotonicCalibrator()
        if len(X_train) >= 10:
            raw_probs = regression_model.predict_over_prob(X_train, default_line)
            calibrator.fit(raw_probs, y_train.values)

        # Test evaluation
        test_metrics = {}
        if len(X_test) >= 5:
            raw_probs_test = regression_model.predict_over_prob(X_test, default_line)
            cal_probs_test = calibrator.calibrate(raw_probs_test)
            test_metrics = {
                "raw_brier": round(compute_brier_score(y_test.values, np.clip(raw_probs_test, 1e-7, 1-1e-7)), 5),
                "calibrated_brier": round(compute_brier_score(y_test.values, np.clip(cal_probs_test, 1e-7, 1-1e-7)), 5),
            }

        # Save both artifacts
        self.registry.save("player_disposals_model", regression_model)
        self.registry.save("player_disposals_calibrator", calibrator)

        metadata = {
            "run_id": self.run_id,
            "model_name": "player_disposals_model",
            "model_type": "XGBoostRegressor+IsotonicCalibrator",
            "market_type": "player_disposals",
            "n_train": len(X_train),
            "n_test": len(X_test),
            "default_line": default_line,
            "residual_std": regression_model._residual_std,
            "feature_names": feature_cols,
            "test_metrics": test_metrics,
            "feature_importance": {k: round(v, 5) for k, v in list(regression_model.feature_importance().items())[:10]},
            "timestamp": datetime.utcnow().isoformat(),
            "status": "success",
        }

        logger.info(f"[{self.run_id}] Player model training complete.")
        return metadata

    def run_all(self) -> Dict:
        """Run full training pipeline for all models."""
        results = {}
        results["match_model"] = self.run_match_model_training()
        results["player_disposals_model"] = self.run_player_disposals_training()
        results["run_id"] = self.run_id
        results["timestamp"] = datetime.utcnow().isoformat()
        return results
