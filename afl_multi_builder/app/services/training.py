"""
Model training service.
Orchestrates feature engineering, out-of-fold (OOF) cross-validation,
model fitting, calibration, signal weight learning, and artifact saving.

OOF framework (v2):
  - TimeSeriesSplit(n_splits=5) replaces the old simple train/val/test split.
  - IsotonicCalibrator is fitted on the full OOF probability vector (not a
    single held-out val set), eliminating the calibration bias introduced by
    a small validation window.
  - Per-signal Brier scores are derived from OOF folds and stored in the
    SignalWeightStore so the pricing layer can up-weight reliable signals.
  - Champion/challenger promotion: a new model is only promoted when its OOF
    Brier score beats the current champion by at least
    settings.model_promotion_brier_improvement.
"""
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from sklearn.model_selection import TimeSeriesSplit

from app.features.pipeline import FeaturePipeline
from app.pricing.models import (
    EnsembleModel, PlayerDisposalsModel, CalibratedModel, ModelRegistry,
)
from app.pricing.calibration import IsotonicCalibrator, evaluate_calibration
from app.pricing.signal_engine import SignalEngine
from app.pricing.signal_weights import get_signal_weight_store
from app.core.metrics import compute_brier_score, compute_log_loss
from app.core.config import settings
from app.data_ingestion.loader import DataLoader


class TrainingService:
    """
    Trains match-level and player-level models.
    Saves artifacts and training metadata.
    """

    _N_OOF_SPLITS = 5

    def __init__(self, loader: Optional[DataLoader] = None):
        self.loader = loader or DataLoader()
        self.pipeline = FeaturePipeline(self.loader)
        self.registry = ModelRegistry()
        self.run_id = str(uuid.uuid4())[:8]

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def run_match_model_training(
        self,
        test_size: float = 0.2,
        force_retrain: bool = False,
    ) -> Dict:
        """
        Train the match win probability ensemble model using OOF cross-validation.

        Steps:
          1. Build model-ready features (chronologically ordered).
          2. Run TimeSeriesSplit OOF loop → accumulate OOF probabilities.
          3. Fit IsotonicCalibrator on the full OOF probability vector.
          4. Retrain final EnsembleModel on ALL data; wrap with OOF calibrator.
          5. Evaluate per-signal Brier scores via OOF → update SignalWeightStore.
          6. Champion/challenger check before saving.

        Returns training metadata dict including oof_brier, oof_logloss,
        signal_weight_update, and n_oof_folds.
        """
        logger.info(f"[{self.run_id}] Starting match model training (OOF framework)...")

        X, y, feature_cols = self.pipeline.get_model_ready_match_data()

        if X.empty or len(y) < 20:
            logger.warning("Insufficient match data for training.")
            return {"status": "insufficient_data", "n_samples": len(y)}

        logger.info(
            f"[{self.run_id}] Training with {len(y)} samples, {len(feature_cols)} features."
        )

        # ── Step 1: OOF loop ──────────────────────────────────────────────
        tscv = TimeSeriesSplit(n_splits=self._N_OOF_SPLITS)
        oof_probs = np.zeros(len(X))
        oof_covered = np.zeros(len(X), dtype=bool)   # which indices have OOF preds

        fold_metrics: List[Dict] = []
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr = X.iloc[train_idx]
            X_v = X.iloc[val_idx]
            y_tr = y.iloc[train_idx]
            y_v = y.iloc[val_idx]

            if len(X_tr) < 10 or len(X_v) < 3:
                logger.warning(
                    f"[{self.run_id}] Fold {fold_idx}: skipped (train={len(X_tr)}, val={len(X_v)})."
                )
                continue

            fold_model = EnsembleModel()
            fold_model.fit(X_tr, y_tr)
            fold_preds = fold_model.predict_proba(X_v)[:, 1]
            oof_probs[val_idx] = fold_preds
            oof_covered[val_idx] = True

            fold_brier = compute_brier_score(y_v.values, fold_preds)
            fold_logloss = compute_log_loss(y_v.values, fold_preds)
            fold_metrics.append({
                "fold": fold_idx,
                "n_train": len(X_tr),
                "n_val": len(X_v),
                "brier": round(fold_brier, 5),
                "logloss": round(fold_logloss, 5),
            })
            logger.info(
                f"[{self.run_id}] Fold {fold_idx}: "
                f"train={len(X_tr)}, val={len(X_v)}, "
                f"brier={fold_brier:.4f}, logloss={fold_logloss:.4f}"
            )

        if not oof_covered.any():
            logger.error(f"[{self.run_id}] OOF loop produced no predictions — aborting.")
            return {"status": "oof_failed", "n_samples": len(y)}

        # OOF metrics on the covered subset only
        oof_y = y.values[oof_covered]
        oof_p = oof_probs[oof_covered]
        oof_brier = round(compute_brier_score(oof_y, oof_p), 5)
        oof_logloss = round(compute_log_loss(oof_y, oof_p), 5)
        logger.info(
            f"[{self.run_id}] OOF Brier={oof_brier:.5f}, OOF LogLoss={oof_logloss:.5f}"
        )

        # ── Step 2: Fit calibrator on the full OOF predictions ────────────
        calibrator = IsotonicCalibrator()
        calibrator.fit(oof_p, oof_y)

        # ── Step 3: Retrain final model on ALL data ───────────────────────
        logger.info(f"[{self.run_id}] Training final model on all {len(X)} samples...")
        final_ensemble = EnsembleModel()
        final_ensemble.fit(X, y)

        # Wrap ensemble with the OOF-calibrated calibrator
        calibrated = CalibratedModel(final_ensemble, calibrator)
        # Generate a calibration report using the OOF probabilities as proxy
        cal_oof_p = calibrator.calibrate(oof_p)
        calibrated._calibration_report = evaluate_calibration(
            oof_y, oof_p, cal_oof_p,
            label="ensemble_oof_calibrated",
        )

        # ── Step 4: Champion/challenger check ─────────────────────────────
        current_champion = self.registry.load("match_win_ensemble")
        promote = True
        if current_champion is not None and hasattr(current_champion, "predict_proba"):
            # Evaluate current champion on the OOF held-out set
            try:
                champ_oof_p = current_champion.predict_proba(X.iloc[oof_covered])[:, 1]
                champ_brier = compute_brier_score(oof_y, champ_oof_p)
                promote = self._should_promote_challenger(champ_brier, oof_brier)
            except Exception as exc:
                logger.warning(
                    f"[{self.run_id}] Champion evaluation failed ({exc}) — promoting challenger by default."
                )

        if not promote and not force_retrain:
            logger.info(f"[{self.run_id}] Challenger not promoted. Keeping current champion.")
            return {
                "run_id": self.run_id,
                "status": "not_promoted",
                "oof_brier": oof_brier,
                "oof_logloss": oof_logloss,
                "n_oof_folds": self._N_OOF_SPLITS,
                "fold_metrics": fold_metrics,
                "timestamp": datetime.utcnow().isoformat(),
            }

        # ── Step 5: Save model artifact ───────────────────────────────────
        model_path = self.registry.save("match_win_ensemble", calibrated)
        fi = calibrated.feature_importance()

        # ── Step 6: Signal weight learning from OOF ───────────────────────
        features_df = self.pipeline.get_team_features()
        signal_weight_update = self._learn_signal_weights_from_oof(X, y, features_df)

        metadata = {
            "run_id": self.run_id,
            "model_name": "match_win_ensemble",
            "model_type": "OOF_CalibratedEnsemble",
            "market_type": "head_to_head",
            "n_samples": len(y),
            "n_oof_folds": self._N_OOF_SPLITS,
            "oof_brier": oof_brier,
            "oof_logloss": oof_logloss,
            "fold_metrics": fold_metrics,
            "signal_weight_update": signal_weight_update,
            "feature_names": feature_cols,
            "feature_importance": {k: round(v, 5) for k, v in list(fi.items())[:10]},
            "calibration_report": calibrated.calibration_report,
            "artifact_path": str(model_path),
            "timestamp": datetime.utcnow().isoformat(),
            "status": "success",
        }

        logger.info(f"[{self.run_id}] Match model training complete. OOF Brier={oof_brier:.5f}")
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
                "raw_brier": round(
                    compute_brier_score(y_test.values, np.clip(raw_probs_test, 1e-7, 1 - 1e-7)), 5
                ),
                "calibrated_brier": round(
                    compute_brier_score(y_test.values, np.clip(cal_probs_test, 1e-7, 1 - 1e-7)), 5
                ),
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
            "feature_importance": {
                k: round(v, 5)
                for k, v in list(regression_model.feature_importance().items())[:10]
            },
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

    # ------------------------------------------------------------------
    # OOF signal weight learning
    # ------------------------------------------------------------------

    def _learn_signal_weights_from_oof(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        features_df: pd.DataFrame,
    ) -> Dict:
        """
        Evaluate per-signal performance across OOF folds and update the
        SignalWeightStore.

        For each held-out fold:
          - For each completed fixture in the fold, extract its feature dict.
          - Run SignalEngine to get per-signal probability predictions.
          - Record (signal_name, predicted_prob, actual_outcome).

        After all folds, compute per-signal Brier score and push to the store.

        Any fixture that fails (e.g. missing features) is skipped gracefully.

        Args:
            X:           model-ready feature matrix (index-aligned with y)
            y:           binary labels (home_win)
            features_df: full team features from pipeline.get_team_features(),
                         used to look up raw signal inputs per fixture_id

        Returns:
            Dict mapping signal_name → Brier score (empty on failure)
        """
        logger.info(f"[{self.run_id}] Learning signal weights from OOF folds...")

        signal_engine = SignalEngine()

        # Build a fixture_id → feature row mapping for fast lookup
        has_fixture_id = "fixture_id" in features_df.columns
        if not has_fixture_id:
            logger.warning(
                "features_df has no 'fixture_id' column — signal weight learning skipped."
            )
            return {}

        # Keep only completed fixtures so we have ground-truth outcomes
        completed_mask = (
            (features_df.get("status") == "completed")
            if "status" in features_df.columns
            else pd.Series(True, index=features_df.index)
        )
        completed_features = features_df[completed_mask].copy()

        if completed_features.empty:
            logger.warning("No completed fixtures in features_df — signal weight learning skipped.")
            return {}

        # Index by fixture_id for O(1) lookup
        feat_by_fid: Dict[int, Dict] = {
            int(row["fixture_id"]): row.to_dict()
            for _, row in completed_features.iterrows()
        }

        # We need a fixture_id column aligned with X's integer position index.
        # features_df rows for completed fixtures should line up with the X / y
        # produced by get_model_ready_match_data() — grab fixture_ids in order.
        if "fixture_id" in completed_features.columns:
            fixture_ids_all = completed_features["fixture_id"].values
        else:
            logger.warning("Cannot map X rows to fixture_ids — signal weight learning skipped.")
            return {}

        # Guard: if lengths differ, fall back (shouldn't happen in normal flow)
        if len(fixture_ids_all) != len(X):
            logger.warning(
                "fixture_ids_all length (%d) != X length (%d) — "
                "signal weight learning skipped.",
                len(fixture_ids_all),
                len(X),
            )
            return {}

        # Per-signal lists of (predicted_prob, actual_outcome)
        signal_records: Dict[str, List[Tuple[float, float]]] = {}

        tscv = TimeSeriesSplit(n_splits=self._N_OOF_SPLITS)
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            for pos in val_idx:
                fid = int(fixture_ids_all[pos])
                actual_outcome = float(y.iloc[pos])
                feat_dict = feat_by_fid.get(fid)

                if feat_dict is None:
                    continue  # fixture has no features — skip

                try:
                    home_result, _ = signal_engine.compute_h2h_signals(feat_dict)
                    for sig in home_result.signals:
                        signal_records.setdefault(sig.name, []).append(
                            (sig.probability, actual_outcome)
                        )
                except Exception as exc:
                    logger.debug(
                        "Signal engine failed for fixture_id={} (fold {}): {}",
                        fid, fold_idx, exc,
                    )
                    # Skip this fixture gracefully — do not re-raise
                    continue

        if not signal_records:
            logger.warning(
                f"[{self.run_id}] No signal records collected — "
                "signal weight store not updated."
            )
            return {}

        # Compute per-signal Brier scores
        signal_briers: Dict[str, float] = {}
        for sig_name, pairs in signal_records.items():
            probs = np.array([p for p, _ in pairs])
            outcomes = np.array([o for _, o in pairs])
            brier = float(np.mean((probs - outcomes) ** 2))
            signal_briers[sig_name] = round(brier, 6)

        logger.info(
            f"[{self.run_id}] Per-signal Brier scores: "
            + ", ".join(f"{k}={v:.4f}" for k, v in sorted(signal_briers.items()))
        )

        # Push to the global signal weight store
        signal_weight_store = get_signal_weight_store()
        signal_weight_store.update_from_oof("head_to_head", signal_briers)

        return signal_briers

    # ------------------------------------------------------------------
    # Champion / challenger promotion
    # ------------------------------------------------------------------

    def _should_promote_challenger(
        self, current_brier: float, challenger_brier: float
    ) -> bool:
        """
        Return True if the challenger's OOF Brier score beats the current
        champion by at least settings.model_promotion_brier_improvement.

        Args:
            current_brier:    OOF Brier of the already-deployed champion model
            challenger_brier: OOF Brier of the newly trained challenger

        Returns:
            True  → promote challenger
            False → keep current champion
        """
        threshold = float(settings.model_promotion_brier_improvement)
        improvement = current_brier - challenger_brier
        promote = improvement >= threshold

        if promote:
            logger.info(
                f"[{self.run_id}] Promoting challenger: "
                f"current_brier={current_brier:.5f}, "
                f"challenger_brier={challenger_brier:.5f}, "
                f"improvement={improvement:.5f} >= threshold={threshold:.5f}."
            )
        else:
            logger.info(
                f"[{self.run_id}] Keeping current champion: "
                f"current_brier={current_brier:.5f}, "
                f"challenger_brier={challenger_brier:.5f}, "
                f"improvement={improvement:.5f} < threshold={threshold:.5f}."
            )
        return promote
