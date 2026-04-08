"""
Probability calibration using isotonic regression and Platt scaling.
Calibration is essential: a model that outputs 0.70 probability should
win roughly 70% of the time on those predictions.
"""
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
import joblib
from pathlib import Path
from loguru import logger

from app.core.metrics import compute_brier_score, compute_log_loss


class IsotonicCalibrator:
    """
    Post-hoc isotonic regression calibrator.
    Maps raw model probabilities to calibrated probabilities.
    Better than Platt scaling when probability distributions are non-monotonic.
    """

    def __init__(self):
        self._calibrator: Optional[IsotonicRegression] = None
        self._is_fitted = False

    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray) -> "IsotonicCalibrator":
        """Fit calibrator on validation data."""
        raw_probs = np.clip(raw_probs, 1e-7, 1 - 1e-7)
        self._calibrator = IsotonicRegression(out_of_bounds="clip")
        self._calibrator.fit(raw_probs, y_true.astype(float))
        self._is_fitted = True
        logger.info("Isotonic calibrator fitted.")
        return self

    def calibrate(self, raw_probs: np.ndarray) -> np.ndarray:
        """Transform raw probabilities to calibrated probabilities."""
        if not self._is_fitted or self._calibrator is None:
            logger.warning("Calibrator not fitted, returning raw probabilities.")
            return raw_probs
        raw_probs = np.clip(raw_probs, 1e-7, 1 - 1e-7)
        return self._calibrator.transform(raw_probs)

    def save(self, path: Path) -> None:
        """Persist calibrator to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._calibrator, path)
        logger.info(f"Calibrator saved to {path}")

    def load(self, path: Path) -> "IsotonicCalibrator":
        """Load calibrator from disk."""
        self._calibrator = joblib.load(path)
        self._is_fitted = True
        return self


class PlattCalibrator:
    """
    Platt scaling (logistic regression) calibrator.
    Works well when the miscalibration is monotonic (e.g., model is overconfident).
    """

    def __init__(self):
        self._lr: Optional[LogisticRegression] = None
        self._is_fitted = False

    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray) -> "PlattCalibrator":
        """Fit logistic regression on logit-transformed probabilities."""
        raw_probs = np.clip(raw_probs, 1e-7, 1 - 1e-7)
        logits = np.log(raw_probs / (1 - raw_probs)).reshape(-1, 1)
        self._lr = LogisticRegression(C=1.0)
        self._lr.fit(logits, y_true.astype(int))
        self._is_fitted = True
        return self

    def calibrate(self, raw_probs: np.ndarray) -> np.ndarray:
        """Transform raw probabilities via Platt scaling."""
        if not self._is_fitted or self._lr is None:
            return raw_probs
        raw_probs = np.clip(raw_probs, 1e-7, 1 - 1e-7)
        logits = np.log(raw_probs / (1 - raw_probs)).reshape(-1, 1)
        return self._lr.predict_proba(logits)[:, 1]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._lr, path)

    def load(self, path: Path) -> "PlattCalibrator":
        self._lr = joblib.load(path)
        self._is_fitted = True
        return self


def evaluate_calibration(
    y_true: np.ndarray,
    raw_probs: np.ndarray,
    calibrated_probs: Optional[np.ndarray] = None,
    n_bins: int = 10,
    label: str = "Model",
) -> dict:
    """
    Comprehensive calibration report.
    Returns dict with Brier score, log loss, bin analysis, and improvement metrics.
    """
    raw_probs = np.clip(raw_probs, 1e-7, 1 - 1e-7)

    raw_brier = compute_brier_score(y_true, raw_probs)
    raw_logloss = compute_log_loss(y_true, raw_probs)

    try:
        frac_pos, mean_pred = calibration_curve(y_true, raw_probs, n_bins=n_bins, strategy="uniform")
    except Exception:
        frac_pos, mean_pred = np.array([]), np.array([])

    report = {
        "label": label,
        "n_samples": int(len(y_true)),
        "base_rate": float(np.mean(y_true)),
        "raw_brier_score": round(raw_brier, 5),
        "raw_log_loss": round(raw_logloss, 5),
        "raw_calibration_bins": {
            "mean_predicted": mean_pred.tolist(),
            "fraction_of_positives": frac_pos.tolist(),
        },
    }

    if calibrated_probs is not None:
        calibrated_probs = np.clip(calibrated_probs, 1e-7, 1 - 1e-7)
        cal_brier = compute_brier_score(y_true, calibrated_probs)
        cal_logloss = compute_log_loss(y_true, calibrated_probs)
        try:
            cal_frac, cal_pred = calibration_curve(
                y_true, calibrated_probs, n_bins=n_bins, strategy="uniform"
            )
        except Exception:
            cal_frac, cal_pred = np.array([]), np.array([])

        report.update({
            "calibrated_brier_score": round(cal_brier, 5),
            "calibrated_log_loss": round(cal_logloss, 5),
            "brier_improvement": round(raw_brier - cal_brier, 5),
            "logloss_improvement": round(raw_logloss - cal_logloss, 5),
            "calibrated_calibration_bins": {
                "mean_predicted": cal_pred.tolist(),
                "fraction_of_positives": cal_frac.tolist(),
            },
        })

    return report
