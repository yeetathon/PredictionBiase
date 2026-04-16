"""
Prediction models for AFL match outcomes and player props.

Stack:
  - EloModel: fast baseline, interpretable
  - LogisticRegressionModel: linear probabilistic model over engineered features
  - GradientBoostingModel: XGBoost, captures non-linear interactions
  - EnsembleModel: weighted average of above
  - PlayerDisposalsModel: XGBoost regression converted to over/under probability
  - CalibratedWrapper: wraps any model with post-hoc isotonic calibration
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import joblib
from loguru import logger

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier, XGBRegressor

from app.pricing.calibration import IsotonicCalibrator, evaluate_calibration
from app.core.config import settings


class EloModel:
    """
    Pure Elo-based win probability model.
    No fitting required — uses pre-computed Elo ratings from feature pipeline.
    """
    name = "elo"

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        X must contain 'elo_win_prob_home' column.
        Returns array of shape (n, 2): [[p_away, p_home], ...]
        """
        if "elo_win_prob_home" in X.columns:
            p_home = X["elo_win_prob_home"].fillna(0.5).values
        else:
            p_home = np.full(len(X), 0.5)
        p_away = 1.0 - p_home
        return np.column_stack([p_away, p_home])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)

    def fit(self, X, y):
        """Elo model requires no fitting."""
        return self


class LogisticRegressionModel:
    """Logistic regression with standardisation for win probability."""
    name = "logistic"

    def __init__(self, C: float = 1.0):
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=C, max_iter=1000, solver="lbfgs")),
        ])
        self._feature_names: List[str] = []
        self._is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogisticRegressionModel":
        self._feature_names = list(X.columns)
        X_arr = X.fillna(0).values
        self.pipeline.fit(X_arr, y.values)
        self._is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            p = np.full(len(X), 0.5)
            return np.column_stack([1 - p, p])
        X_arr = X[self._feature_names].fillna(0).values
        return self.pipeline.predict_proba(X_arr)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def get_coefficients(self) -> Dict[str, float]:
        if not self._is_fitted:
            return {}
        lr = self.pipeline.named_steps["lr"]
        return dict(zip(self._feature_names, lr.coef_[0].tolist()))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "LogisticRegressionModel":
        return joblib.load(path)


class GradientBoostingModel:
    """XGBoost classifier for win probability / over-under."""
    name = "xgboost"

    def __init__(self, **params):
        default_params = {
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "random_state": 42,
            "verbosity": 0,
        }
        default_params.update(params)
        self.model = XGBClassifier(**default_params)
        self._feature_names: List[str] = []
        self._is_fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[List] = None,
    ) -> "GradientBoostingModel":
        self._feature_names = list(X.columns)
        X_arr = X.fillna(0).values
        fit_params: Dict[str, Any] = {"verbose": False}
        if eval_set is not None:
            X_val, y_val = eval_set[0]
            fit_params["eval_set"] = [(X_val.fillna(0).values, y_val.values)]
        self.model.fit(X_arr, y.values, **fit_params)
        self._is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            p = np.full(len(X), 0.5)
            return np.column_stack([1 - p, p])
        X_arr = X[self._feature_names].fillna(0).values
        return self.model.predict_proba(X_arr)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def feature_importance(self) -> Dict[str, float]:
        if not self._is_fitted:
            return {}
        scores = self.model.feature_importances_
        return dict(sorted(
            zip(self._feature_names, scores.tolist()),
            key=lambda x: x[1], reverse=True
        ))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "GradientBoostingModel":
        return joblib.load(path)


class EnsembleModel:
    """
    Weighted ensemble of Elo + Logistic + XGBoost.
    Weights tuned on validation set; defaults are equal-weight average.
    """
    name = "ensemble"

    def __init__(
        self,
        elo_weight: float = 0.25,
        lr_weight: float = 0.35,
        xgb_weight: float = 0.40,
    ):
        self.elo = EloModel()
        self.lr = LogisticRegressionModel()
        self.xgb = GradientBoostingModel()
        self.weights = [elo_weight, lr_weight, xgb_weight]
        self._is_fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: Optional[List] = None,
    ) -> "EnsembleModel":
        logger.info("Training LogisticRegression component...")
        self.lr.fit(X, y)
        logger.info("Training XGBoost component...")
        self.xgb.fit(X, y, eval_set=eval_set)
        self._is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p_elo = self.elo.predict_proba(X)[:, 1]
        p_lr = self.lr.predict_proba(X)[:, 1]
        p_xgb = self.xgb.predict_proba(X)[:, 1]
        w = self.weights
        total = sum(w)
        p_ensemble = (w[0] * p_elo + w[1] * p_lr + w[2] * p_xgb) / total
        return np.column_stack([1 - p_ensemble, p_ensemble])

    def predict_proba_breakdown(self, X: pd.DataFrame) -> Dict[str, float]:
        """
        Return individual component probabilities for a single row.
        Used by the signal engine to detect intra-model disagreement.
        Returns {"elo": 0.58, "logistic": 0.62, "xgboost": 0.61} for P(home win).
        X should be a single-row DataFrame.
        """
        p_elo = float(self.elo.predict_proba(X)[:, 1][0])
        p_lr = float(self.lr.predict_proba(X)[:, 1][0]) if self.lr._is_fitted else p_elo
        p_xgb = float(self.xgb.predict_proba(X)[:, 1][0]) if self.xgb._is_fitted else p_elo
        return {
            "elo": round(p_elo, 5),
            "logistic": round(p_lr, 5),
            "xgboost": round(p_xgb, 5),
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def feature_importance(self) -> Dict[str, float]:
        return self.xgb.feature_importance()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "EnsembleModel":
        return joblib.load(path)


class PlayerDisposalsModel:
    """
    XGBoost regression model for player disposals prediction.
    Converts point estimate to over/under probability using
    a fitted normal distribution around the prediction.
    """
    name = "player_disposals"

    def __init__(self, **params):
        default_params = {
            "n_estimators": 100,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "random_state": 42,
            "verbosity": 0,
        }
        default_params.update(params)
        self.model = XGBRegressor(**default_params)
        self._feature_names: List[str] = []
        self._is_fitted = False
        self._residual_std = 5.0  # fitted on training data

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PlayerDisposalsModel":
        self._feature_names = list(X.columns)
        X_arr = X.fillna(0).values
        self.model.fit(X_arr, y.values)
        preds = self.model.predict(X_arr)
        residuals = y.values - preds
        self._residual_std = max(1.0, float(np.std(residuals)))
        self._is_fitted = True
        logger.info(f"PlayerDisposalsModel fitted. Residual std: {self._residual_std:.2f}")
        return self

    def predict_mean(self, X: pd.DataFrame) -> np.ndarray:
        """Predict expected disposals."""
        if not self._is_fitted:
            return np.full(len(X), 20.0)
        X_arr = X[self._feature_names].fillna(0).values
        return self.model.predict(X_arr)

    def predict_over_prob(self, X: pd.DataFrame, line: float) -> np.ndarray:
        """
        Probability of going OVER the line.
        Uses normal distribution approximation around predicted mean.
        """
        from scipy.stats import norm
        means = self.predict_mean(X)
        # P(X > line) = 1 - CDF(line | mean, std)
        return 1.0 - norm.cdf(line, loc=means, scale=self._residual_std)

    def predict_proba_over(self, X: pd.DataFrame, line: float) -> np.ndarray:
        """Returns [[p_under, p_over], ...] array."""
        p_over = self.predict_over_prob(X, line)
        return np.column_stack([1 - p_over, p_over])

    def feature_importance(self) -> Dict[str, float]:
        if not self._is_fitted:
            return {}
        scores = self.model.feature_importances_
        return dict(sorted(
            zip(self._feature_names, scores.tolist()),
            key=lambda x: x[1], reverse=True
        ))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "PlayerDisposalsModel":
        return joblib.load(path)


class CalibratedModel:
    """
    Wraps a base model with post-hoc isotonic regression calibration.
    The calibrator is fitted separately on held-out validation data.
    """

    def __init__(self, base_model, calibrator: Optional[IsotonicCalibrator] = None):
        self.base_model = base_model
        self.calibrator = calibrator or IsotonicCalibrator()
        self._calibration_report: Optional[dict] = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: pd.DataFrame, y_val: pd.Series) -> "CalibratedModel":
        """Fit base model on train, calibrator on val."""
        self.base_model.fit(X_train, y_train)
        raw_val_probs = self.base_model.predict_proba(X_val)[:, 1]
        self.calibrator.fit(raw_val_probs, y_val.values)

        # Generate calibration report
        cal_probs = self.calibrator.calibrate(raw_val_probs)
        self._calibration_report = evaluate_calibration(
            y_val.values, raw_val_probs, cal_probs,
            label=f"{getattr(self.base_model, 'name', 'model')}_calibrated"
        )
        logger.info(
            f"Calibration report: raw_brier={self._calibration_report['raw_brier_score']:.4f}, "
            f"cal_brier={self._calibration_report.get('calibrated_brier_score', 'N/A')}"
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return calibrated probabilities."""
        raw = self.base_model.predict_proba(X)[:, 1]
        cal = self.calibrator.calibrate(raw)
        return np.column_stack([1 - cal, cal])

    def predict_proba_raw(self, X: pd.DataFrame) -> np.ndarray:
        """Return raw (uncalibrated) probabilities."""
        return self.base_model.predict_proba(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @property
    def calibration_report(self) -> Optional[dict]:
        return self._calibration_report

    def predict_proba_breakdown(self, X: pd.DataFrame) -> Dict[str, float]:
        """Delegate to base model breakdown if available."""
        if hasattr(self.base_model, "predict_proba_breakdown"):
            return self.base_model.predict_proba_breakdown(X)
        return {}

    def feature_importance(self) -> Dict[str, float]:
        if hasattr(self.base_model, "feature_importance"):
            return self.base_model.feature_importance()
        return {}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "CalibratedModel":
        return joblib.load(path)


class ModelRegistry:
    """Central registry for loading/saving trained models."""

    def __init__(self, artifacts_dir: Optional[Path] = None):
        self.artifacts_dir = artifacts_dir or settings.model_artifacts_dir
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._models: Dict[str, Any] = {}

    def save(self, name: str, model: Any) -> Path:
        path = self.artifacts_dir / f"{name}.joblib"
        joblib.dump(model, path)
        logger.info(f"Model '{name}' saved to {path}")
        return path

    def load(self, name: str) -> Optional[Any]:
        path = self.artifacts_dir / f"{name}.joblib"
        if not path.exists():
            logger.warning(f"Model '{name}' not found at {path}")
            return None
        model = joblib.load(path)
        self._models[name] = model
        return model

    def exists(self, name: str) -> bool:
        return (self.artifacts_dir / f"{name}.joblib").exists()

    def list_models(self) -> List[str]:
        return [p.stem for p in self.artifacts_dir.glob("*.joblib")]
