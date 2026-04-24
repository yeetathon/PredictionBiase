"""
Learned meta-blend: combines ML model output with signal engine consensus.

Instead of hardcoded weights (e.g. "60% ML, 40% signals"), the MetaBlender
fits a logistic regression on settled leg history to learn optimal component
weights per market type.

Fallback: if fewer than MIN_SAMPLES settled legs exist for a market, the
blender falls back to a signal-quality-weighted fixed formula so the pipeline
is never blocked.

Training inputs (per settled leg):
  - ml_probability               raw ML model output
  - signal_consensus_probability signal engine consensus
  - signal_agreement             0–1 agreement across signals
  - prediction_variance          variance across signals
  - data_completeness            0–1 data quality indicator
  - trust_score                  0–100 overall trust (normalised internally)

Output: P(win) learned from actual settlement outcomes.
"""
from __future__ import annotations

import os
import math
from typing import Dict, List, Optional
import numpy as np
from loguru import logger


_ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "models")
_MIN_SAMPLES = 30   # below this, fall back to fixed blend


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(np.clip(x, -20, 20))))


class _LogisticModel:
    """Minimal logistic regression (no sklearn dependency)."""

    def __init__(self, n_features: int, lr: float = 0.05, n_iter: int = 300):
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.lr = lr
        self.n_iter = n_iter
        self.fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n = len(y)
        for _ in range(self.n_iter):
            logits = X @ self.w + self.b
            p = np.array([_sigmoid(v) for v in logits])
            err = p - y
            self.w -= self.lr * (X.T @ err) / n
            self.b -= self.lr * err.mean()
        self.fitted = True

    def predict(self, x: np.ndarray) -> float:
        return _sigmoid(float(x @ self.w + self.b))

    def save(self, path: str) -> None:
        np.savez(path, w=self.w, b=np.array([self.b]))

    @classmethod
    def load(cls, path: str) -> "_LogisticModel":
        data = np.load(path + ".npz")
        model = cls(n_features=len(data["w"]))
        model.w = data["w"]
        model.b = float(data["b"][0])
        model.fitted = True
        return model


def _features(ml_prob: float, sig_prob: float, sig_agree: float,
               pred_var: float, data_complete: float, trust: float) -> np.ndarray:
    """Build the 6-dimensional feature vector (all in [0, 1])."""
    return np.array([
        float(np.clip(ml_prob, 0.0, 1.0)),
        float(np.clip(sig_prob, 0.0, 1.0)),
        float(np.clip(sig_agree, 0.0, 1.0)),
        float(np.clip(pred_var, 0.0, 1.0)),
        float(np.clip(data_complete, 0.0, 1.0)),
        float(np.clip(trust / 100.0, 0.0, 1.0)),
    ], dtype=float)


def _fixed_blend(ml_prob: float, sig_prob: float,
                 sig_agree: float, data_complete: float) -> float:
    """
    Fallback fixed blend when insufficient training data.
    ml_weight shifts toward ML (0.60) when signals disagree or data is sparse.
    """
    signal_quality = float(np.clip(sig_agree, 0.0, 1.0)) * float(np.clip(data_complete, 0.0, 1.0))
    ml_weight = 0.60 - signal_quality * 0.20   # 0.40–0.60
    return float(np.clip(ml_weight * ml_prob + (1.0 - ml_weight) * sig_prob, 0.02, 0.98))


class MetaBlender:
    """
    Per-market logistic regression blend. Thread-safe (read-only after init).

    Usage:
        blender = MetaBlender()
        prob = blender.blend(ml_prob=0.62, sig_prob=0.58, ...)

        # Called by the bootstrap cycle after settlement:
        blender.train_from_settlements(settled_legs, market_type="head_to_head")
    """

    def __init__(self) -> None:
        self._models: Dict[str, _LogisticModel] = {}
        self._load_all()

    # ── Inference ─────────────────────────────────────────────────────────

    def blend(
        self,
        ml_prob: float,
        sig_prob: float,
        sig_agree: float,
        pred_var: float,
        data_complete: float,
        trust: float,
        market_type: str = "head_to_head",
    ) -> float:
        model = self._models.get(market_type)
        if model is None or not model.fitted:
            return _fixed_blend(ml_prob, sig_prob, sig_agree, data_complete)

        x = _features(ml_prob, sig_prob, sig_agree, pred_var, data_complete, trust)
        raw = model.predict(x)
        return float(np.clip(raw, 0.02, 0.98))

    # ── Training ──────────────────────────────────────────────────────────

    def train_from_settlements(
        self, legs: List[Dict], market_type: str = "head_to_head"
    ) -> bool:
        """
        Train on a list of settled leg dicts. Each dict must have:
            ml_probability, signal_consensus_probability, signal_agreement,
            prediction_variance, data_completeness, trust_score, outcome (0/1)

        Returns True if the model was updated, False if insufficient data.
        """
        usable = [
            l for l in legs
            if (
                l.get("ml_probability") is not None
                and l.get("signal_consensus_probability") is not None
                and l.get("outcome") in (0, 1)
            )
        ]
        if len(usable) < _MIN_SAMPLES:
            logger.info(
                "MetaBlender: only {} usable samples for {} (need {}), skipping train",
                len(usable), market_type, _MIN_SAMPLES,
            )
            return False

        X = np.array([
            _features(
                ml_prob=l["ml_probability"],
                sig_prob=l["signal_consensus_probability"],
                sig_agree=l.get("signal_agreement") or 0.5,
                pred_var=l.get("prediction_variance") or 0.0,
                data_complete=l.get("data_completeness") or 0.8,
                trust=l.get("trust_score") or 50.0,
            )
            for l in usable
        ])
        y = np.array([float(l["outcome"]) for l in usable])

        model = _LogisticModel(n_features=6)
        model.fit(X, y)
        self._models[market_type] = model
        self._save(market_type)
        logger.info(
            "MetaBlender: trained {} on {} samples, saved to disk",
            market_type, len(usable),
        )
        return True

    # ── Persistence ───────────────────────────────────────────────────────

    def _artifact_path(self, market_type: str) -> str:
        safe = market_type.replace(" ", "_")
        return os.path.join(_ARTIFACT_DIR, f"meta_blender_{safe}")

    def _save(self, market_type: str) -> None:
        os.makedirs(_ARTIFACT_DIR, exist_ok=True)
        path = self._artifact_path(market_type)
        try:
            self._models[market_type].save(path)
        except Exception as exc:
            logger.warning("MetaBlender: could not save {} model: {}", market_type, exc)

    def _load_all(self) -> None:
        for mt in ("head_to_head", "player_disposals", "line", "total"):
            path = self._artifact_path(mt) + ".npz"
            if os.path.exists(path):
                try:
                    self._models[mt] = _LogisticModel.load(
                        self._artifact_path(mt)
                    )
                    logger.debug("MetaBlender: loaded {} model from {}", mt, path)
                except Exception as exc:
                    logger.warning(
                        "MetaBlender: could not load {} model ({}), using fixed blend",
                        mt, exc,
                    )
