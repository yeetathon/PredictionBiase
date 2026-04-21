"""
Learned signal weight store backed by JSON at
``settings.artifacts_dir / "signal_weights.json"``.

Weights are derived from out-of-fold (OOF) Brier-score evaluation:
lower Brier → better signal → higher weight.  New weights are blended
70 / 30 with the previous weights so that a single volatile training run
cannot destroy a well-established weight set.

A rolling history of the last 10 updates is kept inside the JSON so
operators can inspect how weights have evolved over time.

Usage
-----
    from app.pricing.signal_weights import get_signal_weight_store

    store = get_signal_weight_store()
    weights = store.get_weights("head_to_head")           # normalised dict
    store.update_from_oof("head_to_head", brier_scores)   # update + persist
    history = store.get_weight_history("head_to_head")    # list of dicts
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from app.core.config import settings

# ---------------------------------------------------------------------------
# Hard-coded default weight sets
# ---------------------------------------------------------------------------

_H2H_DEFAULTS: Dict[str, float] = {
    "elo": 0.25,
    "form": 0.20,
    "scoring": 0.25,
    "h2h": 0.20,
    "market": 0.10,
}

_PLAYER_DEFAULTS: Dict[str, float] = {
    "short_form": 0.20,
    "medium_form": 0.25,
    "trend": 0.15,
    "matchup": 0.20,
    "ml": 0.10,
    "market": 0.10,
}

# The JSON file suffix inside artifacts_dir
_STORE_PATH_SUFFIX = "signal_weights.json"

# Maximum entries kept in the per-market update history
_HISTORY_MAX = 10

# Blend ratio: proportion of new (OOF-derived) weights to retain
_NEW_WEIGHT_RATIO = 0.70
_OLD_WEIGHT_RATIO = 0.30

# Inverse-Brier epsilon: prevents division by zero when brier = 0
_BRIER_EPSILON = 0.02


# ---------------------------------------------------------------------------
# SignalWeightStore
# ---------------------------------------------------------------------------


class SignalWeightStore:
    """
    Stores per-market, per-signal weights learned from OOF evaluation.

    Persists to a JSON file so weights survive process restarts.  Falls back
    to hard-coded defaults when the file is absent or a market has no data.

    Thread-safe: a single ``threading.Lock`` guards all reads and writes.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._store_path: Path = settings.artifacts_dir / _STORE_PATH_SUFFIX

        # {market_type: {signal_name: normalised_weight}}
        self._weights: Dict[str, Dict[str, float]] = {}

        # {market_type: [{timestamp, weights, brier_scores}, ...]}  (last _HISTORY_MAX each)
        self._history: Dict[str, List[Dict[str, Any]]] = {}

        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_weights(self, market_type: str = "head_to_head") -> Dict[str, float]:
        """
        Return a copy of the normalised weights for *market_type*.

        Falls back to market-specific hard-coded defaults when no learned
        weights exist.  The returned dict always sums to 1.0.
        """
        with self._lock:
            raw = self._weights.get(market_type)
            if raw:
                return dict(raw)

        # No learned weights — use defaults
        return dict(self._default_weights(market_type))

    def update_from_oof(
        self,
        market_type: str,
        signal_brier_scores: Dict[str, float],
    ) -> None:
        """
        Update weights for *market_type* using OOF per-signal Brier scores.

        Algorithm
        ---------
        1. Compute inverse-Brier raw weights: ``w_i = 1 / (brier_i + epsilon)``
        2. Normalise so they sum to 1.
        3. Blend: ``final = 0.70 * new + 0.30 * old`` (renormalised).
        4. Append a timestamped record to the rolling history (capped at 10).
        5. Save to JSON.

        Parameters
        ----------
        market_type:
            e.g. ``"head_to_head"`` or ``"player_disposals"``
        signal_brier_scores:
            Mapping of signal name → mean OOF Brier score.
        """
        if not signal_brier_scores:
            logger.warning(
                "SignalWeightStore.update_from_oof: empty signal_brier_scores "
                "for market '{}' — skipping",
                market_type,
            )
            return

        with self._lock:
            # Step 1+2: inverse-Brier weights, normalised
            new_weights = self._inverse_brier_normalise(signal_brier_scores)

            # Step 3: blend with existing weights
            old_weights = self._weights.get(market_type) or self._default_weights(
                market_type
            )
            blended = self._blend(new_weights, old_weights)

            self._weights[market_type] = blended

            # Step 4: rolling history
            entry: Dict[str, Any] = {
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "weights": {k: round(v, 6) for k, v in blended.items()},
                "brier_scores": {k: round(float(v), 6) for k, v in signal_brier_scores.items()},
            }
            history = self._history.setdefault(market_type, [])
            history.append(entry)
            if len(history) > _HISTORY_MAX:
                self._history[market_type] = history[-_HISTORY_MAX:]

            logger.info(
                "SignalWeightStore [{}]: updated weights {}",
                market_type,
                {k: round(v, 4) for k, v in blended.items()},
            )

            # Step 5: persist
            self._save_locked()

    def get_weight_history(self, market_type: str) -> List[Dict[str, Any]]:
        """
        Return the rolling update history for *market_type* (up to last 10
        entries, ordered oldest → newest).

        Each entry contains:
          - ``timestamp``: ISO-8601 string
          - ``weights``: the blended weights that were stored after this update
          - ``brier_scores``: the raw OOF Brier scores provided to that update
        """
        with self._lock:
            return list(self._history.get(market_type, []))

    # ------------------------------------------------------------------
    # Internal helpers (lock must be held by caller where noted)
    # ------------------------------------------------------------------

    @staticmethod
    def _inverse_brier_normalise(
        brier_scores: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Convert Brier scores to normalised weights via inverse-Brier.

        ``raw_i = 1 / (brier_i + epsilon)``  then normalise to sum = 1.
        """
        raw: Dict[str, float] = {
            sig: 1.0 / (float(b) + _BRIER_EPSILON)
            for sig, b in brier_scores.items()
        }
        total = sum(raw.values())
        if total <= 0.0:
            n = max(len(raw), 1)
            return {sig: 1.0 / n for sig in raw}
        return {sig: w / total for sig, w in raw.items()}

    @staticmethod
    def _blend(
        new_weights: Dict[str, float],
        old_weights: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Blend new and old weight dicts: 70 % new + 30 % old, then renormalise.

        Signals present only in *new_weights* get the full new weight (no old
        counterpart → 0 old contribution).  Signals present only in
        *old_weights* are carried forward at 30 % of their old value so the
        store doesn't silently drop signals that weren't observed in the latest
        OOF run.
        """
        all_signals = set(new_weights) | set(old_weights)
        blended: Dict[str, float] = {}
        for sig in all_signals:
            n = new_weights.get(sig, 0.0)
            o = old_weights.get(sig, 0.0)
            blended[sig] = _NEW_WEIGHT_RATIO * n + _OLD_WEIGHT_RATIO * o

        total = sum(blended.values())
        if total <= 0.0:
            k = max(len(blended), 1)
            return {sig: 1.0 / k for sig in blended}
        return {sig: w / total for sig, w in blended.items()}

    @staticmethod
    def _default_weights(market_type: str) -> Dict[str, float]:
        """Return a copy of the hard-coded default weights for *market_type*."""
        if market_type == "player_disposals":
            return dict(_PLAYER_DEFAULTS)
        # head_to_head and any other market type
        return dict(_H2H_DEFAULTS)

    def _save_locked(self) -> None:
        """Persist current state to JSON.  Caller **must** hold ``_lock``."""
        try:
            self._store_path.parent.mkdir(parents=True, exist_ok=True)
            payload: Dict[str, Any] = {
                "weights": self._weights,
                "history": self._history,
            }
            tmp = self._store_path.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps(payload, indent=2, default=str),
                encoding="utf-8",
            )
            tmp.replace(self._store_path)
            logger.debug("SignalWeightStore: saved to {}", self._store_path)
        except Exception as exc:
            logger.error(
                "SignalWeightStore: failed to save '{}': {}", self._store_path, exc
            )

    def _load(self) -> None:
        """Load weights and history from JSON; fall back to defaults on error."""
        with self._lock:
            if not self._store_path.exists():
                logger.debug(
                    "SignalWeightStore: store not found at {} — using defaults",
                    self._store_path,
                )
                return

            try:
                raw = self._store_path.read_text(encoding="utf-8")
                payload: dict = json.loads(raw)

                self._weights = payload.get("weights", {})
                self._history = payload.get("history", {})

                logger.info(
                    "SignalWeightStore: loaded {} market(s) from {}",
                    len(self._weights),
                    self._store_path,
                )
            except Exception as exc:
                logger.warning(
                    "SignalWeightStore: failed to load '{}': {} — using defaults",
                    self._store_path,
                    exc,
                )
                self._weights = {}
                self._history = {}


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_instance: Optional[SignalWeightStore] = None
_singleton_lock = threading.Lock()


def get_signal_weight_store() -> SignalWeightStore:
    """
    Return the module-level ``SignalWeightStore`` singleton.

    The store is created on first call and shared for the lifetime of the
    process.  Thread-safe.
    """
    global _instance
    with _singleton_lock:
        if _instance is None:
            _instance = SignalWeightStore()
    return _instance
