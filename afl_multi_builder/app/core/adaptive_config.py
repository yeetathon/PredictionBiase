"""
Adaptive, market-specific threshold configuration.

Thresholds are persisted to ``settings.artifacts_dir / "adaptive_thresholds.json"``
and auto-updated after each evaluation cycle.  Falls back to hard-coded defaults
when the file is absent or a market entry is missing.

Profiles
--------
* ``"balanced"``     — defaults as-is (no adjustment)
* ``"conservative"`` — 50 % stricter edge/ev, +0.03 prob, +0.10 trust
* ``"research"``     — looser gates for data-collection runs

Usage
-----
    from app.core.adaptive_config import get_adaptive_config

    cfg = get_adaptive_config("balanced")
    min_edge = cfg.get("head_to_head", "min_edge", fallback=0.05)
    if cfg.is_market_enabled("player_disposals"):
        ...
    cfg.update_from_evaluation(eval_report)
"""
from __future__ import annotations

import json
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

from app.core.config import settings

# ---------------------------------------------------------------------------
# Hard-coded baseline defaults (used when JSON store is absent / market missing)
# ---------------------------------------------------------------------------

_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "head_to_head": {
        "min_edge": 0.05,
        "min_ev": 0.04,
        "min_prob": 0.54,
        "min_confidence": 40.0,
        "min_trust": 35.0,
        "min_signal_agreement": 0.30,
        "max_odds": 4.00,
        "enabled": True,
    },
    "player_disposals": {
        "min_edge": 0.06,
        "min_ev": 0.05,
        "min_prob": 0.55,
        "min_confidence": 45.0,
        "min_trust": 40.0,
        "min_signal_agreement": 0.35,
        "max_odds": 3.50,
        "enabled": True,
    },
    "line": {
        "min_edge": 0.05,
        "min_ev": 0.04,
        "min_prob": 0.54,
        "min_confidence": 40.0,
        "min_trust": 35.0,
        "min_signal_agreement": 0.30,
        "max_odds": 2.20,
        "enabled": True,
    },
    "total": {
        "min_edge": 0.05,
        "min_ev": 0.04,
        "min_prob": 0.54,
        "min_confidence": 40.0,
        "min_trust": 35.0,
        "min_signal_agreement": 0.30,
        "max_odds": 2.20,
        "enabled": True,
    },
}

# ---------------------------------------------------------------------------
# Profile adjustment multipliers / offsets.
# Applied on-the-fly in get(); the stored values are always in "balanced" space.
#
# conservative: 50% stricter edge/ev (factor=1.50), +0.03 prob, +0.10 trust
#               (trust is stored 0-100, so +0.10 → +10.0 in store units)
# balanced:     no adjustments
# research:     loosen all gates to collect more labelled examples
# ---------------------------------------------------------------------------

_PROFILES: Dict[str, Dict[str, Any]] = {
    "conservative": {
        # Multiplicative adjustments for ratio-based thresholds
        "min_edge_factor": 1.50,
        "min_ev_factor": 1.50,
        # Additive adjustments for probability / score thresholds
        "min_prob_delta": 0.03,
        "min_confidence_delta": 0.0,
        "min_trust_delta": 10.0,          # +0.10 × 100 because trust is stored 0-100
        "min_signal_agreement_delta": 0.0,
        "max_odds_factor": 1.0,
    },
    "balanced": {
        "min_edge_factor": 1.0,
        "min_ev_factor": 1.0,
        "min_prob_delta": 0.0,
        "min_confidence_delta": 0.0,
        "min_trust_delta": 0.0,
        "min_signal_agreement_delta": 0.0,
        "max_odds_factor": 1.0,
    },
    "research": {
        # Loosen gates so we collect more labelled examples
        "min_edge_factor": 0.60,
        "min_ev_factor": 0.60,
        "min_prob_delta": -0.04,
        "min_confidence_delta": -10.0,
        "min_trust_delta": -10.0,
        "min_signal_agreement_delta": -0.10,
        "max_odds_factor": 1.50,
    },
}

# ---------------------------------------------------------------------------
# Internal mappings used by _apply_profile
# ---------------------------------------------------------------------------

_STORE_PATH_SUFFIX = "adaptive_thresholds.json"

# Keys that scale multiplicatively when a profile adjustment is applied
_MULTIPLICATIVE_KEYS = {"min_edge", "min_ev", "max_odds"}

# Keys that shift additively; value maps key → profile-dict key for the delta
_ADDITIVE_KEY_MAP: Dict[str, str] = {
    "min_prob": "min_prob_delta",
    "min_confidence": "min_confidence_delta",
    "min_trust": "min_trust_delta",
    "min_signal_agreement": "min_signal_agreement_delta",
}


# ---------------------------------------------------------------------------
# AdaptiveConfig
# ---------------------------------------------------------------------------


class AdaptiveConfig:
    """
    Market-specific threshold configuration with profile support and
    self-updating logic driven by evaluation reports.

    Thread-safe: a single ``threading.Lock`` guards all JSON reads/writes.
    """

    def __init__(self, profile: str = "balanced") -> None:
        if profile not in _PROFILES:
            logger.warning(
                "AdaptiveConfig: unknown profile '{}', falling back to 'balanced'",
                profile,
            )
            profile = "balanced"

        self._profile = profile
        self._lock = threading.Lock()
        self._store_path: Path = settings.artifacts_dir / _STORE_PATH_SUFFIX
        self._data: Dict[str, Dict[str, Any]] = {}
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, market: str, key: str, fallback: Any = None) -> Any:
        """
        Return the threshold value for *market* / *key*, adjusted for the
        current profile.

        Falls back to hard-coded ``_DEFAULTS`` when the market or key is
        absent in the persisted store; if the key is absent in defaults too,
        returns *fallback*.
        """
        with self._lock:
            market_cfg = self._data.get(market) or deepcopy(_DEFAULTS.get(market, {}))
            raw = market_cfg.get(key)
            if raw is None:
                raw = _DEFAULTS.get(market, {}).get(key, fallback)
            if raw is None:
                return fallback

        return self._apply_profile(key, raw)

    def is_market_enabled(self, market: str) -> bool:
        """Return ``True`` if the market is enabled in the current config."""
        with self._lock:
            market_cfg = self._data.get(market) or _DEFAULTS.get(market, {})
            return bool(market_cfg.get("enabled", True))

    def update_from_evaluation(self, eval_report: dict) -> None:
        """
        Adjust stored thresholds based on an evaluation report and persist.

        Rules applied per market when ``n >= 10``:

        * ROI < -0.10 and n >= 20
            → tighten: ``min_edge *= 1.05``, ``min_ev *= 1.05``,
              ``min_trust += 2.0`` (store units, i.e. +0.02 in [0-1] space)
        * ROI > +0.05 and n >= 20
            → slight relaxation: ``min_edge *= 0.98``
        * brier > 0.25 and n >= 15
            → disable market (``enabled = False``)
        * brier < 0.22 and n >= 15
            → re-enable market (``enabled = True``)

        The store is saved only when at least one value changes.
        """
        by_market: Dict[str, Dict[str, Any]] = eval_report.get("by_market_type", {})
        if not by_market:
            logger.debug("AdaptiveConfig.update_from_evaluation: no by_market_type data")
            return

        changed = False
        with self._lock:
            for market, stats in by_market.items():
                n: int = int(stats.get("n", 0))
                if n < 10:
                    continue

                roi: float = float(stats.get("roi", 0.0))
                brier: float = float(stats.get("brier_score", 0.25))

                # Ensure the market has an entry in our live store
                if market not in self._data:
                    self._data[market] = deepcopy(
                        _DEFAULTS.get(market, deepcopy(_DEFAULTS["head_to_head"]))
                    )

                mkt = self._data[market]

                # ── ROI-based tightening / relaxation ─────────────────────
                if roi < -0.10 and n >= 20:
                    old_edge = mkt.get("min_edge", _DEFAULTS.get(market, {}).get("min_edge", 0.05))
                    old_ev = mkt.get("min_ev", _DEFAULTS.get(market, {}).get("min_ev", 0.04))
                    old_trust = mkt.get("min_trust", _DEFAULTS.get(market, {}).get("min_trust", 35.0))
                    mkt["min_edge"] = round(old_edge * 1.05, 5)
                    mkt["min_ev"] = round(old_ev * 1.05, 5)
                    mkt["min_trust"] = round(old_trust + 2.0, 2)
                    logger.info(
                        "AdaptiveConfig [{}]: tightened — ROI={:.2%} n={} "
                        "edge {:.4f}→{:.4f} ev {:.4f}→{:.4f} trust {:.1f}→{:.1f}",
                        market, roi, n,
                        old_edge, mkt["min_edge"],
                        old_ev, mkt["min_ev"],
                        old_trust, mkt["min_trust"],
                    )
                    changed = True

                elif roi > 0.05 and n >= 20:
                    old_edge = mkt.get("min_edge", _DEFAULTS.get(market, {}).get("min_edge", 0.05))
                    mkt["min_edge"] = round(old_edge * 0.98, 5)
                    logger.info(
                        "AdaptiveConfig [{}]: relaxed — ROI={:.2%} n={} "
                        "edge {:.4f}→{:.4f}",
                        market, roi, n, old_edge, mkt["min_edge"],
                    )
                    changed = True

                # ── Brier-based enable / disable ───────────────────────────
                if brier > 0.25 and n >= 15:
                    if mkt.get("enabled", True):
                        mkt["enabled"] = False
                        logger.warning(
                            "AdaptiveConfig [{}]: DISABLED — brier={:.4f} n={}",
                            market, brier, n,
                        )
                        changed = True

                elif brier < 0.22 and n >= 15:
                    if not mkt.get("enabled", True):
                        mkt["enabled"] = True
                        logger.info(
                            "AdaptiveConfig [{}]: re-enabled — brier={:.4f} n={}",
                            market, brier, n,
                        )
                        changed = True

            if changed:
                self._save_locked()

    # ------------------------------------------------------------------
    # Internal helpers (lock must be held by caller where noted)
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load thresholds from JSON; fall back to defaults on any error."""
        with self._lock:
            try:
                if self._store_path.exists():
                    raw = self._store_path.read_text(encoding="utf-8")
                    loaded: dict = json.loads(raw)
                    # Merge: keep defaults for any missing markets / keys
                    for market, defaults in _DEFAULTS.items():
                        if market not in loaded:
                            loaded[market] = deepcopy(defaults)
                        else:
                            for k, v in defaults.items():
                                loaded[market].setdefault(k, v)
                    self._data = loaded
                    logger.debug(
                        "AdaptiveConfig: loaded {} markets from {}",
                        len(self._data),
                        self._store_path,
                    )
                else:
                    self._data = deepcopy(_DEFAULTS)
                    logger.debug(
                        "AdaptiveConfig: store not found at {}, using defaults",
                        self._store_path,
                    )
            except Exception as exc:
                logger.warning(
                    "AdaptiveConfig: failed to load '{}': {} — using defaults",
                    self._store_path,
                    exc,
                )
                self._data = deepcopy(_DEFAULTS)

    def _save_locked(self) -> None:
        """Persist current state to JSON.  Caller **must** hold ``_lock``."""
        try:
            self._store_path.parent.mkdir(parents=True, exist_ok=True)
            # Atomic write via temp file → rename
            tmp = self._store_path.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps(self._data, indent=2, default=str),
                encoding="utf-8",
            )
            tmp.replace(self._store_path)
            logger.debug("AdaptiveConfig: saved to {}", self._store_path)
        except Exception as exc:
            logger.error(
                "AdaptiveConfig: failed to save '{}': {}", self._store_path, exc
            )

    def _apply_profile(self, key: str, value: Any) -> Any:
        """Apply the active profile adjustment to a single threshold value."""
        if not isinstance(value, (int, float)):
            return value

        prof = _PROFILES.get(self._profile, _PROFILES["balanced"])

        if key in _MULTIPLICATIVE_KEYS:
            factor = prof.get(f"{key}_factor", 1.0)
            return value * factor

        if key in _ADDITIVE_KEY_MAP:
            delta = prof.get(_ADDITIVE_KEY_MAP[key], 0.0)
            return value + delta

        return value


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_instance: Optional[AdaptiveConfig] = None
_instance_profile: Optional[str] = None
_singleton_lock = threading.Lock()


def get_adaptive_config(profile: str = "balanced") -> AdaptiveConfig:
    """
    Return the module-level ``AdaptiveConfig`` singleton.

    A new instance is created only when the requested *profile* differs from
    the one used by the existing singleton (e.g. when the pipeline switches
    from ``"balanced"`` to ``"research"``).
    """
    global _instance, _instance_profile
    with _singleton_lock:
        if _instance is None or _instance_profile != profile:
            _instance = AdaptiveConfig(profile=profile)
            _instance_profile = profile
    return _instance
