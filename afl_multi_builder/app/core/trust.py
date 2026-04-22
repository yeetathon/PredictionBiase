"""
Trust scoring: separates *how likely* (probability) from *how reliable
our estimate is* (trust).

A high-trust score means we have abundant, fresh, coherent evidence.
A low-trust score means the estimate is speculative, even if the model
outputs a confident probability.

Public API
----------
    from app.core.trust import compute_trust_score, trust_label, compute_market_calibration_quality

    score = compute_trust_score(
        n_games=18,
        data_completeness=0.85,
        signal_agreement=0.80,
        n_active_signals=4,
        prediction_variance=0.012,
        prediction_low=0.55,
        prediction_high=0.75,
    )
    label = trust_label(score)   # "High"
"""
from __future__ import annotations

import math
from typing import Optional

from loguru import logger


# ---------------------------------------------------------------------------
# Component helpers
# ---------------------------------------------------------------------------


def _sigmoid(x: float, midpoint: float, scale: float) -> float:
    """Logistic function: 1 / (1 + exp(-(x - midpoint) / scale))."""
    return 1.0 / (1.0 + math.exp(-(x - midpoint) / scale))


def _clip(value: float, lo: float, hi: float) -> float:
    """Clamp *value* to [*lo*, *hi*]."""
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Primary trust computation
# ---------------------------------------------------------------------------


def compute_trust_score(
    n_games: int,
    data_completeness: float,
    signal_agreement: float,
    n_active_signals: int,
    prediction_variance: float,
    prediction_low: float,
    prediction_high: float,
    market_calibration_quality: float = 0.5,
    data_freshness_days: float = 7.0,
    lineup_certainty: float = 1.0,
) -> float:
    """
    Compute a composite trust score in the range [0, 100].

    The score is the geometric mean of seven independent factors (each 0–1),
    multiplied by 100.  Using a geometric mean ensures that a single very-low
    factor pulls the whole score down — trust is limited by its weakest
    dimension.

    Parameters
    ----------
    n_games:
        Number of relevant historical games available for the prediction.
        Half-trust is reached at 12 games (sigmoid midpoint).
    data_completeness:
        Fraction of expected features / data present (0–1).
    signal_agreement:
        Degree of agreement across independent prediction signals (0–1).
    n_active_signals:
        Number of signals with meaningful reliability (weight > 0.1).
    prediction_variance:
        Variance of predicted probabilities across signals.
    prediction_low:
        Lower bound of the 80 % prediction interval.
    prediction_high:
        Upper bound of the 80 % prediction interval.
    market_calibration_quality:
        How well-calibrated the market type's model is (0–1).
        Use ``compute_market_calibration_quality`` to derive this.
        Defaults to 0.5 (neutral / unknown).
    data_freshness_days:
        Age of the most recent underlying data in days.  Older → lower trust.
        Exponential decay with 14-day half-life.
    lineup_certainty:
        Fraction of expected players confirmed in the starting lineup (0–1).
        Use 1.0 when not applicable (e.g. team-level H2H markets).

    Returns
    -------
    float
        Trust score in [0, 100], rounded to two decimal places.
    """
    # ── Factor 1: Sample sufficiency ──────────────────────────────────────
    # sigmoid(x, midpoint=12, scale=4): 0.5 at 12 games, ~0.88 at 20, ~0.27 at 6
    sample_factor: float = _sigmoid(float(n_games), midpoint=12.0, scale=4.0)

    # ── Factor 2: Data completeness ────────────────────────────────────────
    completeness_factor: float = _clip(float(data_completeness), 0.0, 1.0)

    # ── Factor 3: Signal coherence ─────────────────────────────────────────
    # 70 % from raw agreement + 30 % from signal richness (saturates at 4 signals)
    signal_richness: float = _clip((n_active_signals - 1) / 3.0, 0.0, 1.0)
    coherence_factor: float = (
        0.7 * _clip(float(signal_agreement), 0.0, 1.0) + 0.3 * signal_richness
    )

    # ── Factor 4: Prediction interval tightness ────────────────────────────
    # Width 0.10 → tightness ≈ 0.83; width 0.50 → tightness ≈ 0.17
    # Width saturates at 0.60 (tightness → 0)
    interval_width: float = float(prediction_high) - float(prediction_low)
    tightness_factor: float = _clip(1.0 - interval_width / 0.60, 0.0, 1.0)

    # ── Factor 5: Market calibration quality ──────────────────────────────
    # Clipped to [0.1, 1] so even an uncalibrated market carries some weight
    calibration_factor: float = _clip(float(market_calibration_quality), 0.1, 1.0)

    # ── Factor 6: Data freshness ───────────────────────────────────────────
    # Exponential decay: half-life ≈ 9.7 days (exp(-7/14) ≈ 0.61 at 7 days)
    # Floor at 0.1 so stale data still contributes a little
    freshness_factor: float = _clip(
        math.exp(-float(data_freshness_days) / 14.0), 0.1, 1.0
    )

    # ── Factor 7: Lineup certainty ─────────────────────────────────────────
    lineup_factor: float = _clip(float(lineup_certainty), 0.0, 1.0)

    # ── Geometric mean → [0, 100] ──────────────────────────────────────────
    # A factor of exactly 0 collapses trust to 0 (legitimate: no data at all).
    # Guard against log(0) by flooring each factor at 1e-12 inside the log.
    factors = [
        sample_factor,
        completeness_factor,
        coherence_factor,
        tightness_factor,
        calibration_factor,
        freshness_factor,
        lineup_factor,
    ]

    if any(f <= 0.0 for f in factors):
        score = 0.0
    else:
        log_mean = sum(math.log(max(f, 1e-12)) for f in factors) / len(factors)
        geo_mean = math.exp(log_mean)
        score = _clip(geo_mean * 100.0, 0.0, 100.0)

    logger.debug(
        "compute_trust_score: sample={:.3f} complete={:.3f} coherence={:.3f} "
        "tight={:.3f} calib={:.3f} fresh={:.3f} lineup={:.3f} → {:.1f}",
        sample_factor,
        completeness_factor,
        coherence_factor,
        tightness_factor,
        calibration_factor,
        freshness_factor,
        lineup_factor,
        score,
    )

    return round(score, 2)


# ---------------------------------------------------------------------------
# Trust label
# ---------------------------------------------------------------------------


def trust_label(score: float) -> str:
    """
    Convert a numeric trust score to a human-readable tier.

    Thresholds
    ----------
    >= 65  → ``"High"``
    >= 45  → ``"Medium"``
    >= 30  → ``"Low"``
    <  30  → ``"Very Low"``
    """
    if score >= 65.0:
        return "High"
    if score >= 45.0:
        return "Medium"
    if score >= 30.0:
        return "Low"
    return "Very Low"


# ---------------------------------------------------------------------------
# Market calibration quality helper
# ---------------------------------------------------------------------------


def compute_player_trust_score(
    n_games: int,
    data_completeness: float,
    signal_agreement: float,
    n_active_signals: int,
    prediction_variance: float,
    prediction_low: float,
    prediction_high: float,
    role_stability: float = 1.0,
    position_baseline_z: float = 0.0,
    market_calibration_quality: float = 0.5,
    data_freshness_days: float = 7.0,
    lineup_certainty: float = 1.0,
) -> float:
    """
    Player-prop-specific trust score in [0, 100].

    Uses the same geometric-mean structure as ``compute_trust_score`` but with
    parameters tuned for player disposals:

    * Sigmoid midpoint at **8 games** (not 12) — players establish baselines faster.
    * ``role_stability`` is a dedicated factor: unstable role = low trust even with
      adequate sample size.
    * ``position_baseline_z`` penalises predictions that are extreme outliers vs the
      player's position population — high |z| → model may be mis-specified.

    Parameters
    ----------
    n_games:
        Completed games with disposals data for this player.
    data_completeness:
        Fraction of expected features present (0–1).
    signal_agreement:
        Agreement across independent prediction signals (0–1).
    n_active_signals:
        Signals with reliability > 0.1.
    prediction_variance:
        Variance of predicted probabilities across signals.
    prediction_low / prediction_high:
        80% prediction interval bounds.
    role_stability:
        Fraction of recent 5 games the player played the same position (0–1).
        0.5 = alternating roles; 1.0 = rock-solid role.
    position_baseline_z:
        |z-score| of player's expected mean vs position-population mean.
        Large values (>2) suggest the model may be using an outlier baseline.
    market_calibration_quality:
        Derived from recent Brier score for player_disposals market (0–1).
    data_freshness_days:
        Age of most recent underlying data in days.
    lineup_certainty:
        Confirmation that player is named / expected to play (0–1).
        0.5 = late scratching risk; 1.0 = confirmed starter.
    """
    # ── Factor 1: Sample sufficiency (midpoint 8 games for players) ──────────
    sample_factor: float = _sigmoid(float(n_games), midpoint=8.0, scale=3.0)

    # ── Factor 2: Data completeness ───────────────────────────────────────────
    completeness_factor: float = _clip(float(data_completeness), 0.0, 1.0)

    # ── Factor 3: Signal coherence ────────────────────────────────────────────
    signal_richness: float = _clip((n_active_signals - 1) / 3.0, 0.0, 1.0)
    coherence_factor: float = (
        0.7 * _clip(float(signal_agreement), 0.0, 1.0) + 0.3 * signal_richness
    )

    # ── Factor 4: Prediction interval tightness ───────────────────────────────
    interval_width: float = float(prediction_high) - float(prediction_low)
    tightness_factor: float = _clip(1.0 - interval_width / 0.60, 0.0, 1.0)

    # ── Factor 5: Role stability (player-specific, weighted heavily) ──────────
    # Unstable role → output is speculative. Below 0.6 stability, trust drops sharply.
    role_factor: float = _clip(float(role_stability), 0.0, 1.0) ** 0.7

    # ── Factor 6: Position baseline plausibility ──────────────────────────────
    # |z| = 0 → factor = 1.0; |z| = 2 → factor ≈ 0.6; |z| > 3 → factor ≈ 0.4
    z = _clip(abs(float(position_baseline_z)), 0.0, 4.0)
    baseline_factor: float = _clip(1.0 / (1.0 + 0.3 * z), 0.3, 1.0)

    # ── Factor 7: Lineup certainty ────────────────────────────────────────────
    lineup_factor: float = _clip(float(lineup_certainty), 0.0, 1.0)

    # ── Factor 8: Market calibration quality ──────────────────────────────────
    calibration_factor: float = _clip(float(market_calibration_quality), 0.1, 1.0)

    # ── Factor 9: Data freshness ──────────────────────────────────────────────
    freshness_factor: float = _clip(
        math.exp(-float(data_freshness_days) / 14.0), 0.1, 1.0
    )

    factors = [
        sample_factor,
        completeness_factor,
        coherence_factor,
        tightness_factor,
        role_factor,
        baseline_factor,
        lineup_factor,
        calibration_factor,
        freshness_factor,
    ]

    if any(f <= 0.0 for f in factors):
        score = 0.0
    else:
        log_mean = sum(math.log(max(f, 1e-12)) for f in factors) / len(factors)
        geo_mean = math.exp(log_mean)
        score = _clip(geo_mean * 100.0, 0.0, 100.0)

    logger.debug(
        "compute_player_trust_score: sample={:.3f} complete={:.3f} coherence={:.3f} "
        "tight={:.3f} role={:.3f} baseline={:.3f} lineup={:.3f} calib={:.3f} "
        "fresh={:.3f} → {:.1f}",
        sample_factor, completeness_factor, coherence_factor, tightness_factor,
        role_factor, baseline_factor, lineup_factor, calibration_factor,
        freshness_factor, score,
    )

    return round(score, 2)


def compute_market_calibration_quality(
    market_type: str,
    eval_report: Optional[dict] = None,
) -> float:
    """
    Derive a calibration-quality score (0–1) for a specific market type
    from a recent evaluation report.

    The mapping is::

        quality = clip(1.0 - brier / 0.25, 0, 1)

    * A perfect Brier score of 0.0  → quality 1.0
    * The no-skill baseline of 0.25 → quality 0.0
    * Brier > 0.25 (worse than baseline) → quality 0.0

    Returns ``0.5`` (neutral / unknown quality) when:

    * *eval_report* is ``None``
    * *market_type* is not present in ``eval_report["by_market_type"]``
    * the Brier score entry is missing or non-numeric

    Parameters
    ----------
    market_type:
        One of ``"head_to_head"``, ``"player_disposals"``, ``"line"``,
        ``"total"``, etc.
    eval_report:
        Dict as returned by ``EvaluationService.evaluate()``.

    Returns
    -------
    float
        Calibration quality in [0, 1], rounded to four decimal places.
    """
    if eval_report is None:
        return 0.5

    by_market: dict = eval_report.get("by_market_type", {})
    market_stats: dict = by_market.get(market_type, {})

    if not market_stats:
        logger.debug(
            "compute_market_calibration_quality: no data for market '{}', returning 0.5",
            market_type,
        )
        return 0.5

    raw_brier = market_stats.get("brier_score")
    if raw_brier is None:
        return 0.5

    try:
        brier = float(raw_brier)
    except (TypeError, ValueError):
        logger.warning(
            "compute_market_calibration_quality: non-numeric brier_score '{}' "
            "for market '{}'",
            raw_brier,
            market_type,
        )
        return 0.5

    quality = _clip(1.0 - brier / 0.25, 0.0, 1.0)
    logger.debug(
        "compute_market_calibration_quality [{}]: brier={:.4f} → quality={:.3f}",
        market_type,
        brier,
        quality,
    )
    return round(quality, 4)
