"""
Multi builder / optimizer — v2.

v2 upgrades:
  - Variance-penalized leg scoring: high prediction_variance discounts EV.
  - Volatility stacking penalty in multi risk score.
  - Better multi objective: EV × sqrt(adj_probability) rewards both value
    AND hit-rate probability rather than pure EV which ignores long-shot risk.
  - Greedy optimizer for large pools (>20 legs) to avoid O(n^k) explosion.
  - Structured rejection log with per-reason detail.
  - Prediction interval width gate: very uncertain legs are rejected.
"""
import itertools
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from loguru import logger

from app.correlation.engine import CorrelationEngine, CorrelationResult, Leg
from app.core.config import settings
from app.core.metrics import compute_ev


# ── Rejection tracking ────────────────────────────────────────────────────────

@dataclass
class RejectionDetail:
    leg_id: str
    selection: str
    reason: str          # category: prob / ev / odds / confidence / edge / signal / interval
    detail: str          # human-readable value that failed the gate
    calibrated_probability: float
    ev: float
    edge: float
    confidence_score: float
    signal_agreement: float
    prediction_variance: float


# ── Multi dataclass ───────────────────────────────────────────────────────────

@dataclass
class Multi:
    multi_id: str
    legs: List[Leg]
    n_legs: int
    multi_type: str
    combined_odds: float
    raw_probability: float
    adjusted_probability: float
    ev: float
    correlation_score: float
    correlation_label: str
    risk_score: float
    penalty_factor: float
    explanation: str
    objective_score: float = 0.0    # v2: ranking objective
    volatility_score: float = 0.0   # v2: sum of leg prediction variances
    conflict_detected: bool = False
    leg_ids: List[str] = field(default_factory=list)


# ── Multi builder ─────────────────────────────────────────────────────────────

class MultiBuilder:
    """
    Generates and ranks multis from candidate legs.

    Constraints:
      - min/max legs per multi
      - max legs per game and per player
      - max allowed correlation
      - minimum EV
      - no conflicting legs

    v2: uses EV × sqrt(adj_prob) as ranking objective to balance edge and
    hit-rate. Applies volatility stacking penalty to risk score.
    Uses greedy construction when pool exceeds GREEDY_THRESHOLD.
    """

    GREEDY_THRESHOLD = 20   # use greedy when more legs than this

    def __init__(
        self,
        min_legs: int = None,
        max_legs: int = None,
        max_legs_per_game: int = None,
        max_legs_per_player: int = None,
        max_correlation: float = None,
        min_ev: float = None,
    ):
        self.min_legs = min_legs or settings.min_multi_legs
        self.max_legs = max_legs or settings.max_multi_legs
        self.max_legs_per_game = max_legs_per_game or settings.max_legs_per_game
        self.max_legs_per_player = max_legs_per_player or settings.max_legs_per_player
        self.max_correlation = max_correlation or settings.max_correlation_score
        self.min_ev = min_ev or settings.min_ev_threshold
        self.correlation_engine = CorrelationEngine()

    def build(
        self,
        legs: List[Leg],
        n_legs: Optional[int] = None,
        max_results: int = 10,
        mode: str = "value",
    ) -> List[Multi]:
        """
        Generate and rank multis from candidate legs.
        Precision-first: returns fewer, better multis.
        """
        min_pool = settings.min_pool_size_for_multi
        if len(legs) < min_pool:
            logger.info(
                "MultiBuilder: pool has {} legs (need {}). No multis generated.",
                len(legs), min_pool,
            )
            return []

        sizes = [n_legs] if n_legs else list(range(self.min_legs, self.max_legs + 1))
        all_multis: List[Multi] = []
        rejection_counts: Dict[str, int] = {}

        for size in sizes:
            if size > len(legs):
                continue

            # Use greedy construction for large pools to avoid combinatorial explosion
            if len(legs) > self.GREEDY_THRESHOLD:
                logger.info(
                    "MultiBuilder: {} legs > {} threshold, using greedy construction for size {}",
                    len(legs), self.GREEDY_THRESHOLD, size,
                )
                candidates = self._greedy_candidates(legs, size, mode)
            else:
                candidates = list(itertools.combinations(legs, size))

            logger.debug("MultiBuilder: evaluating {} combos of size {}", len(candidates), size)

            for combo in candidates:
                combo_legs = list(combo)

                if mode == "same_game":
                    if len({leg.fixture_id for leg in combo_legs}) != 1:
                        continue

                if not self._check_constraints(combo_legs):
                    rejection_counts["constraint"] = rejection_counts.get("constraint", 0) + 1
                    continue

                filtered = self.correlation_engine.filter_conflicting_legs(combo_legs)
                if len(filtered) < len(combo_legs):
                    rejection_counts["conflict"] = rejection_counts.get("conflict", 0) + 1
                    continue

                corr_result = self.correlation_engine.analyse(combo_legs)
                if corr_result.conflict_detected:
                    rejection_counts["conflict"] = rejection_counts.get("conflict", 0) + 1
                    continue
                if corr_result.composite_score > self.max_correlation:
                    key = f"correlation>{self.max_correlation:.2f}"
                    rejection_counts[key] = rejection_counts.get(key, 0) + 1
                    continue

                # Minimum adjusted probability gate
                if corr_result.adjusted_probability < 0.10:
                    rejection_counts["adj_prob<10%"] = rejection_counts.get("adj_prob<10%", 0) + 1
                    continue

                multi = self._build_multi(combo_legs, corr_result)
                if multi.ev < self.min_ev:
                    key = f"ev<{self.min_ev:.0%}"
                    rejection_counts[key] = rejection_counts.get(key, 0) + 1
                    continue

                all_multis.append(multi)

        if rejection_counts:
            logger.info(
                "MultiBuilder rejections — {}",
                ", ".join(f"{k}={v}" for k, v in sorted(rejection_counts.items())),
            )

        all_multis = self._rank(all_multis, mode)
        logger.info(
            "MultiBuilder: {} multis passed all quality gates (from {} legs, mode={})",
            len(all_multis), len(legs), mode,
        )
        return all_multis[:max_results]

    def _greedy_candidates(
        self, legs: List[Leg], size: int, mode: str
    ) -> List[Tuple[Leg, ...]]:
        """
        Greedy multi construction for large pools.
        Sorts legs by objective score, builds top-N combinations by
        incrementally adding the highest-value leg that doesn't violate constraints.
        Returns a manageable set of promising combinations.
        """
        # Sort legs by their individual objective score
        sorted_legs = sorted(
            legs,
            key=lambda l: l.ev * (l.signal_agreement or 0.5) * (1.0 - min(0.5, l.prediction_variance * 10)),
            reverse=True,
        )

        # Take top 15 for brute-force
        top_legs = sorted_legs[:15]
        return list(itertools.combinations(top_legs, size))

    def _check_constraints(self, legs: List[Leg]) -> bool:
        game_counts = Counter(leg.fixture_id for leg in legs)
        if any(v > self.max_legs_per_game for v in game_counts.values()):
            return False
        player_counts = Counter(leg.player_id for leg in legs if leg.player_id is not None)
        if any(v > self.max_legs_per_player for v in player_counts.values()):
            return False
        return True

    def _build_multi(self, legs: List[Leg], corr: CorrelationResult) -> Multi:
        n = len(legs)
        combined_odds = float(np.prod([leg.decimal_odds for leg in legs]))
        adj_prob = corr.adjusted_probability
        ev = compute_ev(adj_prob, combined_odds)

        fixture_ids = {leg.fixture_id for leg in legs}
        if len(fixture_ids) == 1:
            multi_type = "same_game"
        elif len(fixture_ids) == n:
            multi_type = "cross_game"
        else:
            multi_type = "mixed"

        # v2: volatility stacking — sum of prediction variances across legs
        # High total variance = legs with uncertain predictions stacked together
        volatility_score = float(sum(leg.prediction_variance for leg in legs))
        volatility_penalty = min(0.30, volatility_score * 5.0)  # cap at 30% penalty

        # Signal disagreement average
        avg_disagree = 0.0
        n_with_signals = sum(1 for l in legs if l.signal_agreement > 0.0)
        if n_with_signals > 0:
            avg_disagree = sum(
                1.0 - l.signal_agreement for l in legs if l.signal_agreement > 0.0
            ) / n_with_signals

        # v2: risk score — more principled formula
        # Components: low adj_prob, high correlation, more legs, signal uncertainty, volatility
        risk = min(100.0, (
            (1.0 - adj_prob) * 35.0 +            # probability risk (max 35)
            corr.composite_score * 25.0 +          # correlation risk (max 25)
            (n - 2) * 5.0 +                        # leg count risk (max ~10 for 4-leg)
            avg_disagree * 20.0 +                  # signal uncertainty (max 20)
            volatility_score * 10.0                # prediction variance stacking (max ~10)
        ))

        # v2: objective score — balance EV and adj_prob
        # sqrt(adj_prob) penalizes very low probability multis even if high EV
        objective_score = float(ev * np.sqrt(max(0, adj_prob)))

        leg_summaries = [f"{leg.selection}@{leg.decimal_odds:.2f}" for leg in legs]
        explanation = (
            f"{n}-leg {multi_type} multi: {', '.join(leg_summaries)}. "
            f"Adj. prob: {adj_prob:.1%}, EV: {ev:+.1%}, Risk: {risk:.0f}/100. "
            f"Volatility: {volatility_score:.3f}. {corr.explanation}"
        )

        import hashlib
        leg_key = "_".join(sorted(leg.leg_id for leg in legs))
        multi_id = "M_" + hashlib.md5(leg_key.encode()).hexdigest()[:8]

        return Multi(
            multi_id=multi_id,
            legs=legs,
            n_legs=n,
            multi_type=multi_type,
            combined_odds=round(combined_odds, 2),
            raw_probability=corr.raw_probability,
            adjusted_probability=round(adj_prob, 6),
            ev=round(ev, 5),
            correlation_score=corr.composite_score,
            correlation_label=corr.correlation_label,
            risk_score=round(risk, 1),
            penalty_factor=corr.penalty_factor,
            explanation=explanation,
            objective_score=round(objective_score, 6),
            volatility_score=round(volatility_score, 4),
            conflict_detected=corr.conflict_detected,
            leg_ids=[leg.leg_id for leg in legs],
        )

    def _rank(self, multis: List[Multi], mode: str) -> List[Multi]:
        """
        Rank multis by mode.
        v2: "value" and "same_game" use objective_score (EV × √adj_prob).
        "safe" uses adjusted_probability (maximize hit rate).
        """
        if mode == "safe":
            return sorted(multis, key=lambda m: m.adjusted_probability, reverse=True)
        else:
            # value / same_game: objective_score balances edge AND probability
            return sorted(multis, key=lambda m: m.objective_score, reverse=True)

    def build_same_game_multis(
        self,
        legs: List[Leg],
        fixture_id: int,
        max_results: int = 10,
    ) -> List[Multi]:
        game_legs = [l for l in legs if l.fixture_id == fixture_id]
        return self.build(game_legs, mode="same_game", max_results=max_results)

    def build_cross_game_multis(
        self, legs: List[Leg], max_results: int = 10
    ) -> List[Multi]:
        return self.build(legs, mode="value", max_results=max_results)


# ── Leg ranker ────────────────────────────────────────────────────────────────

class LegRanker:
    """
    Ranks individual legs by value.

    Precision-first: every leg must pass ALL hard gates.

    Gate hierarchy (v2):
      1. Probability ≥ min_prob
      2. EV ≥ min_ev
      3. Odds in [1.05, max_odds]
      4. Confidence ≥ min_confidence
      5. Edge ≥ min_edge (vs vig-adjusted market)
      6. Signal agreement ≥ min_signal_agreement
      7. Prediction interval width ≤ max_interval_width (v2)
         Wide intervals = highly uncertain predictions

    v2 scoring: variance-penalized sort key rewards low-uncertainty edges.
    """

    MAX_INTERVAL_WIDTH = 0.60   # reject legs with >60pp wide prediction interval

    def __init__(
        self,
        min_edge: float = None,
        min_ev: float = None,
        min_prob: float = None,
        min_confidence: float = None,
        max_odds: float = None,
        min_signal_agreement: float = None,
    ):
        self.min_edge = min_edge if min_edge is not None else settings.min_edge_threshold
        self.min_ev = min_ev if min_ev is not None else settings.min_ev_threshold
        self.min_prob = min_prob if min_prob is not None else settings.min_calibrated_probability
        self.min_confidence = min_confidence if min_confidence is not None else settings.min_confidence_score
        self.max_odds = max_odds if max_odds is not None else settings.max_leg_odds
        self.min_signal_agreement = (
            min_signal_agreement if min_signal_agreement is not None
            else settings.min_signal_agreement
        )

    def rank(
        self, legs: List[Leg], log_rejections: bool = True
    ) -> Tuple[List[Leg], List[RejectionDetail]]:
        """
        Apply all hard gates and rank surviving legs.

        v2: returns (valid_legs, rejection_log) tuple.
        Sorting: variance-penalized EV × agreement × (1 - uncertainty_penalty).
        """
        valid: List[Leg] = []
        rejections: List[RejectionDetail] = []
        rejection_counts: Dict[str, int] = {}

        for leg in legs:
            reason, detail = self._reject_reason(leg)
            if reason:
                rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                implied_mkt = 1.0 / leg.decimal_odds if leg.decimal_odds > 0 else 1.0
                edge = leg.calibrated_probability - implied_mkt
                rejections.append(RejectionDetail(
                    leg_id=leg.leg_id,
                    selection=leg.selection,
                    reason=reason,
                    detail=detail,
                    calibrated_probability=leg.calibrated_probability,
                    ev=leg.ev,
                    edge=edge,
                    confidence_score=leg.confidence_score,
                    signal_agreement=leg.signal_agreement,
                    prediction_variance=leg.prediction_variance,
                ))
                if log_rejections:
                    logger.debug(
                        "LEG REJECTED [{}] {} — {} | prob={:.1%} ev={:.2%} "
                        "edge={:.2%} conf={:.0f} agree={:.0%} var={:.4f}",
                        reason, leg.selection, detail,
                        leg.calibrated_probability, leg.ev, edge,
                        leg.confidence_score, leg.signal_agreement,
                        leg.prediction_variance,
                    )
                continue
            valid.append(leg)

        if rejection_counts and log_rejections:
            logger.info(
                "LegRanker: {} rejected — {}",
                sum(rejection_counts.values()),
                ", ".join(f"{k}={v}" for k, v in sorted(rejection_counts.items())),
            )
        logger.info(
            "LegRanker: {}/{} legs passed all quality gates",
            len(valid), len(legs),
        )

        # v2 sort key: variance-penalized EV × signal agreement
        # uncertainty_penalty from prediction_variance: var=0.01 → 10% penalty
        def _score(l: Leg) -> Tuple[float, float, float]:
            uncertainty_penalty = min(0.50, l.prediction_variance * 10.0)
            quality = l.ev * (l.signal_agreement or 0.5) * (1.0 - uncertainty_penalty)
            return quality, l.ev, l.signal_agreement

        return sorted(valid, key=_score, reverse=True), rejections

    def _reject_reason(self, leg: Leg) -> Tuple[Optional[str], str]:
        """Return (reason_category, detail_string) or (None, '') if passes."""
        if leg.calibrated_probability < self.min_prob:
            return "prob", f"{leg.calibrated_probability:.1%} < {self.min_prob:.0%}"
        if leg.ev < self.min_ev:
            return "ev", f"{leg.ev:.2%} < {self.min_ev:.0%}"
        if leg.decimal_odds < 1.05 or leg.decimal_odds > self.max_odds:
            return "odds", f"{leg.decimal_odds:.2f} out of [1.05, {self.max_odds:.2f}]"
        if leg.confidence_score < self.min_confidence:
            return "confidence", f"{leg.confidence_score:.0f} < {self.min_confidence:.0f}"

        implied_market = 1.0 / leg.decimal_odds if leg.decimal_odds > 0 else 1.0
        edge = leg.calibrated_probability - implied_market
        if edge < self.min_edge:
            return "edge", f"{edge:.1%} < {self.min_edge:.0%}"

        if leg.signal_agreement > 0.0 and leg.signal_agreement < self.min_signal_agreement:
            return "signal_agree", f"{leg.signal_agreement:.0%} < {self.min_signal_agreement:.0%}"

        # v2: prediction interval width gate
        # prediction_low/high not stored on Leg directly — proxy via variance
        # if variance implies an extremely wide interval, reject
        # std ≈ sqrt(variance); wide = std > 0.25 → interval ~50pp wide
        if leg.prediction_variance > 0.0:
            approx_std = float(np.sqrt(leg.prediction_variance))
            interval_width = approx_std * 3.0   # ≈ 3-sigma spread across signals
            if interval_width > self.MAX_INTERVAL_WIDTH:
                return "uncertainty", (
                    f"predicted interval ≈{interval_width:.0%} wide "
                    f"(var={leg.prediction_variance:.4f})"
                )

        return None, ""

    def get_value_legs(self, legs: List[Leg], top_n: int = 10) -> List[Leg]:
        ranked, _ = self.rank(legs)
        return ranked[:top_n]

    def get_safe_legs(self, legs: List[Leg], top_n: int = 10) -> List[Leg]:
        ranked, _ = self.rank(legs)
        return sorted(ranked, key=lambda l: l.calibrated_probability, reverse=True)[:top_n]

    def get_rejections(self, legs: List[Leg]) -> List[RejectionDetail]:
        _, rejections = self.rank(legs, log_rejections=False)
        return rejections
