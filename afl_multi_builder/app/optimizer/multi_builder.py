"""
Multi builder / optimizer.
Generates and ranks 2-4 leg multis from a set of candidate legs.
Applies correlation penalties, filters conflicts, enforces constraints.
"""
import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from loguru import logger

from app.correlation.engine import CorrelationEngine, CorrelationResult, Leg
from app.core.config import settings
from app.core.metrics import compute_ev


@dataclass
class Multi:
    """A generated multi with full metadata."""
    multi_id: str
    legs: List[Leg]
    n_legs: int
    multi_type: str                # same_game / cross_game / mixed
    combined_odds: float
    raw_probability: float
    adjusted_probability: float
    ev: float
    correlation_score: float
    correlation_label: str
    risk_score: float              # 0-100, lower = safer
    penalty_factor: float
    explanation: str
    conflict_detected: bool = False
    leg_ids: List[str] = field(default_factory=list)


class MultiBuilder:
    """
    Generates and ranks multis from candidate legs.

    Constraints:
      - min/max legs per multi
      - max legs per game
      - max legs per player
      - max allowed correlation score
      - minimum EV threshold
      - no conflicting legs
    """

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
        mode: str = "value",  # "value" | "safe" | "same_game"
    ) -> List[Multi]:
        """
        Generate and rank multis from candidate legs.

        Precision-first: returns fewer, better multis. If the pool is too
        thin or no combination clears all quality gates, returns empty list.
        The caller is responsible for surfacing a "no bets found" message.
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
            combos = list(itertools.combinations(legs, size))
            logger.debug("MultiBuilder: evaluating {} combinations of size {}", len(combos), size)

            for combo in combos:
                combo_legs = list(combo)

                # Mode filter: same-game multis must share a fixture
                if mode == "same_game":
                    if len({leg.fixture_id for leg in combo_legs}) != 1:
                        continue

                # Structural constraints (per-game, per-player caps)
                if not self._check_constraints(combo_legs):
                    rejection_counts["constraint"] = rejection_counts.get("constraint", 0) + 1
                    continue

                # Conflict detection
                filtered = self.correlation_engine.filter_conflicting_legs(combo_legs)
                if len(filtered) < len(combo_legs):
                    rejection_counts["conflict"] = rejection_counts.get("conflict", 0) + 1
                    continue

                # Correlation gate — strict
                corr_result = self.correlation_engine.analyse(combo_legs)
                if corr_result.conflict_detected:
                    rejection_counts["conflict"] = rejection_counts.get("conflict", 0) + 1
                    continue
                if corr_result.composite_score > self.max_correlation:
                    rejection_counts[f"correlation>{self.max_correlation:.2f}"] = (
                        rejection_counts.get(f"correlation>{self.max_correlation:.2f}", 0) + 1
                    )
                    continue

                # Minimum adjusted probability gate:
                # a 3-leg multi at 54%^3 = ~15.7%; reject below 10%
                min_adj_prob = 0.10
                if corr_result.adjusted_probability < min_adj_prob:
                    rejection_counts[f"adj_prob<{min_adj_prob}"] = (
                        rejection_counts.get(f"adj_prob<{min_adj_prob}", 0) + 1
                    )
                    continue

                # EV gate
                multi = self._build_multi(combo_legs, corr_result)
                if multi.ev < self.min_ev:
                    rejection_counts[f"ev<{self.min_ev}"] = (
                        rejection_counts.get(f"ev<{self.min_ev}", 0) + 1
                    )
                    continue

                all_multis.append(multi)

        if rejection_counts:
            logger.info(
                "MultiBuilder: rejections — {}",
                ", ".join(f"{k}={v}" for k, v in sorted(rejection_counts.items())),
            )

        all_multis = self._rank(all_multis, mode)
        logger.info(
            "MultiBuilder: {} multis passed all quality gates (from {} legs, mode={})",
            len(all_multis), len(legs), mode,
        )
        return all_multis[:max_results]

    def _check_constraints(self, legs: List[Leg]) -> bool:
        """Check per-game and per-player constraints."""
        from collections import Counter
        game_counts = Counter(leg.fixture_id for leg in legs)
        if any(v > self.max_legs_per_game for v in game_counts.values()):
            return False
        player_counts = Counter(
            leg.player_id for leg in legs if leg.player_id is not None
        )
        if any(v > self.max_legs_per_player for v in player_counts.values()):
            return False
        return True

    def _build_multi(self, legs: List[Leg], corr: CorrelationResult) -> Multi:
        """Assemble a Multi dataclass from legs and correlation result."""
        n = len(legs)
        combined_odds = float(np.prod([leg.decimal_odds for leg in legs]))
        adj_prob = corr.adjusted_probability
        ev = compute_ev(adj_prob, combined_odds)

        # Multi type
        fixture_ids = {leg.fixture_id for leg in legs}
        if len(fixture_ids) == 1:
            multi_type = "same_game"
        elif len(fixture_ids) == n:
            multi_type = "cross_game"
        else:
            multi_type = "mixed"

        # Signal uncertainty: average disagreement across legs (if computed)
        signal_disagree_total = 0.0
        n_with_signals = 0
        for leg in legs:
            if leg.signal_agreement > 0.0:
                signal_disagree_total += 1.0 - leg.signal_agreement
                n_with_signals += 1
        avg_signal_disagree = signal_disagree_total / n_with_signals if n_with_signals > 0 else 0.0

        # Risk score (0-100): higher odds, higher correlation, more legs,
        # and higher signal disagreement all increase risk
        risk = min(100.0, (
            (1 - adj_prob) * 40 +
            corr.composite_score * 25 +
            (n - 2) * 5 +
            avg_signal_disagree * 20     # signal uncertainty contribution
        ))

        # Explanation
        leg_summaries = [f"{leg.selection}@{leg.decimal_odds:.2f}" for leg in legs]
        explanation = (
            f"{n}-leg {multi_type} multi: {', '.join(leg_summaries)}. "
            f"Adj. prob: {adj_prob:.1%}, EV: {ev:+.1%}. "
            f"{corr.explanation}"
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
            conflict_detected=corr.conflict_detected,
            leg_ids=[leg.leg_id for leg in legs],
        )

    def _rank(self, multis: List[Multi], mode: str) -> List[Multi]:
        """Sort multis by mode criterion."""
        if mode == "safe":
            return sorted(multis, key=lambda m: m.adjusted_probability, reverse=True)
        elif mode == "same_game":
            return sorted(multis, key=lambda m: m.ev, reverse=True)
        else:  # value
            return sorted(multis, key=lambda m: m.ev, reverse=True)

    def build_same_game_multis(
        self,
        legs: List[Leg],
        fixture_id: int,
        max_results: int = 10,
    ) -> List[Multi]:
        """Build multis restricted to a single game."""
        game_legs = [l for l in legs if l.fixture_id == fixture_id]
        return self.build(game_legs, mode="same_game", max_results=max_results)

    def build_cross_game_multis(
        self,
        legs: List[Leg],
        max_results: int = 10,
    ) -> List[Multi]:
        """Build multis preferring legs from different games."""
        return self.build(legs, mode="value", max_results=max_results)


class LegRanker:
    """
    Ranks individual legs by value.

    Precision-first: every leg must pass ALL hard gates before entering
    the pool. Nothing is included just to fill a count. If no legs pass,
    the pool is empty and the caller must return a no-bet result.

    Gate hierarchy (applied in order):
      1. Probability ≥ min_prob
      2. EV ≥ min_ev
      3. Odds in [1.05, max_odds]
      4. Confidence ≥ min_confidence
      5. Edge ≥ min_edge (vs vig-adjusted market)
      6. Signal agreement ≥ min_signal_agreement (if signals were computed)
    """

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

    def rank(self, legs: List[Leg], log_rejections: bool = True) -> List[Leg]:
        """
        Apply all hard gates and rank surviving legs by quality-weighted EV.

        Every rejection is logged with a specific reason. If the caller wants
        to understand why no multis were built, the log tells them exactly.
        """
        valid = []
        rejected_counts: Dict[str, int] = {}

        for leg in legs:
            reason = self._reject_reason(leg)
            if reason:
                rejected_counts[reason] = rejected_counts.get(reason, 0) + 1
                if log_rejections:
                    logger.debug(
                        "LEG REJECTED [{}] {} — {}: prob={:.1%} ev={:.2%} "
                        "edge={:.2%} conf={:.0f} odds={:.2f} sig_agree={:.0%}",
                        reason,
                        leg.selection,
                        leg.explanation[:60],
                        leg.calibrated_probability,
                        leg.ev,
                        leg.calibrated_probability - (1.0 / leg.decimal_odds if leg.decimal_odds > 0 else 1.0),
                        leg.confidence_score,
                        leg.decimal_odds,
                        leg.signal_agreement,
                    )
                continue
            valid.append(leg)

        if rejected_counts and log_rejections:
            logger.info(
                "LegRanker: {} legs rejected — {}",
                sum(rejected_counts.values()),
                ", ".join(f"{k}={v}" for k, v in sorted(rejected_counts.items())),
            )
        logger.info(
            "LegRanker: {}/{} legs passed all quality gates",
            len(valid), len(legs),
        )

        # Sort: primary = confidence-weighted EV (penalises high-uncertainty legs);
        # secondary = raw EV; tertiary = signal agreement
        return sorted(
            valid,
            key=lambda l: (
                l.confidence_score * l.ev,
                l.ev,
                l.signal_agreement,
            ),
            reverse=True,
        )

    def _reject_reason(self, leg: Leg) -> Optional[str]:
        """Return a reason string if the leg should be rejected, else None."""
        if leg.calibrated_probability < self.min_prob:
            return f"prob<{self.min_prob:.0%}"
        if leg.ev < self.min_ev:
            return f"ev<{self.min_ev:.0%}"
        if leg.decimal_odds < 1.05 or leg.decimal_odds > self.max_odds:
            return f"odds_out_of_range[{leg.decimal_odds:.2f}]"
        if leg.confidence_score < self.min_confidence:
            return f"confidence<{self.min_confidence:.0f}"
        # Edge: model must beat market by min_edge
        implied_market = 1.0 / leg.decimal_odds if leg.decimal_odds > 0 else 1.0
        edge = leg.calibrated_probability - implied_market
        if edge < self.min_edge:
            return f"edge<{self.min_edge:.0%}[{edge:.1%}]"
        # Signal agreement gate: only applied when signals were actually computed
        if leg.signal_agreement > 0.0 and leg.signal_agreement < self.min_signal_agreement:
            return f"signal_disagree<{self.min_signal_agreement:.0%}[{leg.signal_agreement:.0%}]"
        return None

    def get_value_legs(self, legs: List[Leg], top_n: int = 10) -> List[Leg]:
        return self.rank(legs)[:top_n]

    def get_safe_legs(self, legs: List[Leg], top_n: int = 10) -> List[Leg]:
        valid = self.rank(legs)
        return sorted(valid, key=lambda l: l.calibrated_probability, reverse=True)[:top_n]
