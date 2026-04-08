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
        max_results: int = 20,
        mode: str = "value",  # "value" | "safe" | "same_game"
    ) -> List[Multi]:
        """
        Generate and rank multis from candidate legs.

        Args:
            legs: candidate legs (already filtered for edge/EV)
            n_legs: specific number of legs, or None for all sizes
            max_results: max multis to return
            mode: "value" = rank by EV, "safe" = rank by hit probability, "same_game" = same fixture only
        """
        if not legs:
            return []

        sizes = [n_legs] if n_legs else list(range(self.min_legs, self.max_legs + 1))
        all_multis: List[Multi] = []

        for size in sizes:
            if size > len(legs):
                continue
            combos = list(itertools.combinations(legs, size))
            logger.debug(f"Evaluating {len(combos)} combinations of size {size}")

            for combo in combos:
                combo_legs = list(combo)

                # Apply mode filter
                if mode == "same_game":
                    fixture_ids = {leg.fixture_id for leg in combo_legs}
                    if len(fixture_ids) != 1:
                        continue

                # Constraint checks
                if not self._check_constraints(combo_legs):
                    continue

                # Conflict detection
                filtered = self.correlation_engine.filter_conflicting_legs(combo_legs)
                if len(filtered) < len(combo_legs):
                    continue  # Skip if any leg was removed as conflicting

                # Correlation analysis
                corr_result = self.correlation_engine.analyse(combo_legs)
                if corr_result.conflict_detected:
                    continue
                if corr_result.composite_score > self.max_correlation:
                    continue

                # Build multi
                multi = self._build_multi(combo_legs, corr_result)
                if multi.ev < self.min_ev:
                    continue

                all_multis.append(multi)

        # Rank
        all_multis = self._rank(all_multis, mode)
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

        # Risk score (0-100): higher odds, higher correlation, more legs = riskier
        risk = min(100.0, (
            (1 - adj_prob) * 50 +
            corr.composite_score * 30 +
            (n - 2) * 5
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
    Filters by edge, EV, uncertainty, market sanity.
    """

    def __init__(
        self,
        min_edge: float = None,
        min_ev: float = None,
        max_uncertainty: float = 0.5,
    ):
        self.min_edge = min_edge or settings.min_edge_threshold
        self.min_ev = min_ev or settings.min_ev_threshold
        self.max_uncertainty = max_uncertainty

    def rank(self, legs: List[Leg]) -> List[Leg]:
        """Filter and rank legs by value."""
        valid = []
        for leg in legs:
            if leg.ev < self.min_ev:
                continue
            # Uncertainty filter: reject legs where probability is very uncertain
            # (close to 0.5 with no edge)
            if abs(leg.calibrated_probability - 0.5) < 0.02:
                continue
            # Sanity: odds must be reasonable
            if leg.decimal_odds < 1.01 or leg.decimal_odds > 50.0:
                continue
            valid.append(leg)

        # Sort by EV descending, then by confidence
        return sorted(valid, key=lambda l: (l.ev, l.confidence_score), reverse=True)

    def get_value_legs(self, legs: List[Leg], top_n: int = 10) -> List[Leg]:
        """Get top value legs."""
        return self.rank(legs)[:top_n]

    def get_safe_legs(self, legs: List[Leg], top_n: int = 10) -> List[Leg]:
        """Get highest probability legs (with minimum edge)."""
        valid = [l for l in legs if l.ev >= self.min_ev]
        return sorted(valid, key=lambda l: l.calibrated_probability, reverse=True)[:top_n]
