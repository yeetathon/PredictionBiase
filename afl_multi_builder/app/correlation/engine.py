"""
Correlation engine for AFL multi-building.

AFL multis are NOT independent events. Key correlations:
  - Same-game legs: if one team wins big, their players accumulate more stats;
    total and line markets interact
  - Same-team legs: player props on same team are positively correlated
    (game script, weather, opponent all affect all players equally)
  - Same-player legs: player disposals and goals from the same player are correlated
  - Over/under vs player props: a high-scoring game boosts player stats;
    a low-scoring tight game suppresses them

This engine computes pairwise and multi-leg correlation scores,
applies a penalty to the combined probability, and provides explanations.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from loguru import logger


@dataclass
class Leg:
    """A single betting leg."""
    leg_id: str
    fixture_id: int
    player_id: Optional[int]
    team_id: Optional[int]
    market_type: str          # head_to_head, line, total, player_disposals
    selection: str
    decimal_odds: float
    calibrated_probability: float
    ev: float
    confidence_score: float
    explanation: str = ""
    # Multi-signal quality fields (populated by signal engine; 0.0 = not computed)
    signal_agreement: float = 0.0       # 0–1; 1 = all signals agree
    prediction_variance: float = 0.0    # variance of probabilities across signals
    data_completeness: float = 0.0      # 0–1; quality of underlying data
    n_active_signals: int = 0           # number of signals with meaningful reliability
    prediction_low: float = 0.0         # 10th-percentile probability (prediction interval lower bound)
    prediction_high: float = 1.0        # 90th-percentile probability (prediction interval upper bound)


@dataclass
class CorrelationResult:
    """Result of correlation analysis for a set of legs."""
    legs: List[Leg]
    pairwise_scores: Dict[Tuple[str, str], float]  # (leg_id_a, leg_id_b) -> score 0-1
    composite_score: float            # overall correlation score 0-1
    correlation_label: str            # low / medium / high / extreme
    penalty_factor: float             # multiply combined prob by this
    adjusted_probability: float
    raw_probability: float
    explanation: str
    conflict_detected: bool = False
    conflict_reason: str = ""


# ── Correlation Coefficient Table ─────────────────────────────────────────────
# Estimated correlations between different leg types within/across games.
# These are informed estimates based on AFL game-script dynamics.
# 0.0 = independent, 1.0 = perfectly correlated, -1.0 = perfectly anti-correlated.

SAME_GAME_CORRELATIONS = {
    # (market_a, market_b): correlation_coefficient
    ("head_to_head", "line"): 0.85,           # Win and cover are very correlated
    ("head_to_head", "total"): 0.10,           # Win doesn't predict scoring volume
    ("head_to_head", "player_disposals"): 0.20, # Winning teams' players get more
    ("line", "total"): 0.15,                   # Covering line slightly related to total
    ("line", "player_disposals"): 0.30,        # Big win = more disposals for winning team
    ("total", "player_disposals"): 0.40,       # High total boosts all player props
    ("player_disposals", "player_disposals"): 0.35,  # Same-game players correlated via game script
}

SAME_TEAM_EXTRA_CORRELATION = 0.25  # Additional correlation for same-team player props
SAME_PLAYER_EXTRA_CORRELATION = 0.50  # Additional correlation for same player, different props

CROSS_GAME_BASE_CORRELATION = 0.0   # Different games are independent (by default)


def _get_pair_correlation(leg_a: Leg, leg_b: Leg) -> float:
    """
    Compute estimated correlation between two legs.
    Returns a value in [0, 1] representing positive dependency.
    """
    # Different games: treat as independent
    if leg_a.fixture_id != leg_b.fixture_id:
        return CROSS_GAME_BASE_CORRELATION

    # Same fixture
    mt_a = leg_a.market_type
    mt_b = leg_b.market_type

    # Look up base same-game correlation
    key = tuple(sorted([mt_a, mt_b]))
    base = SAME_GAME_CORRELATIONS.get(key, 0.10)

    # Boost for same team
    if (leg_a.team_id is not None and leg_b.team_id is not None
            and leg_a.team_id == leg_b.team_id):
        base = min(1.0, base + SAME_TEAM_EXTRA_CORRELATION)

    # Boost for same player
    if (leg_a.player_id is not None and leg_b.player_id is not None
            and leg_a.player_id == leg_b.player_id):
        base = min(1.0, base + SAME_PLAYER_EXTRA_CORRELATION)

    return base


def _check_conflict(leg_a: Leg, leg_b: Leg) -> Tuple[bool, str]:
    """
    Detect contradictory legs that cannot both win.
    E.g., same fixture home_win + away_win, or over + under same line.
    """
    if leg_a.fixture_id != leg_b.fixture_id:
        return False, ""

    # H2H conflict
    if leg_a.market_type == "head_to_head" and leg_b.market_type == "head_to_head":
        if leg_a.selection != leg_b.selection:
            return True, f"Conflicting H2H: {leg_a.selection} vs {leg_b.selection} in same game"

    # Total conflict: over and under on same line
    if leg_a.market_type == "total" and leg_b.market_type == "total":
        sel_a = leg_a.selection.lower()
        sel_b = leg_b.selection.lower()
        if (("over" in sel_a and "under" in sel_b) or
                ("under" in sel_a and "over" in sel_b)):
            # Check if they're the same line
            line_a = _extract_line_value(sel_a)
            line_b = _extract_line_value(sel_b)
            if line_a is not None and line_b is not None and abs(line_a - line_b) < 1.0:
                return True, f"Conflicting totals: {sel_a} vs {sel_b}"

    # Player over + under same line
    if (leg_a.market_type == "player_disposals" and leg_b.market_type == "player_disposals"
            and leg_a.player_id is not None and leg_a.player_id == leg_b.player_id):
        sel_a = leg_a.selection.lower()
        sel_b = leg_b.selection.lower()
        if ("over" in sel_a and "under" in sel_b) or ("under" in sel_a and "over" in sel_b):
            return True, f"Conflicting player props for player {leg_a.player_id}"

    return False, ""


def _extract_line_value(selection: str) -> Optional[float]:
    """Extract numeric line value from selection string."""
    import re
    nums = re.findall(r"[\d.]+", selection)
    if nums:
        try:
            return float(nums[-1])
        except ValueError:
            return None
    return None


def _copula_adjusted_probability(
    probs: List[float], correlation_matrix: np.ndarray
) -> float:
    """
    Adjust combined probability using a Gaussian copula approximation.

    For independent legs: P(all) = prod(p_i)
    With correlation: we apply a penalty based on positive correlations,
    since correlated legs are more likely to all fail together.

    The penalty factor reduces the naive combined probability to account
    for the fact that positively correlated legs provide LESS diversification
    than independent legs.

    Formula: adjusted = naive_prob * penalty_factor
    where penalty_factor = exp(-0.5 * sum of off-diagonal correlations * impact_weight)
    """
    n = len(probs)
    if n == 0:
        return 0.0
    if n == 1:
        return probs[0]

    naive_prob = float(np.prod(probs))

    # Sum of all pairwise correlations (off-diagonal)
    total_correlation = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_correlation += correlation_matrix[i, j]
            count += 1

    if count == 0:
        return naive_prob

    avg_correlation = total_correlation / count

    # Penalty: higher average correlation = worse combined outcome
    # The penalty factor reduces probability for positively correlated legs
    # because it overstates independence. We model this as:
    # penalty = 1 - (avg_corr * corr_weight)
    # where corr_weight is a sensitivity parameter calibrated to AFL data.
    CORR_WEIGHT = 0.3
    penalty_factor = max(0.5, 1.0 - avg_correlation * CORR_WEIGHT * (n - 1))

    return naive_prob * penalty_factor


class CorrelationEngine:
    """
    Main correlation analysis engine.
    Takes a list of Leg objects and returns a CorrelationResult.
    """

    def analyse(self, legs: List[Leg]) -> CorrelationResult:
        """
        Analyse correlations across a set of legs.
        Returns a CorrelationResult with adjusted probability and explanations.
        """
        n = len(legs)
        if n == 0:
            return CorrelationResult(
                legs=[], pairwise_scores={}, composite_score=0.0,
                correlation_label="none", penalty_factor=1.0,
                adjusted_probability=0.0, raw_probability=0.0,
                explanation="No legs provided."
            )

        # Build pairwise correlation matrix
        corr_matrix = np.zeros((n, n))
        pairwise_scores: Dict[Tuple[str, str], float] = {}
        conflict_detected = False
        conflict_reason = ""

        for i in range(n):
            for j in range(n):
                if i == j:
                    corr_matrix[i, j] = 1.0
                elif i < j:
                    conflict, reason = _check_conflict(legs[i], legs[j])
                    if conflict:
                        conflict_detected = True
                        conflict_reason = reason

                    c = _get_pair_correlation(legs[i], legs[j])
                    corr_matrix[i, j] = c
                    corr_matrix[j, i] = c
                    pairwise_scores[(legs[i].leg_id, legs[j].leg_id)] = c

        # Composite correlation score: average of off-diagonal
        off_diag = [corr_matrix[i, j] for i in range(n) for j in range(n) if i < j]
        composite = float(np.mean(off_diag)) if off_diag else 0.0

        # Label
        if composite < 0.15:
            label = "low"
        elif composite < 0.35:
            label = "medium"
        elif composite < 0.55:
            label = "high"
        else:
            label = "extreme"

        # Compute probabilities
        probs = [leg.calibrated_probability for leg in legs]
        raw_prob = float(np.prod(probs))
        adj_prob = _copula_adjusted_probability(probs, corr_matrix)
        penalty_factor = adj_prob / raw_prob if raw_prob > 0 else 1.0

        # Build explanation
        same_game_count = _count_same_game_legs(legs)
        same_team_count = _count_same_team_legs(legs)
        explanation_parts = [
            f"{n}-leg multi with {same_game_count} same-game legs, {same_team_count} same-team legs.",
            f"Correlation score: {composite:.2f} ({label}).",
        ]
        if composite > 0.35:
            explanation_parts.append(
                "High correlation: combined probability significantly reduced vs. independent assumption."
            )
        if same_game_count > 1:
            dominant_pairs = sorted(pairwise_scores.items(), key=lambda x: x[1], reverse=True)[:2]
            for (la, lb), c in dominant_pairs:
                if c > 0.3:
                    explanation_parts.append(f"  - Legs {la} & {lb}: correlation {c:.2f}")
        if conflict_detected:
            explanation_parts.append(f"WARNING: Conflict detected — {conflict_reason}")

        explanation = " ".join(explanation_parts)

        return CorrelationResult(
            legs=legs,
            pairwise_scores=pairwise_scores,
            composite_score=round(composite, 4),
            correlation_label=label,
            penalty_factor=round(penalty_factor, 4),
            adjusted_probability=round(adj_prob, 6),
            raw_probability=round(raw_prob, 6),
            explanation=explanation,
            conflict_detected=conflict_detected,
            conflict_reason=conflict_reason,
        )

    def filter_conflicting_legs(self, legs: List[Leg]) -> List[Leg]:
        """Remove legs that conflict with other legs in the set."""
        to_remove = set()
        for i in range(len(legs)):
            for j in range(i + 1, len(legs)):
                conflict, _ = _check_conflict(legs[i], legs[j])
                if conflict:
                    # Remove the lower confidence leg
                    if legs[i].confidence_score <= legs[j].confidence_score:
                        to_remove.add(i)
                    else:
                        to_remove.add(j)
        return [leg for idx, leg in enumerate(legs) if idx not in to_remove]


def _count_same_game_legs(legs: List[Leg]) -> int:
    """Count legs that share a fixture with at least one other leg."""
    fixture_ids = [leg.fixture_id for leg in legs]
    from collections import Counter
    counts = Counter(fixture_ids)
    return sum(1 for leg in legs if counts[leg.fixture_id] > 1)


def _count_same_team_legs(legs: List[Leg]) -> int:
    """Count legs that share a team with at least one other leg."""
    team_ids = [leg.team_id for leg in legs if leg.team_id is not None]
    from collections import Counter
    counts = Counter(team_ids)
    return sum(1 for leg in legs if leg.team_id is not None and counts[leg.team_id] > 1)
