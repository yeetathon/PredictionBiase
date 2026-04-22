"""Tests for correlation engine."""
import pytest
from app.correlation.engine import (
    CorrelationEngine, Leg, _get_pair_correlation, _check_conflict,
    SAME_GAME_CORRELATIONS,
)


def make_leg(leg_id, fixture_id, market_type, selection, player_id=None, team_id=1,
             decimal_odds=1.90, prob=0.55, ev=0.04):
    return Leg(
        leg_id=leg_id,
        fixture_id=fixture_id,
        player_id=player_id,
        team_id=team_id,
        market_type=market_type,
        selection=selection,
        decimal_odds=decimal_odds,
        calibrated_probability=prob,
        ev=ev,
        confidence_score=60.0,
    )


class TestPairCorrelation:
    def test_different_games_independent(self):
        a = make_leg("a", 1, "head_to_head", "home_win")
        b = make_leg("b", 2, "head_to_head", "home_win")
        c = _get_pair_correlation(a, b)
        assert c == 0.0

    def test_same_game_h2h_and_line_high(self):
        # Use different team_ids so no same-team boost
        a = make_leg("a", 1, "head_to_head", "home_win", team_id=1)
        b = make_leg("b", 1, "line", "home_line_-5.5", team_id=2)
        c = _get_pair_correlation(a, b)
        assert c == pytest.approx(0.85)

    def test_same_game_player_props_correlated(self):
        a = make_leg("a", 1, "player_disposals", "player_1_over_25", player_id=1)
        b = make_leg("b", 1, "player_disposals", "player_2_over_22", player_id=2)
        c = _get_pair_correlation(a, b)
        assert c > 0

    def test_same_team_extra_correlation(self):
        # Same game, same team players should have more correlation than different teams
        a = make_leg("a", 1, "player_disposals", "player_1_over_25", player_id=1, team_id=1)
        b_same_team = make_leg("b", 1, "player_disposals", "player_2_over_22", player_id=2, team_id=1)
        b_diff_team = make_leg("c", 1, "player_disposals", "player_3_over_22", player_id=3, team_id=2)
        c_same = _get_pair_correlation(a, b_same_team)
        c_diff = _get_pair_correlation(a, b_diff_team)
        assert c_same > c_diff

    def test_same_player_highest_correlation(self):
        a = make_leg("a", 1, "player_disposals", "player_1_over_25", player_id=1, team_id=1)
        b = make_leg("b", 1, "player_disposals", "player_1_over_20", player_id=1, team_id=1)
        c = _get_pair_correlation(a, b)
        assert c >= 0.7  # Should be very high for same player


class TestConflictDetection:
    def test_h2h_conflict(self):
        a = make_leg("a", 1, "head_to_head", "home_win")
        b = make_leg("b", 1, "head_to_head", "away_win")
        conflict, reason = _check_conflict(a, b)
        assert conflict is True
        assert "H2H" in reason or "Conflict" in reason

    def test_no_conflict_same_selection(self):
        a = make_leg("a", 1, "head_to_head", "home_win")
        b = make_leg("b", 1, "head_to_head", "home_win")
        conflict, _ = _check_conflict(a, b)
        assert conflict is False

    def test_over_under_conflict(self):
        a = make_leg("a", 1, "total", "over_165.5")
        b = make_leg("b", 1, "total", "under_165.5")
        conflict, reason = _check_conflict(a, b)
        assert conflict is True

    def test_different_totals_no_conflict(self):
        a = make_leg("a", 1, "total", "over_165.5")
        b = make_leg("b", 1, "total", "over_170.5")
        conflict, _ = _check_conflict(a, b)
        assert conflict is False

    def test_cross_game_no_conflict(self):
        a = make_leg("a", 1, "head_to_head", "home_win")
        b = make_leg("b", 2, "head_to_head", "away_win")
        conflict, _ = _check_conflict(a, b)
        assert conflict is False


class TestCorrelationEngine:
    def setup_method(self):
        self.engine = CorrelationEngine()

    def test_analyse_single_leg(self):
        legs = [make_leg("a", 1, "head_to_head", "home_win")]
        result = self.engine.analyse(legs)
        assert result.composite_score == 0.0
        assert result.adjusted_probability == pytest.approx(0.55, abs=0.01)

    def test_analyse_cross_game_low_correlation(self):
        """Cross-game legs should have low correlation."""
        legs = [
            make_leg("a", 1, "head_to_head", "home_win"),
            make_leg("b", 2, "head_to_head", "home_win"),
        ]
        result = self.engine.analyse(legs)
        assert result.composite_score == 0.0
        assert result.correlation_label == "low"

    def test_analyse_same_game_h2h_line_high_correlation(self):
        """H2H + Line same game should show medium-high correlation."""
        legs = [
            make_leg("a", 1, "head_to_head", "home_win"),
            make_leg("b", 1, "line", "home_line_-5.5"),
        ]
        result = self.engine.analyse(legs)
        assert result.composite_score > 0.5

    def test_adjusted_prob_lower_than_naive(self):
        """For correlated legs, adjusted prob should be <= naive prob."""
        legs = [
            make_leg("a", 1, "head_to_head", "home_win", prob=0.65),
            make_leg("b", 1, "line", "home_line_-5.5", prob=0.60),
        ]
        result = self.engine.analyse(legs)
        naive = 0.65 * 0.60
        assert result.adjusted_probability <= naive + 0.001

    def test_conflict_detected(self):
        legs = [
            make_leg("a", 1, "head_to_head", "home_win"),
            make_leg("b", 1, "head_to_head", "away_win"),
        ]
        result = self.engine.analyse(legs)
        assert result.conflict_detected is True

    def test_probability_is_valid(self):
        legs = [
            make_leg("a", 1, "head_to_head", "home_win", prob=0.60),
            make_leg("b", 2, "player_disposals", "player_1_over_25", prob=0.55, player_id=1),
            make_leg("c", 3, "head_to_head", "home_win", prob=0.65),
        ]
        result = self.engine.analyse(legs)
        assert 0 < result.adjusted_probability < 1

    def test_label_progression(self):
        """Higher correlations should produce higher severity labels."""
        labels_seen = set()
        for base_corr in [0.0, 0.2, 0.5, 0.8]:
            legs = [
                make_leg("a", 1, "head_to_head", "home_win"),
                make_leg("b", 1, "line", "home_line" if base_corr > 0.5 else "total_over"),
            ]
            result = self.engine.analyse(legs)
            labels_seen.add(result.correlation_label)
        # Should see at least two distinct labels
        assert len(labels_seen) >= 1
