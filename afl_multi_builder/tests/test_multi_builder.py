"""Tests for multi builder and leg ranker."""
import pytest
from app.optimizer.multi_builder import MultiBuilder, LegRanker, Multi
from app.correlation.engine import Leg


def make_leg(leg_id, fixture_id=1, market_type="head_to_head", selection="home_win",
             decimal_odds=1.90, prob=0.60, ev=0.14, conf=65.0, player_id=None, team_id=1):
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
        confidence_score=conf,
        explanation="Test leg",
    )


class TestMultiBuilder:
    def setup_method(self):
        self.builder = MultiBuilder(min_legs=2, max_legs=4, max_correlation=0.9, min_ev=-0.5)

    def _legs(self):
        return [
            make_leg("L1", fixture_id=1, decimal_odds=1.85, prob=0.60, ev=0.11),
            make_leg("L2", fixture_id=2, decimal_odds=1.90, prob=0.58, ev=0.10),
            make_leg("L3", fixture_id=3, decimal_odds=2.10, prob=0.55, ev=0.16),
            make_leg("L4", fixture_id=4, decimal_odds=1.75, prob=0.62, ev=0.09),
        ]

    def test_build_returns_multis(self):
        legs = self._legs()
        multis = self.builder.build(legs, max_results=10)
        assert len(multis) > 0

    def test_all_multis_have_required_fields(self):
        legs = self._legs()
        multis = self.builder.build(legs, max_results=5)
        for m in multis:
            assert m.multi_id
            assert m.n_legs >= 2
            assert 0 < m.adjusted_probability < 1
            assert m.combined_odds > 1.0
            assert m.correlation_label in ("low", "medium", "high", "extreme")

    def test_combined_odds_correct(self):
        """Combined odds = product of leg odds."""
        legs = [make_leg(f"L{i}", fixture_id=i, decimal_odds=2.0) for i in range(3)]
        multis = self.builder.build(legs, n_legs=2, max_results=5)
        for m in multis:
            expected = 2.0 ** m.n_legs
            assert m.combined_odds == pytest.approx(expected, abs=0.01)

    def test_no_conflicting_legs(self):
        """Multi builder should reject conflicting legs."""
        legs = [
            make_leg("L1", fixture_id=1, market_type="head_to_head", selection="home_win"),
            make_leg("L2", fixture_id=1, market_type="head_to_head", selection="away_win"),
        ]
        multis = self.builder.build(legs, n_legs=2, max_results=5)
        for m in multis:
            assert not m.conflict_detected

    def test_max_legs_per_game_constraint(self):
        """Should not exceed max_legs_per_game per fixture."""
        builder = MultiBuilder(min_legs=2, max_legs=4, max_legs_per_game=1, min_ev=-1.0)
        legs = [make_leg(f"L{i}", fixture_id=1) for i in range(4)]  # all same game
        multis = builder.build(legs, n_legs=3, max_results=10)
        # With max 1 leg per game and all legs from same game, no 3-leg multi possible
        assert len(multis) == 0

    def test_same_game_mode(self):
        """Same-game mode only includes legs from one fixture."""
        legs = [
            make_leg("L1", fixture_id=1),
            make_leg("L2", fixture_id=1),
            make_leg("L3", fixture_id=2),
        ]
        multis = self.builder.build(legs, mode="same_game", max_results=10)
        for m in multis:
            fixture_ids = {leg.fixture_id for leg in m.legs}
            assert len(fixture_ids) == 1

    def test_value_mode_highest_ev_first(self):
        """Value mode should rank by EV descending."""
        legs = self._legs()
        multis = self.builder.build(legs, mode="value", max_results=10)
        if len(multis) >= 2:
            assert multis[0].ev >= multis[1].ev

    def test_safe_mode_highest_prob_first(self):
        """Safe mode should rank by adjusted probability descending."""
        legs = self._legs()
        multis = self.builder.build(legs, mode="safe", max_results=10)
        if len(multis) >= 2:
            assert multis[0].adjusted_probability >= multis[1].adjusted_probability


class TestLegRanker:
    def setup_method(self):
        self.ranker = LegRanker(min_edge=0.01, min_ev=0.01)

    def test_negative_ev_filtered(self):
        ranker = LegRanker(min_ev=0.05)
        legs = [
            make_leg("L1", ev=-0.05),
            make_leg("L2", ev=0.10),
        ]
        ranked = ranker.rank(legs)
        assert all(l.ev >= 0.05 for l in ranked)

    def test_ranking_by_ev(self):
        legs = [make_leg(f"L{i}", ev=float(i)/20) for i in range(1, 6)]
        ranked = self.ranker.rank(legs)
        evs = [l.ev for l in ranked]
        assert evs == sorted(evs, reverse=True)

    def test_get_value_legs_top_n(self):
        legs = [make_leg(f"L{i}", ev=0.10) for i in range(20)]
        top = self.ranker.get_value_legs(legs, top_n=5)
        assert len(top) <= 5

    def test_unrealistic_odds_filtered(self):
        legs = [
            make_leg("L1", decimal_odds=1000.0, ev=50.0),  # Too extreme
            make_leg("L2", decimal_odds=1.95, ev=0.05),
        ]
        ranked = self.ranker.rank(legs)
        assert all(1.01 <= l.decimal_odds <= 50.0 for l in ranked)
