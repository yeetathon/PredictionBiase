"""Tests for multi builder and leg ranker."""
import pytest
from app.optimizer.multi_builder import MultiBuilder, LegRanker, Multi
from app.correlation.engine import Leg


def make_leg(leg_id, fixture_id=1, market_type="head_to_head", selection="home_win",
             decimal_odds=1.90, prob=0.60, ev=0.14, conf=65.0, player_id=None, team_id=1,
             prediction_low=0.0, prediction_high=0.0, signal_agreement=0.0,
             prediction_variance=0.0):
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
        prediction_low=prediction_low,
        prediction_high=prediction_high,
        signal_agreement=signal_agreement,
        prediction_variance=prediction_variance,
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


class TestCIWidthGate:
    """Stage 3: multi is rejected when avg prediction interval width > 0.55."""

    def setup_method(self):
        self.builder = MultiBuilder(min_legs=2, max_legs=4, max_correlation=0.9, min_ev=-0.5)

    def _wide_leg(self, leg_id, fixture_id):
        """Leg with very wide CI (width = 0.80 > 0.55)."""
        return make_leg(
            leg_id, fixture_id=fixture_id, decimal_odds=2.0, prob=0.55, ev=0.10,
            prediction_low=0.10, prediction_high=0.90,  # width = 0.80
        )

    def _tight_leg(self, leg_id, fixture_id):
        """Leg with tight CI (width = 0.10 < 0.55)."""
        return make_leg(
            leg_id, fixture_id=fixture_id, decimal_odds=2.0, prob=0.58, ev=0.12,
            prediction_low=0.53, prediction_high=0.63,  # width = 0.10
        )

    def test_wide_ci_legs_rejected_by_sentinel_ev(self):
        """Multis built from wide-CI legs should have ev=-999 (CI gate sentinel)."""
        legs = [self._wide_leg(f"W{i}", i) for i in range(3)]
        multis = self.builder.build(legs, n_legs=2, max_results=20)
        # All should be rejected (ev=-999) or filtered (len=0)
        for m in multis:
            assert m.ev == pytest.approx(-999.0) or "REJECTED" in m.explanation

    def test_tight_ci_legs_not_rejected_by_ci_gate(self):
        """Multis from tight-CI legs should NOT be rejected by the CI gate."""
        legs = [self._tight_leg(f"T{i}", i) for i in range(4)]
        multis = self.builder.build(legs, max_results=20)
        # At least some should survive (not all -999)
        non_rejected = [m for m in multis if m.ev != pytest.approx(-999.0)]
        assert len(non_rejected) > 0

    def test_zero_ci_legs_not_rejected(self):
        """Legs with prediction_low=prediction_high=0.0 are excluded from CI calc (no gate)."""
        legs = [
            make_leg(f"L{i}", fixture_id=i, decimal_odds=1.9, prob=0.58, ev=0.10,
                     prediction_low=0.0, prediction_high=0.0)
            for i in range(3)
        ]
        multis = self.builder.build(legs, n_legs=2, max_results=10)
        # Should build normally (no CI widths to compute → gate not triggered)
        non_rejected = [m for m in multis if "REJECTED: avg CI width" not in (m.explanation or "")]
        assert len(non_rejected) > 0

    def test_mixed_ci_average_below_threshold_passes(self):
        """One wide + one tight leg where average CI width < 0.55 should pass."""
        legs = [
            make_leg("A", fixture_id=1, decimal_odds=2.0, prob=0.55, ev=0.10,
                     prediction_low=0.40, prediction_high=0.70),  # width=0.30
            make_leg("B", fixture_id=2, decimal_odds=2.0, prob=0.57, ev=0.12,
                     prediction_low=0.48, prediction_high=0.66),  # width=0.18
        ]
        # avg width = (0.30+0.18)/2 = 0.24 < 0.55 → should NOT trigger CI gate
        multis = self.builder.build(legs, n_legs=2, max_results=5)
        ci_rejected = [m for m in multis if "REJECTED: avg CI width" in (m.explanation or "")]
        assert len(ci_rejected) == 0


class TestMarketWeightedSignalQuality:
    """Stage 3: signal disagreement weighted by market type in objective score."""

    def setup_method(self):
        self.builder = MultiBuilder(min_legs=2, max_legs=4, max_correlation=0.9, min_ev=-0.5)

    def test_high_signal_agreement_yields_higher_objective(self):
        """Legs with high signal_agreement should rank above low-agreement legs."""
        legs_agree = [
            make_leg(f"A{i}", fixture_id=i, decimal_odds=2.0, prob=0.60, ev=0.20,
                     signal_agreement=0.95, prediction_low=0.55, prediction_high=0.65)
            for i in range(3)
        ]
        legs_disagree = [
            make_leg(f"D{i}", fixture_id=i+10, decimal_odds=2.0, prob=0.60, ev=0.20,
                     signal_agreement=0.10, prediction_low=0.55, prediction_high=0.65)
            for i in range(3)
        ]
        multis_agree = self.builder.build(legs_agree, n_legs=2, max_results=5)
        multis_disagree = self.builder.build(legs_disagree, n_legs=2, max_results=5)

        if multis_agree and multis_disagree:
            best_agree = max(m.objective_score for m in multis_agree)
            best_disagree = max(m.objective_score for m in multis_disagree)
            assert best_agree > best_disagree

    def test_objective_score_positive_for_positive_ev(self):
        """Multis with positive EV should have positive objective_score."""
        legs = [
            make_leg(f"L{i}", fixture_id=i, decimal_odds=2.0, prob=0.62, ev=0.24,
                     signal_agreement=0.80, prediction_low=0.57, prediction_high=0.67)
            for i in range(3)
        ]
        multis = self.builder.build(legs, n_legs=2, max_results=5)
        valid = [m for m in multis if m.ev != pytest.approx(-999.0) and m.ev > 0]
        for m in valid:
            assert m.objective_score > 0.0


class TestLegRanker:
    def setup_method(self):
        self.ranker = LegRanker(min_edge=0.01, min_ev=0.01)

    def test_negative_ev_filtered(self):
        ranker = LegRanker(min_ev=0.05)
        legs = [
            make_leg("L1", ev=-0.05),
            make_leg("L2", ev=0.10),
        ]
        ranked, _ = ranker.rank(legs)
        assert all(l.ev >= 0.05 for l in ranked)

    def test_ranking_by_ev(self):
        legs = [make_leg(f"L{i}", ev=float(i)/20) for i in range(1, 6)]
        ranked, _ = self.ranker.rank(legs)
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
        ranked, _ = self.ranker.rank(legs)
        assert all(1.01 <= l.decimal_odds <= 50.0 for l in ranked)
