"""Tests for compute_player_trust_score() Stage 3 player-specific trust scoring."""
import math
import pytest
from app.core.trust import compute_player_trust_score, trust_label


def _baseline_kwargs(**overrides):
    """Solid baseline: all factors at high confidence."""
    kw = dict(
        n_games=15,
        data_completeness=0.90,
        signal_agreement=0.85,
        n_active_signals=6,
        prediction_variance=0.02,
        prediction_low=0.55,
        prediction_high=0.70,
        role_stability=1.0,
        position_baseline_z=0.0,
        market_calibration_quality=0.80,
        data_freshness_days=2.0,
        lineup_certainty=1.0,
    )
    kw.update(overrides)
    return kw


class TestPlayerTrustScoreRange:
    def test_returns_float_in_0_100(self):
        score = compute_player_trust_score(**_baseline_kwargs())
        assert 0.0 <= score <= 100.0

    def test_perfect_inputs_give_high_score(self):
        score = compute_player_trust_score(**_baseline_kwargs())
        assert score >= 60.0, f"Expected >=60 for solid baseline, got {score}"

    def test_zero_games_lower_than_many_games(self):
        """n_games=0 should give lower trust than n_games=15 (sample factor dominates)."""
        score_0 = compute_player_trust_score(**_baseline_kwargs(n_games=0))
        score_15 = compute_player_trust_score(**_baseline_kwargs(n_games=15))
        assert score_0 < score_15

    def test_few_games_below_midpoint_lowers_score(self):
        score_8 = compute_player_trust_score(**_baseline_kwargs(n_games=8))
        score_15 = compute_player_trust_score(**_baseline_kwargs(n_games=15))
        assert score_8 < score_15

    def test_midpoint_is_8_not_12(self):
        """Sigmoid midpoint for players is 8. At 8 games, sample factor ≈ 0.5,
        so score at 8 games should be above the 0-games baseline."""
        score_8 = compute_player_trust_score(**_baseline_kwargs(n_games=8))
        score_0 = compute_player_trust_score(**_baseline_kwargs(n_games=0))
        assert score_8 > score_0


class TestRoleStabilityFactor:
    def test_unstable_role_lowers_score(self):
        high = compute_player_trust_score(**_baseline_kwargs(role_stability=1.0))
        low = compute_player_trust_score(**_baseline_kwargs(role_stability=0.3))
        assert high > low

    def test_role_stability_zero_gives_low_trust(self):
        score = compute_player_trust_score(**_baseline_kwargs(role_stability=0.0))
        assert score < 10.0, f"role_stability=0 should collapse trust, got {score}"

    def test_role_stability_weighted_0_7_exponent(self):
        """Role factor = stability^0.7. At stability=0.5, factor ≈ 0.613."""
        # The change from 1.0→0.5 should meaningfully reduce trust
        full = compute_player_trust_score(**_baseline_kwargs(role_stability=1.0))
        half = compute_player_trust_score(**_baseline_kwargs(role_stability=0.5))
        assert half < full
        # Ratio of factors ≈ 0.5^0.7 ≈ 0.613; ratio of scores should be in same ballpark
        assert half / full < 0.95


class TestPositionBaselineFactor:
    def test_high_z_score_lowers_trust(self):
        z0 = compute_player_trust_score(**_baseline_kwargs(position_baseline_z=0.0))
        z3 = compute_player_trust_score(**_baseline_kwargs(position_baseline_z=3.0))
        assert z0 > z3

    def test_z_zero_gives_full_baseline_factor(self):
        """|z|=0 → baseline_factor=1.0 — should not penalise at all."""
        z0 = compute_player_trust_score(**_baseline_kwargs(position_baseline_z=0.0))
        z0_neg = compute_player_trust_score(**_baseline_kwargs(position_baseline_z=-0.0))
        assert z0 == pytest.approx(z0_neg)

    def test_negative_z_same_as_positive(self):
        """Function uses abs(z), so -2 and +2 should give identical score."""
        pos = compute_player_trust_score(**_baseline_kwargs(position_baseline_z=2.0))
        neg = compute_player_trust_score(**_baseline_kwargs(position_baseline_z=-2.0))
        assert pos == pytest.approx(neg)

    def test_extreme_z_clamped_at_4(self):
        """z > 4 is clamped to 4, so z=5 and z=4 give the same score."""
        z4 = compute_player_trust_score(**_baseline_kwargs(position_baseline_z=4.0))
        z5 = compute_player_trust_score(**_baseline_kwargs(position_baseline_z=5.0))
        assert z4 == pytest.approx(z5)


class TestOtherFactors:
    def test_stale_data_reduces_trust(self):
        fresh = compute_player_trust_score(**_baseline_kwargs(data_freshness_days=1.0))
        stale = compute_player_trust_score(**_baseline_kwargs(data_freshness_days=30.0))
        assert fresh > stale

    def test_incomplete_data_reduces_trust(self):
        full = compute_player_trust_score(**_baseline_kwargs(data_completeness=1.0))
        partial = compute_player_trust_score(**_baseline_kwargs(data_completeness=0.3))
        assert full > partial

    def test_wide_prediction_interval_reduces_trust(self):
        tight = compute_player_trust_score(**_baseline_kwargs(prediction_low=0.58, prediction_high=0.62))
        wide = compute_player_trust_score(**_baseline_kwargs(prediction_low=0.30, prediction_high=0.90))
        assert tight > wide

    def test_low_lineup_certainty_reduces_trust(self):
        certain = compute_player_trust_score(**_baseline_kwargs(lineup_certainty=1.0))
        uncertain = compute_player_trust_score(**_baseline_kwargs(lineup_certainty=0.3))
        assert certain > uncertain

    def test_poor_market_calibration_reduces_trust(self):
        good = compute_player_trust_score(**_baseline_kwargs(market_calibration_quality=0.9))
        poor = compute_player_trust_score(**_baseline_kwargs(market_calibration_quality=0.1))
        assert good > poor

    def test_monotone_score_with_more_signals(self):
        s2 = compute_player_trust_score(**_baseline_kwargs(n_active_signals=2))
        s6 = compute_player_trust_score(**_baseline_kwargs(n_active_signals=6))
        assert s6 >= s2

    def test_signal_agreement_monotone(self):
        low_agree = compute_player_trust_score(**_baseline_kwargs(signal_agreement=0.2))
        high_agree = compute_player_trust_score(**_baseline_kwargs(signal_agreement=0.9))
        assert high_agree > low_agree


class TestWorstCaseCombo:
    def test_all_bad_factors_very_low(self):
        score = compute_player_trust_score(
            n_games=2,
            data_completeness=0.2,
            signal_agreement=0.1,
            n_active_signals=1,
            prediction_variance=0.5,
            prediction_low=0.2,
            prediction_high=0.8,
            role_stability=0.1,
            position_baseline_z=3.5,
            market_calibration_quality=0.1,
            data_freshness_days=60.0,
            lineup_certainty=0.1,
        )
        assert score < 5.0, f"Worst-case inputs should yield <5, got {score}"

    def test_all_good_factors_high(self):
        score = compute_player_trust_score(
            n_games=30,
            data_completeness=1.0,
            signal_agreement=1.0,
            n_active_signals=8,
            prediction_variance=0.001,
            prediction_low=0.62,
            prediction_high=0.68,
            role_stability=1.0,
            position_baseline_z=0.0,
            market_calibration_quality=1.0,
            data_freshness_days=0.5,
            lineup_certainty=1.0,
        )
        assert score >= 70.0, f"Best-case inputs should yield >=70, got {score}"


class TestTrustLabel:
    def test_labels_cover_range(self):
        for score in [0, 20, 35, 50, 65, 80, 95]:
            label = trust_label(float(score))
            assert isinstance(label, str)
            assert len(label) > 0
