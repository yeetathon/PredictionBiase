"""Tests for core metrics calculations."""
import pytest
import numpy as np
from app.core.metrics import (
    implied_probability, remove_vig, compute_edge, compute_ev,
    compute_brier_score, compute_log_loss, compute_kelly_fraction,
    compute_confidence_score, compute_roi,
)


class TestOddsConversions:
    def test_implied_probability_fair(self):
        """Even money -> 50%."""
        assert implied_probability(2.0) == pytest.approx(0.5)

    def test_implied_probability_favourite(self):
        """Odds-on favourite."""
        assert implied_probability(1.5) == pytest.approx(0.6667, abs=0.001)

    def test_implied_probability_underdog(self):
        """Heavy underdog."""
        assert implied_probability(4.0) == pytest.approx(0.25)

    def test_implied_probability_edge_case(self):
        """Odds of 1.0 -> probability 1.0."""
        assert implied_probability(1.0) == 1.0


class TestVigRemoval:
    def test_vig_removal_sums_to_one(self):
        """After vig removal, probabilities sum to 1."""
        probs = [0.55, 0.50]  # Sum = 1.05 (5% overround)
        fair = remove_vig(probs)
        assert sum(fair) == pytest.approx(1.0, abs=1e-9)

    def test_vig_removal_proportional(self):
        """Vig removal is proportional."""
        probs = [0.60, 0.50]  # Asymmetric
        fair = remove_vig(probs)
        assert fair[0] > fair[1]

    def test_vig_removal_no_overround(self):
        """Already fair market (sums to 1) is unchanged."""
        probs = [0.5, 0.5]
        fair = remove_vig(probs)
        assert fair[0] == pytest.approx(0.5)
        assert fair[1] == pytest.approx(0.5)

    def test_vig_removal_empty(self):
        """Empty list returns empty list."""
        assert remove_vig([]) == []


class TestEVCalculation:
    def test_ev_positive_edge(self):
        """Positive EV when model prob > implied prob."""
        # Model says 60%, odds at 2.0 (implies 50%)
        ev = compute_ev(0.60, 2.0)
        assert ev == pytest.approx(0.20)  # 0.60 * 2.0 - 1 = 0.20

    def test_ev_negative(self):
        """Negative EV when model prob < implied prob."""
        ev = compute_ev(0.40, 2.0)
        assert ev == pytest.approx(-0.20)

    def test_ev_zero(self):
        """Zero EV when model exactly matches odds."""
        ev = compute_ev(0.5, 2.0)
        assert ev == pytest.approx(0.0)

    def test_edge_positive(self):
        """Edge is positive when model > market."""
        edge = compute_edge(0.60, 0.50)
        assert edge == pytest.approx(0.10)

    def test_edge_negative(self):
        edge = compute_edge(0.40, 0.50)
        assert edge == pytest.approx(-0.10)


class TestKelly:
    def test_kelly_positive_ev(self):
        """Kelly should be positive for positive EV bet."""
        k = compute_kelly_fraction(0.6, 2.0, fraction=1.0)
        assert k > 0

    def test_kelly_zero_for_negative_ev(self):
        """Kelly should be 0 or negative for negative EV (clipped at 0)."""
        k = compute_kelly_fraction(0.3, 2.0, fraction=1.0)
        assert k == pytest.approx(0.0)

    def test_fractional_kelly(self):
        """Fractional Kelly is fraction of full Kelly."""
        full_k = compute_kelly_fraction(0.6, 2.0, fraction=1.0)
        half_k = compute_kelly_fraction(0.6, 2.0, fraction=0.5)
        assert half_k == pytest.approx(full_k * 0.5, abs=1e-9)


class TestBrierScore:
    def test_perfect_predictions(self):
        """Brier score 0 for perfect predictions."""
        y = np.array([1, 0, 1, 0])
        p = np.array([1.0, 0.0, 1.0, 0.0])
        assert compute_brier_score(y, p) == pytest.approx(0.0)

    def test_worst_predictions(self):
        """Brier score 1 for perfectly wrong predictions."""
        y = np.array([1, 1])
        p = np.array([0.0, 0.0])
        assert compute_brier_score(y, p) == pytest.approx(1.0)

    def test_baseline_uniform(self):
        """Uniform 0.5 predictions -> Brier score 0.25."""
        y = np.array([1, 0, 1, 0])
        p = np.array([0.5, 0.5, 0.5, 0.5])
        assert compute_brier_score(y, p) == pytest.approx(0.25)


class TestROI:
    def test_roi_all_wins(self):
        """100% win rate at 2.0 odds -> 100% ROI."""
        outcomes = np.array([1, 1, 1])
        odds = np.array([2.0, 2.0, 2.0])
        r = compute_roi(outcomes, odds)
        assert r["roi"] == pytest.approx(1.0)
        assert r["hit_rate"] == pytest.approx(1.0)

    def test_roi_all_losses(self):
        """0% win rate -> -100% ROI."""
        outcomes = np.array([0, 0, 0])
        odds = np.array([2.0, 2.0, 2.0])
        r = compute_roi(outcomes, odds)
        assert r["roi"] == pytest.approx(-1.0)

    def test_roi_breakeven(self):
        """50% win rate at 2.0 odds -> 0% ROI."""
        outcomes = np.array([1, 0, 1, 0])
        odds = np.array([2.0, 2.0, 2.0, 2.0])
        r = compute_roi(outcomes, odds)
        assert r["roi"] == pytest.approx(0.0)

    def test_roi_fields(self):
        outcomes = np.array([1, 0])
        odds = np.array([3.0, 2.0])
        r = compute_roi(outcomes, odds)
        assert "n_bets" in r
        assert "hit_rate" in r
        assert "profit" in r


class TestConfidenceScore:
    def test_high_edge_high_confidence(self):
        c = compute_confidence_score(0.70, 0.50, 0.20, n_historical_games=60)
        assert c > 50

    def test_no_edge_low_confidence(self):
        c = compute_confidence_score(0.50, 0.50, 0.0, n_historical_games=0)
        assert c < 40
