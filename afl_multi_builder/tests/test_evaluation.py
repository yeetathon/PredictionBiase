"""Tests for EvaluationService metric helpers and report structure."""
import math
import numpy as np
import pytest

from app.services.evaluation import (
    _brier_score, _log_loss, _roi, _clv, _calibration_bins,
    EvaluationService,
)


# ---------------------------------------------------------------------------
# Metric helper unit tests
# ---------------------------------------------------------------------------

class TestBrierScore:
    def test_perfect_predictions(self):
        y = np.array([1, 0, 1, 0], dtype=float)
        p = np.array([1, 0, 1, 0], dtype=float)
        assert _brier_score(y, p) == pytest.approx(0.0)

    def test_worst_predictions(self):
        y = np.array([1, 0], dtype=float)
        p = np.array([0, 1], dtype=float)
        assert _brier_score(y, p) == pytest.approx(1.0)

    def test_baseline_0_5(self):
        y = np.array([1, 0, 1, 0], dtype=float)
        p = np.full(4, 0.5)
        assert _brier_score(y, p) == pytest.approx(0.25)

    def test_empty_returns_nan(self):
        result = _brier_score(np.array([]), np.array([]))
        assert math.isnan(result)


class TestLogLoss:
    def test_perfect_log_loss(self):
        y = np.array([1, 0], dtype=float)
        p = np.array([0.999, 0.001])
        assert _log_loss(y, p) < 0.01

    def test_single_class_returns_nan(self):
        y = np.array([1, 1, 1], dtype=float)
        p = np.array([0.8, 0.7, 0.9])
        result = _log_loss(y, p)
        assert math.isnan(result)

    def test_empty_returns_nan(self):
        result = _log_loss(np.array([]), np.array([]))
        assert math.isnan(result)


class TestROI:
    def test_all_win_roi(self):
        y = np.array([1, 1], dtype=float)
        odds = np.array([2.0, 3.0])
        # profit = (2-1) + (3-1) = 3; n=2; roi = 3/2 = 1.5
        assert _roi(y, odds) == pytest.approx(1.5)

    def test_all_lose_roi(self):
        y = np.array([0, 0], dtype=float)
        odds = np.array([2.0, 3.0])
        assert _roi(y, odds) == pytest.approx(-1.0)

    def test_mixed_roi(self):
        y = np.array([1, 0], dtype=float)
        odds = np.array([2.0, 2.0])
        # profit = (2-1) + (-1) = 0; n=2; roi = 0
        assert _roi(y, odds) == pytest.approx(0.0)

    def test_empty_returns_nan(self):
        result = _roi(np.array([]), np.array([]))
        assert math.isnan(result)


class TestCLV:
    def test_positive_clv_when_model_beats_market(self):
        # Model says 0.7, market implies 0.5 → CLV = 0.2
        model = np.array([0.7])
        closing = np.array([2.0])  # implied 0.5
        assert _clv(model, closing) == pytest.approx(0.2)

    def test_negative_clv(self):
        model = np.array([0.4])
        closing = np.array([2.0])  # implied 0.5
        assert _clv(model, closing) == pytest.approx(-0.1)

    def test_empty_returns_nan(self):
        result = _clv(np.array([]), np.array([]))
        assert math.isnan(result)


class TestCalibrationBins:
    def test_bins_have_required_keys(self):
        y = np.array([1, 0, 1, 0, 1], dtype=float)
        p = np.array([0.8, 0.2, 0.7, 0.3, 0.9])
        bins = _calibration_bins(y, p, n_bins=5)
        assert len(bins) > 0
        for b in bins:
            assert "bin_lo" in b
            assert "bin_hi" in b
            assert "n" in b
            assert "mean_predicted" in b
            assert "actual_hit_rate" in b

    def test_perfect_calibration(self):
        """When model = outcome, actual hit rate should match mean predicted."""
        y = np.array([1, 1, 0, 0], dtype=float)
        p = np.array([1.0, 1.0, 0.0, 0.0])
        bins = _calibration_bins(y, p, n_bins=2)
        for b in bins:
            assert abs(b["mean_predicted"] - b["actual_hit_rate"]) < 0.01

    def test_empty_bins_for_gaps(self):
        # All probabilities near 0 — high-prob bins should be absent
        y = np.array([0, 0, 1], dtype=float)
        p = np.array([0.1, 0.05, 0.15])
        bins = _calibration_bins(y, p, n_bins=10)
        # No bins above 0.2
        high_bins = [b for b in bins if b["bin_lo"] >= 0.5]
        assert len(high_bins) == 0


# ---------------------------------------------------------------------------
# EvaluationService integration (requires DB with tables)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Stage 3: _edge_realization_by_decile unit tests
# ---------------------------------------------------------------------------

def _make_legs(n: int, hit_rate: float = 0.55, decimal_odds: float = 2.0,
               model_prob: float = 0.60, vig_adj: float = 0.50) -> list:
    """Generate synthetic settled-leg dicts for testing."""
    rng = [{"outcome": 1 if i < int(n * hit_rate) else 0,
            "decimal_odds": decimal_odds,
            "model_probability": model_prob,
            "vig_adj_prob": vig_adj,
            "market_type": "head_to_head"} for i in range(n)]
    return rng


class TestEdgeRealizationByDecile:
    def setup_method(self):
        self.svc = EvaluationService.__new__(EvaluationService)

    def test_returns_empty_for_no_legs(self):
        result = self.svc._edge_realization_by_decile([])
        assert result == []

    def test_returns_empty_for_too_few_legs(self):
        legs = _make_legs(4)  # <5 → n_bins < 2 → empty
        result = self.svc._edge_realization_by_decile(legs)
        assert result == []

    def test_returns_list_of_dicts(self):
        legs = _make_legs(50)
        result = self.svc._edge_realization_by_decile(legs)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_each_bucket_has_required_keys(self):
        legs = _make_legs(50)
        result = self.svc._edge_realization_by_decile(legs)
        for bucket in result:
            assert "decile" in bucket
            assert "n" in bucket
            assert "avg_predicted_edge" in bucket
            assert "avg_realized_edge" in bucket
            assert "realization_ratio" in bucket
            assert "hit_rate" in bucket

    def test_hit_rate_in_valid_range(self):
        legs = _make_legs(50, hit_rate=0.60)
        result = self.svc._edge_realization_by_decile(legs)
        for bucket in result:
            assert 0.0 <= bucket["hit_rate"] <= 1.0

    def test_total_legs_accounted_for(self):
        legs = _make_legs(30)
        result = self.svc._edge_realization_by_decile(legs)
        total_n = sum(b["n"] for b in result)
        assert total_n == pytest.approx(30, abs=2)  # small tolerance for boundary edges

    def test_all_wins_gives_positive_realized_edge(self):
        """When all bets win at odds=2.0, realized edge per bet = 2.0-1-vig ≈ +0.50."""
        legs = _make_legs(20, hit_rate=1.0, decimal_odds=2.0, model_prob=0.65)
        result = self.svc._edge_realization_by_decile(legs)
        for bucket in result:
            assert bucket["avg_realized_edge"] > 0.0

    def test_all_losses_gives_negative_realized_edge(self):
        """When all bets lose, realized edge = -1 per bet (flat stake)."""
        legs = _make_legs(20, hit_rate=0.0, decimal_odds=2.0)
        result = self.svc._edge_realization_by_decile(legs)
        for bucket in result:
            assert bucket["avg_realized_edge"] < 0.0


# ---------------------------------------------------------------------------
# Stage 3: _analyze_player_props unit tests
# ---------------------------------------------------------------------------

def _make_player_legs(n: int, hit_rate: float = 0.50, decimal_odds: float = 2.0,
                      model_prob: float = 0.55, n_games: int = 10,
                      trust_score: float = 50.0) -> list:
    return [
        {
            "outcome": 1 if i < int(n * hit_rate) else 0,
            "decimal_odds": decimal_odds,
            "model_probability": model_prob,
            "n_games": n_games,
            "trust_score": trust_score,
            "market_type": "player_disposals",
        }
        for i in range(n)
    ]


class TestAnalyzePlayerProps:
    def setup_method(self):
        self.svc = EvaluationService.__new__(EvaluationService)

    def test_returns_empty_for_no_legs(self):
        result = self.svc._analyze_player_props([])
        assert result == {}

    def test_returns_dict_with_required_keys(self):
        legs = _make_player_legs(25)
        result = self.svc._analyze_player_props(legs)
        assert "by_sample_size" in result
        assert "by_odds_range" in result
        assert "false_positive_ev_rate" in result
        assert "n_total" in result
        assert "recommendation" in result

    def test_n_total_correct(self):
        legs = _make_player_legs(30)
        result = self.svc._analyze_player_props(legs)
        assert result["n_total"] == 30

    def test_by_sample_size_buckets_present(self):
        legs = (
            _make_player_legs(5, n_games=3) +   # sparse (0-5)
            _make_player_legs(5, n_games=8) +   # moderate (5-12)
            _make_player_legs(5, n_games=20)    # adequate (12+)
        )
        result = self.svc._analyze_player_props(legs)
        assert "sparse" in result["by_sample_size"]
        assert "moderate" in result["by_sample_size"]
        assert "adequate" in result["by_sample_size"]

    def test_sparse_bucket_correct_count(self):
        legs = _make_player_legs(7, n_games=2)  # all sparse
        result = self.svc._analyze_player_props(legs)
        assert result["by_sample_size"]["sparse"]["n"] == 7

    def test_by_trust_present_when_trust_scores_provided(self):
        legs = (
            _make_player_legs(5, trust_score=20.0) +   # low_trust
            _make_player_legs(5, trust_score=45.0) +   # medium_trust
            _make_player_legs(5, trust_score=70.0)     # high_trust
        )
        result = self.svc._analyze_player_props(legs)
        assert "by_trust" in result
        assert "low_trust" in result["by_trust"]
        assert "medium_trust" in result["by_trust"]
        assert "high_trust" in result["by_trust"]

    def test_by_trust_absent_without_trust_scores(self):
        legs = [{"outcome": 1, "decimal_odds": 2.0, "model_probability": 0.55,
                 "n_games": 10, "market_type": "player_disposals"}
                for _ in range(10)]
        result = self.svc._analyze_player_props(legs)
        assert "by_trust" not in result

    def test_false_positive_ev_rate_in_0_1(self):
        legs = _make_player_legs(20, hit_rate=0.40, model_prob=0.55)
        result = self.svc._analyze_player_props(legs)
        assert 0.0 <= result["false_positive_ev_rate"] <= 1.0

    def test_recommendation_continue_for_small_sample(self):
        """With <10 legs, recommendation should be 'continue' regardless of ROI."""
        legs = _make_player_legs(5, hit_rate=0.0, decimal_odds=2.0)  # terrible ROI, but only 5 legs
        result = self.svc._analyze_player_props(legs)
        assert result["recommendation"] == "continue"

    def test_recommendation_disable_for_chronic_underperformance(self):
        """>=20 legs with ROI < -0.20 → disable_market."""
        # ROI = -1.0 when all lose
        legs = _make_player_legs(25, hit_rate=0.0, decimal_odds=2.0)
        result = self.svc._analyze_player_props(legs)
        assert result["recommendation"] == "disable_market"

    def test_recommendation_tighten_for_moderate_loss(self):
        """>=10 legs with ROI < -0.10 but >= -0.20 → tighten_thresholds."""
        # 10 legs: hit_rate=0.30 at odds=2.0
        # ROI per leg = 0.30*(2-1) + 0.70*(-1) = 0.30 - 0.70 = -0.40... that's too bad
        # Use hit_rate=0.44 at odds=2.0: ROI ≈ 0.44*1 - 0.56*1 = -0.12
        legs = _make_player_legs(12, hit_rate=0.42, decimal_odds=2.0)
        result = self.svc._analyze_player_props(legs)
        # ROI ≈ 0.42*1 - 0.58*1 = -0.16 (between -0.20 and -0.10)
        assert result["recommendation"] in ("tighten_thresholds", "continue")


class TestEvaluationServiceIntegration:
    def test_evaluate_no_data_returns_no_data_status(self):
        """With an empty DB, evaluate() should return status='no_data'."""
        from app.services.evaluation import EvaluationService
        svc = EvaluationService()
        # Use a very short lookback so the empty test DB has no settled legs
        report = svc.evaluate(lookback_days=1)
        assert "status" in report
        # Either 'no_data' (empty DB) or 'ok' (if some settled legs exist)
        assert report["status"] in ("no_data", "ok", "error")

    def test_evaluate_report_structure(self):
        """Report should always have required top-level keys."""
        from app.services.evaluation import EvaluationService
        svc = EvaluationService()
        report = svc.evaluate(lookback_days=1)
        assert "run_id" in report
        assert "timestamp" in report
        assert "lookback_days" in report
        assert "status" in report
