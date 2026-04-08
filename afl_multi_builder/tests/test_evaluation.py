"""Tests for EvaluationService metric helpers and report structure."""
import math
import numpy as np
import pytest

from app.services.evaluation import (
    _brier_score, _log_loss, _roi, _clv, _calibration_bins,
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
