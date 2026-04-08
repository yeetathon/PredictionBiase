"""Tests for probability calibration."""
import pytest
import numpy as np
from app.pricing.calibration import (
    IsotonicCalibrator, PlattCalibrator, evaluate_calibration
)


class TestIsotonicCalibrator:
    def setup_method(self):
        self.cal = IsotonicCalibrator()
        np.random.seed(42)
        self.raw_probs = np.clip(np.random.beta(5, 5, 100), 0.01, 0.99)
        self.y_true = (self.raw_probs + np.random.normal(0, 0.1, 100) > 0.5).astype(int)

    def test_fit_returns_self(self):
        result = self.cal.fit(self.raw_probs, self.y_true)
        assert result is self.cal

    def test_calibrate_returns_valid_probs(self):
        self.cal.fit(self.raw_probs, self.y_true)
        calibrated = self.cal.calibrate(self.raw_probs)
        assert all(0 <= p <= 1 for p in calibrated)

    def test_calibrate_same_shape(self):
        self.cal.fit(self.raw_probs, self.y_true)
        calibrated = self.cal.calibrate(self.raw_probs)
        assert len(calibrated) == len(self.raw_probs)

    def test_unfitted_returns_raw(self):
        """Unfitted calibrator returns raw probabilities."""
        cal = IsotonicCalibrator()
        result = cal.calibrate(self.raw_probs)
        np.testing.assert_array_equal(result, self.raw_probs)

    def test_calibrated_improves_or_maintains_brier(self):
        """Calibration should not significantly hurt Brier score."""
        from sklearn.metrics import brier_score_loss
        self.cal.fit(self.raw_probs[:80], self.y_true[:80])
        test_probs = self.raw_probs[80:]
        test_y = self.y_true[80:]
        cal_probs = self.cal.calibrate(test_probs)
        raw_brier = brier_score_loss(test_y, test_probs)
        cal_brier = brier_score_loss(test_y, cal_probs)
        # Allow slight degradation (calibrator may overfit on small samples)
        assert cal_brier < raw_brier + 0.05


class TestPlattCalibrator:
    def setup_method(self):
        self.cal = PlattCalibrator()
        np.random.seed(42)
        # Simulate overconfident model
        self.raw_probs = np.clip(np.random.beta(2, 2, 100), 0.1, 0.9)
        self.y_true = (np.random.uniform(0, 1, 100) < self.raw_probs).astype(int)

    def test_fit_and_calibrate(self):
        self.cal.fit(self.raw_probs, self.y_true)
        calibrated = self.cal.calibrate(self.raw_probs)
        assert all(0 <= p <= 1 for p in calibrated)
        assert len(calibrated) == len(self.raw_probs)


class TestEvaluateCalibration:
    def test_returns_required_fields(self):
        y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        probs = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.7, 0.3])
        report = evaluate_calibration(y, probs)
        assert "raw_brier_score" in report
        assert "raw_log_loss" in report
        assert "n_samples" in report
        assert "base_rate" in report

    def test_with_calibrated_includes_improvement(self):
        y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        raw = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.7, 0.3])
        cal = np.array([0.7, 0.3, 0.65, 0.35, 0.6, 0.4, 0.55, 0.45, 0.6, 0.4])
        report = evaluate_calibration(y, raw, cal)
        assert "calibrated_brier_score" in report
        assert "brier_improvement" in report

    def test_brier_range(self):
        y = np.array([1, 0, 1, 0])
        probs = np.array([0.5, 0.5, 0.5, 0.5])
        report = evaluate_calibration(y, probs)
        assert 0 <= report["raw_brier_score"] <= 1
