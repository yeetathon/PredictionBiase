"""Tests for MetaBlender learned probability blending."""
import pytest
import numpy as np
from app.core.meta_blender import MetaBlender, _fixed_blend, _features


class TestFixedBlend:
    def test_high_signal_quality_leans_toward_signals(self):
        """High agreement + completeness → more weight on signal."""
        b = _fixed_blend(ml_prob=0.60, sig_prob=0.70, sig_agree=1.0, data_complete=1.0)
        # ml_weight = 0.60 - 1.0*0.20 = 0.40; blend = 0.40*0.60 + 0.60*0.70 = 0.66
        assert b == pytest.approx(0.66, abs=0.01)

    def test_low_signal_quality_leans_toward_ml(self):
        """Low agreement → heavier ML weight."""
        b = _fixed_blend(ml_prob=0.60, sig_prob=0.70, sig_agree=0.0, data_complete=1.0)
        # ml_weight = 0.60; blend = 0.60*0.60 + 0.40*0.70 = 0.64
        assert b == pytest.approx(0.64, abs=0.01)

    def test_output_clipped_to_valid_range(self):
        b = _fixed_blend(ml_prob=0.99, sig_prob=0.99, sig_agree=1.0, data_complete=1.0)
        assert 0.02 <= b <= 0.98

    def test_symmetric_inputs_give_same_output(self):
        b1 = _fixed_blend(0.60, 0.60, 0.5, 0.5)
        b2 = _fixed_blend(0.60, 0.60, 0.8, 0.8)
        # Both produce 0.60 (same ml/sig prob), regardless of weights
        assert b1 == pytest.approx(0.60, abs=0.001)
        assert b2 == pytest.approx(0.60, abs=0.001)


class TestFeatureVector:
    def test_feature_vector_length(self):
        x = _features(0.6, 0.55, 0.8, 0.01, 0.9, 60.0)
        assert len(x) == 6

    def test_trust_normalized_by_100(self):
        x = _features(0.6, 0.55, 0.8, 0.01, 0.9, 80.0)
        assert x[5] == pytest.approx(0.80, abs=0.001)

    def test_values_clipped_to_0_1(self):
        x = _features(1.5, -0.5, 2.0, -1.0, 3.0, 200.0)
        assert all(0.0 <= v <= 1.0 for v in x)


class TestMetaBlenderFallback:
    def setup_method(self):
        self.blender = MetaBlender()

    def test_blend_returns_float_in_range(self):
        p = self.blender.blend(
            ml_prob=0.62, sig_prob=0.58, sig_agree=0.8,
            pred_var=0.01, data_complete=0.9, trust=55.0,
        )
        assert 0.02 <= p <= 0.98

    def test_blend_no_trained_model_uses_fixed(self):
        """Without a trained model, blend should fall back to fixed formula."""
        # Ensure market has no trained model by using an uncommon market type
        p_untrained = self.blender.blend(
            ml_prob=0.60, sig_prob=0.70, sig_agree=1.0,
            pred_var=0.01, data_complete=1.0, trust=60.0,
            market_type="__nonexistent__",
        )
        p_fixed = _fixed_blend(0.60, 0.70, 1.0, 1.0)
        assert p_untrained == pytest.approx(p_fixed, abs=0.001)

    def test_blend_is_a_convex_combination_in_fixed_mode(self):
        """Fixed blend output must lie between ml_prob and sig_prob."""
        ml, sig = 0.50, 0.70
        p = self.blender.blend(
            ml_prob=ml, sig_prob=sig, sig_agree=0.6,
            pred_var=0.02, data_complete=0.8, trust=50.0,
            market_type="__nonexistent__",
        )
        assert min(ml, sig) - 0.01 <= p <= max(ml, sig) + 0.01


class TestMetaBlenderTraining:
    def _make_legs(self, n: int, hit_rate: float = 0.55) -> list:
        rng = np.random.default_rng(42)
        legs = []
        for i in range(n):
            outcome = 1 if rng.random() < hit_rate else 0
            legs.append({
                "ml_probability": 0.55 + rng.normal(0, 0.05),
                "signal_consensus_probability": 0.57 + rng.normal(0, 0.04),
                "signal_agreement": float(rng.uniform(0.5, 1.0)),
                "prediction_variance": float(rng.uniform(0.0, 0.05)),
                "data_completeness": float(rng.uniform(0.7, 1.0)),
                "trust_score": float(rng.uniform(40, 80)),
                "outcome": outcome,
                "market_type": "head_to_head",
            })
        return legs

    def test_train_rejects_insufficient_data(self):
        blender = MetaBlender()
        legs = self._make_legs(10)  # < MIN_SAMPLES (30)
        updated = blender.train_from_settlements(legs, market_type="head_to_head")
        assert updated is False

    def test_train_succeeds_with_adequate_data(self):
        blender = MetaBlender()
        legs = self._make_legs(50)
        updated = blender.train_from_settlements(legs, market_type="head_to_head")
        assert updated is True

    def test_trained_blend_is_in_valid_range(self):
        blender = MetaBlender()
        legs = self._make_legs(50)
        blender.train_from_settlements(legs, market_type="head_to_head")
        p = blender.blend(
            ml_prob=0.60, sig_prob=0.65, sig_agree=0.8,
            pred_var=0.01, data_complete=0.9, trust=55.0,
            market_type="head_to_head",
        )
        assert 0.02 <= p <= 0.98

    def test_legs_missing_required_fields_skipped(self):
        """Legs with missing ml/signal probs should be silently skipped."""
        blender = MetaBlender()
        legs = self._make_legs(40)
        # Corrupt some legs
        for i in range(10):
            legs[i]["ml_probability"] = None
        # Should still train on the 30 valid legs
        updated = blender.train_from_settlements(legs, market_type="player_disposals")
        assert updated is True

    def test_train_per_market_type_independent(self):
        """Training on H2H must not interfere with the player_disposals model."""
        blender = MetaBlender()
        legs = self._make_legs(50)

        # Explicitly ensure player_disposals is untrained for this assertion
        blender._models.pop("player_disposals", None)

        # Train H2H only
        blender.train_from_settlements(legs, market_type="head_to_head")

        # player_disposals model was NOT trained → must use fixed blend
        p_pd = blender.blend(
            ml_prob=0.60, sig_prob=0.70, sig_agree=1.0,
            pred_var=0.01, data_complete=1.0, trust=60.0,
            market_type="player_disposals",
        )
        p_fixed = _fixed_blend(0.60, 0.70, 1.0, 1.0)
        assert p_pd == pytest.approx(p_fixed, abs=0.001)
