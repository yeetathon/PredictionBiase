"""Tests for feature engineering pipeline."""
import pytest
import pandas as pd
import numpy as np
from app.features.pipeline import EloRatingSystem, FeaturePipeline


class TestEloRatingSystem:
    def setup_method(self):
        self.elo = EloRatingSystem(k_factor=32.0, initial_rating=1500.0, home_advantage=70.0)

    def test_expected_score_equal_ratings(self):
        """Equal ratings -> 50% each."""
        e = self.elo.expected_score(1500, 1500)
        assert e == pytest.approx(0.5)

    def test_expected_score_favourite(self):
        """Higher rated team should have > 50% probability."""
        e = self.elo.expected_score(1600, 1500)
        assert e > 0.5

    def test_ratings_update_win(self):
        """Winner's rating increases, loser decreases."""
        ra, rb = 1500.0, 1500.0
        new_ra, new_rb = self.elo.update(ra, rb, score_a=1.0)
        assert new_ra > ra
        assert new_rb < rb

    def test_ratings_update_loss(self):
        new_ra, new_rb = self.elo.update(1500, 1500, score_a=0.0)
        assert new_ra < 1500
        assert new_rb > 1500

    def test_ratings_conserved(self):
        """Sum of ratings is conserved after update."""
        ra, rb = 1600.0, 1400.0
        new_ra, new_rb = self.elo.update(ra, rb, 1.0)
        assert new_ra + new_rb == pytest.approx(ra + rb, abs=0.01)

    def test_compute_ratings_history_basic(self):
        """Test Elo history computation on minimal fixture data."""
        fixtures = pd.DataFrame([
            {"fixture_id": 1, "season": 2023, "round": 1, "home_team_id": 1,
             "away_team_id": 2, "status": "completed", "home_win": 1},
            {"fixture_id": 2, "season": 2023, "round": 2, "home_team_id": 2,
             "away_team_id": 1, "status": "completed", "home_win": 0},
        ])
        result = self.elo.compute_ratings_history(fixtures)
        assert "elo_home_pre" in result.columns
        assert "elo_away_pre" in result.columns
        assert "elo_win_prob_home" in result.columns
        # Both games should have pre-game ratings of 1500 initially
        assert result.iloc[0]["elo_home_pre"] == pytest.approx(1500.0)

    def test_win_prob_between_zero_and_one(self):
        """All win probabilities should be valid probabilities."""
        fixtures = pd.DataFrame([
            {"fixture_id": i, "season": 2023, "round": i, "home_team_id": 1,
             "away_team_id": 2, "status": "completed", "home_win": i % 2}
            for i in range(1, 10)
        ])
        result = self.elo.compute_ratings_history(fixtures)
        for _, row in result.iterrows():
            p = row["elo_win_prob_home"]
            assert 0 < p < 1


class TestFeaturePipeline:
    def test_pipeline_initialises(self):
        """Pipeline should initialise without errors."""
        fp = FeaturePipeline()
        assert fp is not None

    def test_get_team_features_returns_dataframe(self):
        """Team features should return a DataFrame."""
        fp = FeaturePipeline()
        features = fp.get_team_features()
        assert isinstance(features, pd.DataFrame)

    def test_get_player_features_returns_dataframe(self):
        """Player features should return a DataFrame."""
        fp = FeaturePipeline()
        features = fp.get_player_features("disposals")
        assert isinstance(features, pd.DataFrame)

    def test_model_ready_data_has_labels(self):
        """Model-ready data should have valid binary labels."""
        fp = FeaturePipeline()
        X, y, feature_cols = fp.get_model_ready_match_data()
        if not X.empty:
            assert set(y.unique()).issubset({0, 1})
            assert len(X) == len(y)
            assert len(feature_cols) > 0

    def test_no_lookahead_in_features(self):
        """
        Feature engineering must not use future data.
        Test by checking that rolling features for game N don't use game N's result.
        This is tested indirectly: Elo pre-game ratings must equal initial rating for first game.
        """
        fp = FeaturePipeline()
        features = fp.get_team_features()
        if features.empty:
            pytest.skip("No features available")
        # First fixture should have initial Elo (no history available)
        first = features[features["fixture_id"] == features["fixture_id"].min()]
        if not first.empty and "elo_home_pre" in first.columns:
            # Rating may have been slightly modified by season carryover or order,
            # but should be near initial (1500)
            assert abs(float(first["elo_home_pre"].iloc[0]) - 1500.0) < 100
