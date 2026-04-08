"""Tests for odds processing layer."""
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from app.pricing.odds_processor import OddsProcessor, ProcessedOdds
from app.core.metrics import implied_probability, remove_vig


class TestOddsProcessor:
    def _make_odds_df(self):
        return pd.DataFrame([
            {"odds_id": 1, "fixture_id": 64, "market_type": "head_to_head",
             "selection": "home_win", "bookmaker": "Sportsbet", "decimal_odds": 1.80,
             "american_odds": -125, "timestamp": "2024-03-28T09:00:00", "status": "active"},
            {"odds_id": 2, "fixture_id": 64, "market_type": "head_to_head",
             "selection": "away_win", "bookmaker": "Sportsbet", "decimal_odds": 2.10,
             "american_odds": 110, "timestamp": "2024-03-28T09:00:00", "status": "active"},
            {"odds_id": 3, "fixture_id": 64, "market_type": "head_to_head",
             "selection": "home_win", "bookmaker": "TAB", "decimal_odds": 1.85,
             "american_odds": -118, "timestamp": "2024-03-28T09:00:00", "status": "active"},
        ])

    def test_process_fixture_returns_market_dict(self):
        processor = OddsProcessor()
        processor.loader.load_odds_df = MagicMock(return_value=self._make_odds_df())
        result = processor.process_fixture_odds(64)
        assert "head_to_head" in result
        assert len(result["head_to_head"]) == 2  # home + away

    def test_best_odds_selected(self):
        """Processor selects best (highest) odds per selection."""
        processor = OddsProcessor()
        processor.loader.load_odds_df = MagicMock(return_value=self._make_odds_df())
        result = processor.process_fixture_odds(64)
        h2h = {po.selection: po for po in result["head_to_head"]}
        # TAB offers 1.85 for home_win, better than Sportsbet's 1.80
        assert h2h["home_win"].decimal_odds == pytest.approx(1.85)

    def test_vig_adjusted_prob_sums_to_one(self):
        """Vig-adjusted probabilities should sum to ~1 for a binary market."""
        processor = OddsProcessor()
        processor.loader.load_odds_df = MagicMock(return_value=self._make_odds_df())
        result = processor.process_fixture_odds(64)
        h2h = result["head_to_head"]
        total_vig_adj = sum(po.vig_adjusted_prob for po in h2h)
        assert total_vig_adj == pytest.approx(1.0, abs=0.01)

    def test_overround_positive(self):
        """Overround should be positive (bookmaker takes margin)."""
        processor = OddsProcessor()
        processor.loader.load_odds_df = MagicMock(return_value=self._make_odds_df())
        result = processor.process_fixture_odds(64)
        for po in result["head_to_head"]:
            assert po.overround > 0

    def test_attach_model_probabilities(self):
        """Edge and EV are computed after attaching model probabilities."""
        processor = OddsProcessor()
        processor.loader.load_odds_df = MagicMock(return_value=self._make_odds_df())
        result = processor.process_fixture_odds(64)
        processed = result["head_to_head"]
        model_probs = {"home_win": 0.60, "away_win": 0.40}
        calibrated_probs = {"home_win": 0.58, "away_win": 0.42}
        updated = processor.attach_model_probabilities(processed, model_probs, calibrated_probs)
        home = next(po for po in updated if po.selection == "home_win")
        assert home.model_probability == pytest.approx(0.60)
        assert home.calibrated_probability == pytest.approx(0.58)
        assert home.edge is not None
        assert home.ev is not None

    def test_empty_odds_returns_empty(self):
        processor = OddsProcessor()
        processor.loader.load_odds_df = MagicMock(return_value=pd.DataFrame())
        result = processor.process_fixture_odds(99)
        assert result == {}

    def test_market_consensus(self):
        """Market consensus averages multiple bookmakers."""
        processor = OddsProcessor()
        processor.loader.load_odds_df = MagicMock(return_value=self._make_odds_df())
        consensus = processor.get_market_consensus(64, "head_to_head")
        assert consensus is not None
        assert abs(sum(consensus.values()) - 1.0) < 0.01
