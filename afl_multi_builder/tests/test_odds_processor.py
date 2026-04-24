"""Tests for odds processing layer."""
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from app.pricing.odds_processor import OddsProcessor, ProcessedOdds
from app.core.metrics import implied_probability, remove_vig


def _make_loader():
    """Build a minimal DataLoader with in-memory data (no API calls)."""
    from app.data_ingestion.loader import DataLoader

    class _Provider:
        @property
        def fixtures_df(self):      return pd.DataFrame()
        @property
        def teams_df(self):         return pd.DataFrame()
        @property
        def players_df(self):       return pd.DataFrame()
        @property
        def team_stats_df(self):    return pd.DataFrame()
        @property
        def player_stats_df(self):  return pd.DataFrame()
        def load_upcoming_fixtures_df(self):
            return pd.DataFrame()

    class _NullOdds:
        @property
        def odds_df(self): return pd.DataFrame()

    return DataLoader(afl_provider=_Provider(), odds_provider=_NullOdds())


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
        processor = OddsProcessor(loader=_make_loader())
        processor.loader.load_odds_df = MagicMock(return_value=self._make_odds_df())
        result = processor.process_fixture_odds(64)
        assert "head_to_head" in result
        assert len(result["head_to_head"]) == 2  # home + away

    def test_best_odds_selected(self):
        """Processor selects best (highest) odds per selection."""
        processor = OddsProcessor(loader=_make_loader())
        processor.loader.load_odds_df = MagicMock(return_value=self._make_odds_df())
        result = processor.process_fixture_odds(64)
        h2h = {po.selection: po for po in result["head_to_head"]}
        # TAB offers 1.85 for home_win, better than Sportsbet's 1.80
        assert h2h["home_win"].decimal_odds == pytest.approx(1.85)

    def test_vig_adjusted_prob_sums_to_one(self):
        """Vig-adjusted probabilities should sum to ~1 for a binary market."""
        processor = OddsProcessor(loader=_make_loader())
        processor.loader.load_odds_df = MagicMock(return_value=self._make_odds_df())
        result = processor.process_fixture_odds(64)
        h2h = result["head_to_head"]
        total_vig_adj = sum(po.vig_adjusted_prob for po in h2h)
        assert total_vig_adj == pytest.approx(1.0, abs=0.01)

    def test_overround_positive(self):
        """Overround should be positive (bookmaker takes margin)."""
        processor = OddsProcessor(loader=_make_loader())
        processor.loader.load_odds_df = MagicMock(return_value=self._make_odds_df())
        result = processor.process_fixture_odds(64)
        for po in result["head_to_head"]:
            assert po.overround > 0

    def test_attach_model_probabilities(self):
        """Edge and EV are computed after attaching model probabilities."""
        processor = OddsProcessor(loader=_make_loader())
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
        processor = OddsProcessor(loader=_make_loader())
        processor.loader.load_odds_df = MagicMock(return_value=pd.DataFrame())
        result = processor.process_fixture_odds(99)
        assert result == {}

    def test_market_consensus(self):
        """Market consensus averages multiple bookmakers."""
        processor = OddsProcessor(loader=_make_loader())
        processor.loader.load_odds_df = MagicMock(return_value=self._make_odds_df())
        consensus = processor.get_market_consensus(64, "head_to_head")
        assert consensus is not None
        assert abs(sum(consensus.values()) - 1.0) < 0.01


class TestPlayerDisposalsIndependentVig:
    """Stage 3: player_disposals market uses independent over/under vig."""

    def _make_player_odds_both_sides(self):
        """Both over and under present for one player at a given line."""
        return pd.DataFrame([
            {
                "odds_id": 10, "fixture_id": 1, "market_type": "player_disposals",
                "selection": "player_42_over_25.5", "bookmaker": "Sportsbet",
                "decimal_odds": 1.90, "timestamp": "2024-04-01T10:00:00", "status": "active",
            },
            {
                "odds_id": 11, "fixture_id": 1, "market_type": "player_disposals",
                "selection": "player_42_under_25.5", "bookmaker": "Sportsbet",
                "decimal_odds": 1.95, "timestamp": "2024-04-01T10:00:00", "status": "active",
            },
        ])

    def _make_player_odds_one_side_only(self):
        """Only over is available (no under) for this player/line."""
        return pd.DataFrame([
            {
                "odds_id": 12, "fixture_id": 1, "market_type": "player_disposals",
                "selection": "player_99_over_18.5", "bookmaker": "TAB",
                "decimal_odds": 2.10, "timestamp": "2024-04-01T10:00:00", "status": "active",
            },
        ])

    def _make_player_odds_multi_bookmakers(self):
        """Both sides present from two bookmakers — best odds selected."""
        return pd.DataFrame([
            {
                "odds_id": 20, "fixture_id": 2, "market_type": "player_disposals",
                "selection": "player_7_over_30.5", "bookmaker": "Sportsbet",
                "decimal_odds": 1.80, "timestamp": "2024-04-01T10:00:00", "status": "active",
            },
            {
                "odds_id": 21, "fixture_id": 2, "market_type": "player_disposals",
                "selection": "player_7_over_30.5", "bookmaker": "TAB",
                "decimal_odds": 1.85, "timestamp": "2024-04-01T10:00:00", "status": "active",
            },
            {
                "odds_id": 22, "fixture_id": 2, "market_type": "player_disposals",
                "selection": "player_7_under_30.5", "bookmaker": "Sportsbet",
                "decimal_odds": 2.00, "timestamp": "2024-04-01T10:00:00", "status": "active",
            },
        ])

    def test_both_sides_vig_adjusted_sum_to_one(self):
        """When both over and under are present, vig-adj probs sum to 1."""
        processor = OddsProcessor(loader=_make_loader())
        processor.loader.load_odds_df = MagicMock(return_value=self._make_player_odds_both_sides())
        result = processor.process_fixture_odds(1)
        assert "player_disposals" in result
        entries = result["player_disposals"]
        assert len(entries) == 2
        total_vig_adj = sum(e.vig_adjusted_prob for e in entries)
        assert total_vig_adj == pytest.approx(1.0, abs=0.002)

    def test_both_sides_overround_positive(self):
        """Overround should be positive when both sides are present."""
        processor = OddsProcessor(loader=_make_loader())
        processor.loader.load_odds_df = MagicMock(return_value=self._make_player_odds_both_sides())
        result = processor.process_fixture_odds(1)
        for entry in result["player_disposals"]:
            assert entry.overround > 0.0

    def test_one_side_uses_raw_implied_no_normalisation(self):
        """When only one side is available, vig_adjusted_prob == raw_implied_prob."""
        processor = OddsProcessor(loader=_make_loader())
        processor.loader.load_odds_df = MagicMock(return_value=self._make_player_odds_one_side_only())
        result = processor.process_fixture_odds(1)
        assert "player_disposals" in result
        entries = result["player_disposals"]
        assert len(entries) == 1
        entry = entries[0]
        # No normalisation — vig_adjusted = raw (best estimate with one side)
        assert entry.vig_adjusted_prob == pytest.approx(entry.raw_implied_prob, abs=1e-6)
        # Overround is 0 (unknown, single-sided)
        assert entry.overround == 0.0

    def test_one_side_raw_implied_correct(self):
        """raw_implied_prob should equal 1/decimal_odds for one-sided market."""
        processor = OddsProcessor(loader=_make_loader())
        processor.loader.load_odds_df = MagicMock(return_value=self._make_player_odds_one_side_only())
        result = processor.process_fixture_odds(1)
        entry = result["player_disposals"][0]
        expected = 1.0 / 2.10
        assert entry.raw_implied_prob == pytest.approx(expected, abs=0.001)

    def test_best_odds_selected_per_direction(self):
        """When two bookmakers price the over, the higher odds wins."""
        processor = OddsProcessor(loader=_make_loader())
        processor.loader.load_odds_df = MagicMock(
            return_value=self._make_player_odds_multi_bookmakers()
        )
        result = processor.process_fixture_odds(2)
        entries = {e.selection: e for e in result["player_disposals"]}
        over_entry = entries["player_7_over_30.5"]
        # TAB offers 1.85 for over — should be selected over Sportsbet's 1.80
        assert over_entry.decimal_odds == pytest.approx(1.85)

    def test_multi_bookmaker_vig_adj_sums_to_one(self):
        """Best-odds selected from multi-bookmaker player market still sums to 1."""
        processor = OddsProcessor(loader=_make_loader())
        processor.loader.load_odds_df = MagicMock(
            return_value=self._make_player_odds_multi_bookmakers()
        )
        result = processor.process_fixture_odds(2)
        total = sum(e.vig_adjusted_prob for e in result["player_disposals"])
        assert total == pytest.approx(1.0, abs=0.002)

    def test_player_market_routed_separately_from_h2h(self):
        """player_disposals market is processed independently of head_to_head."""
        both = pd.concat([
            pd.DataFrame([
                {"odds_id": 1, "fixture_id": 5, "market_type": "head_to_head",
                 "selection": "home_win", "bookmaker": "TAB", "decimal_odds": 1.80,
                 "timestamp": "", "status": "active"},
                {"odds_id": 2, "fixture_id": 5, "market_type": "head_to_head",
                 "selection": "away_win", "bookmaker": "TAB", "decimal_odds": 2.10,
                 "timestamp": "", "status": "active"},
            ]),
            self._make_player_odds_both_sides().assign(fixture_id=5),
        ], ignore_index=True)
        processor = OddsProcessor(loader=_make_loader())
        processor.loader.load_odds_df = MagicMock(return_value=both)
        result = processor.process_fixture_odds(5)
        assert "head_to_head" in result
        assert "player_disposals" in result
        # H2H vig-adj sums to 1
        h2h_total = sum(e.vig_adjusted_prob for e in result["head_to_head"])
        assert h2h_total == pytest.approx(1.0, abs=0.01)
        # Player vig-adj also sums to 1 (independent)
        pd_total = sum(e.vig_adjusted_prob for e in result["player_disposals"])
        assert pd_total == pytest.approx(1.0, abs=0.002)
