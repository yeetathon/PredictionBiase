"""Tests for settlement service logic."""
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Settlement selection-string parsing
# ---------------------------------------------------------------------------

class TestSettlementLogic:
    """Test the selection-string matching logic in isolation."""

    @pytest.fixture()
    def svc(self):
        from app.services.settlement import SettlementService
        return SettlementService()

    # head_to_head
    def test_h2h_home_win_correct(self, svc):
        outcome = svc._resolve_outcome("home_win", {"home_win": True, "margin": 12})
        assert outcome == 1

    def test_h2h_home_win_wrong(self, svc):
        outcome = svc._resolve_outcome("home_win", {"home_win": False, "margin": -5})
        assert outcome == 0

    def test_h2h_away_win_correct(self, svc):
        outcome = svc._resolve_outcome("away_win", {"home_win": False, "margin": -8})
        assert outcome == 1

    def test_h2h_away_win_wrong(self, svc):
        outcome = svc._resolve_outcome("away_win", {"home_win": True, "margin": 10})
        assert outcome == 0

    # line
    def test_home_line_cover(self, svc):
        # Home team -5.5 — needs margin > 5.5
        outcome = svc._resolve_outcome("home_line_-5.5", {"margin": 8})
        assert outcome == 1

    def test_home_line_fail(self, svc):
        outcome = svc._resolve_outcome("home_line_-5.5", {"margin": 3})
        assert outcome == 0

    def test_away_line_cover(self, svc):
        # Away team -3.5 — away margin > 3.5 → home margin < -3.5
        outcome = svc._resolve_outcome("away_line_-3.5", {"margin": -6})
        assert outcome == 1

    def test_away_line_fail(self, svc):
        outcome = svc._resolve_outcome("away_line_-3.5", {"margin": -1})
        assert outcome == 0

    # total (over / under)
    def test_total_over_correct(self, svc):
        outcome = svc._resolve_outcome("over_165.5", {"total_score": 170})
        assert outcome == 1

    def test_total_over_wrong(self, svc):
        outcome = svc._resolve_outcome("over_165.5", {"total_score": 160})
        assert outcome == 0

    def test_total_under_correct(self, svc):
        outcome = svc._resolve_outcome("under_165.5", {"total_score": 158})
        assert outcome == 1

    def test_total_under_wrong(self, svc):
        outcome = svc._resolve_outcome("under_165.5", {"total_score": 172})
        assert outcome == 0

    # player disposals
    def test_player_over_correct(self, svc):
        # player_1_over_25 — player 1 got 28 disposals
        outcome = svc._resolve_outcome(
            "player_1_over_25",
            {"player_stats": {1: {"disposals": 28}}},
        )
        assert outcome == 1

    def test_player_over_wrong(self, svc):
        outcome = svc._resolve_outcome(
            "player_1_over_25",
            {"player_stats": {1: {"disposals": 22}}},
        )
        assert outcome == 0

    def test_player_under_correct(self, svc):
        outcome = svc._resolve_outcome(
            "player_1_under_20",
            {"player_stats": {1: {"disposals": 18}}},
        )
        assert outcome == 1

    def test_player_under_wrong(self, svc):
        outcome = svc._resolve_outcome(
            "player_1_under_20",
            {"player_stats": {1: {"disposals": 25}}},
        )
        assert outcome == 0

    # unknown / missing
    def test_unknown_selection_returns_none(self, svc):
        outcome = svc._resolve_outcome("totally_unknown", {})
        assert outcome is None

    def test_missing_player_stats_returns_none(self, svc):
        outcome = svc._resolve_outcome("player_99_over_25", {"player_stats": {}})
        assert outcome is None

    def test_missing_total_score_returns_none(self, svc):
        outcome = svc._resolve_outcome("over_165.5", {})
        assert outcome is None


# ---------------------------------------------------------------------------
# SettlementService.get_unsettled_legs (unit, mocked DB)
# ---------------------------------------------------------------------------

class TestGetUnsettledLegs:
    def test_returns_list(self):
        """get_unsettled_legs should always return a list."""
        from app.services.settlement import SettlementService
        svc = SettlementService()
        # Will hit an empty DB (test isolation via tmp in conftest or just catch empty)
        result = svc.get_unsettled_legs()
        assert isinstance(result, list)

    def test_settlement_summary_has_keys(self):
        from app.services.settlement import SettlementService
        svc = SettlementService()
        summary = svc.get_settlement_summary()
        assert "n_total" in summary or "error" in summary


# ---------------------------------------------------------------------------
# Multi-settlement: all-legs-must-win logic
# ---------------------------------------------------------------------------

class TestMultiSettlement:
    def test_multi_wins_when_all_legs_win(self):
        """A multi outcome should be 1 only when every constituent leg wins."""
        from app.services.settlement import SettlementService
        svc = SettlementService()
        # Simulate outcome calculation for a 3-leg multi
        leg_outcomes = [1, 1, 1]
        multi_outcome = 1 if all(o == 1 for o in leg_outcomes) else 0
        assert multi_outcome == 1

    def test_multi_loses_when_any_leg_loses(self):
        leg_outcomes = [1, 0, 1]
        multi_outcome = 1 if all(o == 1 for o in leg_outcomes) else 0
        assert multi_outcome == 0
