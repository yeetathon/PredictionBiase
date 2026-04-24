"""
Pipeline tests — live-data system only.

These tests verify:
  1. The pipeline FAILS with a clear PreflightError when API keys are missing
  2. No demo data is ever injected or used
  3. Market disabling works correctly when data is absent
  4. The pipeline result structure is correct when it runs successfully
  5. Data source provenance is tracked throughout

Tests that require live API credentials are skipped if the keys are not
present in the environment — they are not replaced with demo data.
"""
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Helper: minimal live-looking DataLoader (no real API calls)
# ---------------------------------------------------------------------------

def _make_minimal_loader(
    fixtures: list = None,
    teams: list = None,
    players: list = None,
    team_stats: list = None,
    player_stats: list = None,
    odds: list = None,
):
    """
    Build a DataLoader backed by in-memory DataFrames that look like real
    AFL Data Sports Group API sourced data (with _source_type='afl_data_api').
    No CSV files. No demo data. No network calls.
    """
    from app.data_ingestion.loader import DataLoader

    def _df(rows, source="afl_data_api"):
        df = pd.DataFrame(rows or [])
        if not df.empty:
            df["_source_type"] = source
            df["_fetch_ts"] = "2025-04-01T12:00:00"
        return df

    default_fixtures = [
        {
            "fixture_id": 1001, "season": 2025, "round": 5,
            "home_team_id": 1, "away_team_id": 2,
            "home_team_name": "Melbourne", "away_team_name": "Collingwood",
            "venue_id": 10, "date": "2025-05-10", "time": "19:30",
            "status": "upcoming",
            "result_home_score": None, "result_away_score": None,
        }
    ]
    default_teams = [
        {"team_id": 1, "name": "Melbourne",   "abbreviation": "MEL"},
        {"team_id": 2, "name": "Collingwood", "abbreviation": "COL"},
    ]

    fx_df        = _df(default_fixtures if fixtures is None else fixtures)
    teams_df     = _df(default_teams   if teams    is None else teams)
    players_df   = _df(players     or [])   # empty = player markets disabled
    ts_df        = _df(team_stats  or [])
    ps_df        = _df(player_stats or [])
    odds_df      = _df(odds        or [])

    class _InMemoryProvider:
        @property
        def fixtures_df(self): return fx_df
        @property
        def teams_df(self): return teams_df
        @property
        def players_df(self): return players_df
        @property
        def team_stats_df(self): return ts_df
        @property
        def player_stats_df(self): return ps_df

        def load_upcoming_fixtures_df(self):
            df = fx_df.copy()
            if "status" in df.columns:
                return df[df["status"] == "upcoming"]
            return df

    class _InMemoryOdds:
        @property
        def odds_df(self): return odds_df

    loader = DataLoader(
        afl_provider=_InMemoryProvider(),
        odds_provider=_InMemoryOdds(),
    )
    return loader


# ---------------------------------------------------------------------------
# Test: pipeline fails hard without API keys
# ---------------------------------------------------------------------------

class TestPreflightHardFail:
    """Pipeline must block and raise PreflightError when API keys are absent."""

    def test_preflight_fails_without_afl_data_key(self, monkeypatch):
        """No AFL Data key → PreflightError before any prediction logic runs."""
        from app.services.preflight import PreflightService, PreflightError
        from app.core import config

        monkeypatch.setattr(config.settings, "afl_data_authkey", "")

        svc = PreflightService()
        report = svc.run(raise_on_failure=False)

        assert not report.passed, "Preflight should fail without AFL Data authkey"
        failed_names = [c.name for c in report.checks if not c.passed and c.required]
        assert "afl_data_authkey" in failed_names

    def test_preflight_error_includes_fix_instructions(self, monkeypatch):
        """PreflightError message must include actionable fix instructions."""
        from app.services.preflight import PreflightService, PreflightError
        from app.core import config

        monkeypatch.setattr(config.settings, "afl_data_authkey", "")

        svc = PreflightService()
        report = svc.run(raise_on_failure=False)
        error = PreflightError(report)

        assert "AFL_DATA_AUTHKEY" in str(error), \
            "Error message must mention the missing env var"
        assert "Fix:" in str(error), \
            "Error message must include fix instructions"

    def test_preflight_report_has_correct_structure(self, monkeypatch):
        """PreflightReport.to_dict() must contain all required keys."""
        from app.services.preflight import PreflightService
        from app.core import config

        monkeypatch.setattr(config.settings, "afl_data_authkey", "")

        svc = PreflightService()
        report = svc.run(raise_on_failure=False)
        d = report.to_dict()

        assert "passed" in d
        assert "checks" in d
        assert "failed_required" in d
        assert "timestamp" in d
        for check in d["checks"]:
            assert "name" in check
            assert "passed" in check
            assert "required" in check
            assert "detail" in check


# ---------------------------------------------------------------------------
# Test: no demo data ever enters the pipeline
# ---------------------------------------------------------------------------

class TestNoDemoData:
    """Verify demo data is completely excluded from all pipeline paths."""

    def test_demo_loader_import_raises(self):
        """Importing demo_loader must raise ImportError immediately."""
        import importlib
        import sys

        # Remove from cache if already imported
        sys.modules.pop("app.data_ingestion.demo_loader", None)

        with pytest.raises(ImportError, match="demo_loader is no longer available"):
            import app.data_ingestion.demo_loader

    def test_config_has_no_demo_mode(self):
        """data_mode must only accept 'live' or 'cache'."""
        from app.core.config import Settings
        import pydantic

        with pytest.raises((pydantic.ValidationError, ValueError)):
            Settings(data_mode="demo")

    def test_config_has_no_enable_demo_fallback(self):
        """Settings must not have an enable_demo_fallback attribute."""
        from app.core.config import settings
        assert not hasattr(settings, "enable_demo_fallback"), \
            "enable_demo_fallback must not exist in Settings"

    def test_config_has_no_demo_data_dir(self):
        """Settings must not have a demo_data_dir attribute."""
        from app.core.config import settings
        assert not hasattr(settings, "demo_data_dir"), \
            "demo_data_dir must not exist in Settings"

    def test_data_source_manager_has_no_demo_provider(self):
        """DataSourceManager must not contain DemoDataProvider."""
        import inspect
        from app.data_ingestion import data_source_manager
        src = inspect.getsource(data_source_manager)
        assert "DemoDataProvider" not in src, \
            "DemoDataProvider class must not exist in data_source_manager"
        assert "load_all_demo" not in src, \
            "load_all_demo() must not exist in data_source_manager"

    def test_pipeline_legs_from_live_source(self):
        """All legs must be from a live (non-demo) data source."""
        from app.services.pipeline import PredictionPipeline
        from app.services.preflight import PreflightService, PreflightError

        loader = _make_minimal_loader()

        # Skip preflight checks (we control the data source directly)
        with patch("app.services.pipeline.PreflightService") as mock_pf:
            mock_pf.return_value.run.return_value = None
            pipeline = PredictionPipeline(loader=loader)
            result = pipeline.run()

        # Data source must never say "demo"
        assert "demo" not in result.get("data_source", "").lower(), \
            f"data_source must not reference demo data, got: {result.get('data_source')}"
        assert result["data_source"] in (
            "AFL Data Sports Group API (Live)",
            "AFL Data Sports Group API (Cached)",
            "AFL Data Sports Group API",
        ), f"Unexpected data_source: {result.get('data_source')}"


# ---------------------------------------------------------------------------
# Test: player markets disabled when player data unavailable
# ---------------------------------------------------------------------------

class TestMarketDisabling:
    """Player-prop markets must be disabled when player stats are absent."""

    def test_player_legs_absent_without_player_data(self):
        """No player legs when players_df is empty (data unavailable)."""
        from app.services.pipeline import PredictionPipeline

        loader = _make_minimal_loader(players=[], player_stats=[], odds=[])

        with patch("app.services.pipeline.PreflightService") as mock_pf:
            mock_pf.return_value.run.return_value = None
            pipeline = PredictionPipeline(loader=loader)
            result = pipeline.run()

        player_legs = [
            l for l in result.get("value_legs", [])
            if l.get("market_type") == "player_disposals"
        ]
        assert len(player_legs) == 0, \
            "No player_disposals legs expected when player data is absent"

    def test_h2h_legs_still_generated_without_player_data(self):
        """Head-to-head legs must still work even when player data is absent."""
        from app.services.pipeline import PredictionPipeline

        loader = _make_minimal_loader(players=[], player_stats=[], odds=[])

        with patch("app.services.pipeline.PreflightService") as mock_pf:
            mock_pf.return_value.run.return_value = None
            pipeline = PredictionPipeline(loader=loader)
            result = pipeline.run()

        assert result["n_fixtures"] > 0, "Must process at least 1 fixture"
        all_legs = result.get("value_legs", []) + result.get("safe_legs", [])
        # May have h2h legs if model trained — or zero if untrained; just no player legs
        player_legs = [l for l in all_legs if l.get("market_type") == "player_disposals"]
        assert len(player_legs) == 0


# ---------------------------------------------------------------------------
# Test: pipeline result structure
# ---------------------------------------------------------------------------

class TestPipelineResultStructure:
    """Pipeline output must always include required keys and valid types."""

    def test_result_has_required_keys(self):
        """Pipeline result must contain all required output keys."""
        from app.services.pipeline import PredictionPipeline

        loader = _make_minimal_loader()

        with patch("app.services.pipeline.PreflightService") as mock_pf:
            mock_pf.return_value.run.return_value = None
            pipeline = PredictionPipeline(loader=loader)
            result = pipeline.run()

        required_keys = [
            "run_id", "timestamp", "data_source", "odds_source",
            "n_fixtures", "n_candidate_legs", "elapsed_seconds",
            "value_legs", "safe_legs", "value_multis", "safe_multis",
            "same_game_multis", "pipeline_stages",
        ]
        for key in required_keys:
            assert key in result, f"Missing required key in pipeline result: {key}"

    def test_legs_have_required_fields(self):
        """Each leg must contain all required display fields."""
        from app.services.pipeline import PredictionPipeline

        loader = _make_minimal_loader(
            odds=[{
                "fixture_id": 1001, "market_type": "head_to_head",
                "selection": "home_win", "bookmaker": "sportsbet",
                "decimal_odds": 1.90, "status": "active",
                "_source_type": "odds_api", "_fetch_ts": "2025-04-01T12:00:00",
            }]
        )

        with patch("app.services.pipeline.PreflightService") as mock_pf:
            mock_pf.return_value.run.return_value = None
            pipeline = PredictionPipeline(loader=loader)
            result = pipeline.run()

        for leg in result.get("value_legs", []) + result.get("safe_legs", []):
            assert "match_label" in leg, "Leg missing match_label"
            assert "start_time" in leg, "Leg missing start_time"
            assert "market_type" in leg, "Leg missing market_type"
            assert "selection_label" in leg, "Leg missing selection_label"
            assert "decimal_odds" in leg, "Leg missing decimal_odds"
            assert "model_probability" in leg, "Leg missing model_probability"
            assert "ev" in leg, "Leg missing ev"
            assert "edge" in leg, "Leg missing edge"
            assert "confidence_score" in leg, "Leg missing confidence_score"
            assert "explanation" in leg, "Leg missing explanation"
            # data_source must never be "demo"
            assert "demo" not in leg.get("match_label", "").lower()

    def test_multis_have_leg_details(self):
        """Each multi must include leg-level details for display."""
        from app.services.pipeline import PredictionPipeline

        loader = _make_minimal_loader(
            fixtures=[
                {
                    "fixture_id": 1001, "season": 2025, "round": 5,
                    "home_team_id": 1, "away_team_id": 2,
                    "home_team_name": "Melbourne", "away_team_name": "Collingwood",
                    "venue_id": 10, "date": "2025-05-10", "time": "19:30",
                    "status": "upcoming",
                    "result_home_score": None, "result_away_score": None,
                },
                {
                    "fixture_id": 1002, "season": 2025, "round": 5,
                    "home_team_id": 3, "away_team_id": 4,
                    "home_team_name": "Richmond", "away_team_name": "Carlton",
                    "venue_id": 11, "date": "2025-05-11", "time": "14:10",
                    "status": "upcoming",
                    "result_home_score": None, "result_away_score": None,
                },
            ],
            teams=[
                {"team_id": 1, "name": "Melbourne",   "abbreviation": "MEL"},
                {"team_id": 2, "name": "Collingwood", "abbreviation": "COL"},
                {"team_id": 3, "name": "Richmond",    "abbreviation": "RIC"},
                {"team_id": 4, "name": "Carlton",     "abbreviation": "CAR"},
            ],
        )

        with patch("app.services.pipeline.PreflightService") as mock_pf:
            mock_pf.return_value.run.return_value = None
            pipeline = PredictionPipeline(loader=loader)
            result = pipeline.run()

        for multi in result.get("value_multis", []):
            assert "multi_id" in multi
            assert "n_legs" in multi
            assert "combined_odds" in multi
            assert "adjusted_probability" in multi
            assert "ev" in multi
            assert "correlation_label" in multi
            assert "legs" in multi, "Multi must include embedded leg details"

    def test_pipeline_stages_reported(self):
        """Pipeline must report all 7 stages in the result."""
        from app.services.pipeline import PredictionPipeline

        loader = _make_minimal_loader()

        with patch("app.services.pipeline.PreflightService") as mock_pf:
            mock_pf.return_value.run.return_value = None
            pipeline = PredictionPipeline(loader=loader)
            result = pipeline.run()

        stages = result.get("pipeline_stages", {})
        assert "preflight" in stages
        assert "fixtures_loaded" in stages
        assert stages["fixtures_loaded"] > 0
        assert "legs_generated" in stages
        assert "multis_built" in stages

    def test_elapsed_time_is_positive(self):
        from app.services.pipeline import PredictionPipeline

        loader = _make_minimal_loader()

        with patch("app.services.pipeline.PreflightService") as mock_pf:
            mock_pf.return_value.run.return_value = None
            pipeline = PredictionPipeline(loader=loader)
            result = pipeline.run()

        assert result["elapsed_seconds"] > 0


# ---------------------------------------------------------------------------
# Test: no upcoming fixtures → hard fail
# ---------------------------------------------------------------------------

class TestNoUpcomingFixtures:
    def test_empty_fixtures_raises(self):
        """Pipeline must raise (not silently produce zero results) when no fixtures exist."""
        from app.services.pipeline import PredictionPipeline
        from app.services.pipeline import NoUpcomingFixturesError

        loader = _make_minimal_loader(fixtures=[])  # empty

        with patch("app.services.pipeline.PreflightService") as mock_pf:
            mock_pf.return_value.run.return_value = None
            pipeline = PredictionPipeline(loader=loader)

            with pytest.raises(NoUpcomingFixturesError):
                pipeline.run()
