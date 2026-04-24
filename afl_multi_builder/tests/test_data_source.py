"""
Data source tests — live-only system.

Verifies:
  1. DataSourceManager requires AFL Data API key (no silent fallback)
  2. Provenance stamping (_source_type, _fetch_ts) works correctly
  3. DataSourceStatus reports correct live/cache modes
  4. config.Settings rejects 'demo' as a data_mode
  5. No DemoDataProvider or load_all_demo() exists anywhere
  6. AFLDataProvider stamps DataFrames with correct source types
"""
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Test: config enforces live-only mode
# ---------------------------------------------------------------------------

class TestConfigLiveOnly:
    def test_data_mode_default_is_live(self):
        from app.core.config import settings
        assert settings.data_mode in ("live", "cache"), \
            f"Default data_mode must be 'live' or 'cache', got {settings.data_mode!r}"

    def test_demo_mode_rejected(self):
        """data_mode='demo' must be rejected by pydantic validation."""
        from app.core.config import Settings
        import pydantic

        with pytest.raises((pydantic.ValidationError, ValueError)):
            Settings(data_mode="demo")

    def test_enable_demo_fallback_absent(self):
        """enable_demo_fallback must not exist in Settings."""
        from app.core.config import settings
        assert not hasattr(settings, "enable_demo_fallback"), \
            "enable_demo_fallback field must be removed from Settings"

    def test_demo_data_dir_absent(self):
        """demo_data_dir must not exist in Settings."""
        from app.core.config import settings
        assert not hasattr(settings, "demo_data_dir"), \
            "demo_data_dir field must be removed from Settings"

    def test_effective_data_mode_live_or_cache(self, monkeypatch):
        """effective_data_mode must return 'live' or 'cache'."""
        from app.core import config

        for mode in ("live", "cache"):
            monkeypatch.setattr(config.settings, "data_mode", mode)
            assert config.settings.effective_data_mode == mode

    def test_is_afl_data_configured_false_without_key(self, monkeypatch):
        from app.core import config
        monkeypatch.setattr(config.settings, "afl_data_authkey", "")
        assert config.settings.is_afl_data_configured is False

    def test_is_afl_data_configured_true_with_key(self, monkeypatch):
        from app.core import config
        monkeypatch.setattr(config.settings, "afl_data_authkey", "abc123xyz")
        assert config.settings.is_afl_data_configured is True

    def test_is_odds_api_configured_false_without_key(self, monkeypatch):
        from app.core import config
        monkeypatch.setattr(config.settings, "odds_api_key", "")
        assert config.settings.is_odds_api_configured is False

    def test_no_sportradar_settings(self):
        """Sportradar settings must be completely removed from config."""
        from app.core.config import settings
        assert not hasattr(settings, "sportradar_api_key"), \
            "sportradar_api_key must be removed"
        assert not hasattr(settings, "sportradar_base_url"), \
            "sportradar_base_url must be removed"

    def test_no_api_sports_settings(self):
        """API-Sports settings must be completely removed from config."""
        from app.core.config import settings
        assert not hasattr(settings, "api_sports_key"), \
            "api_sports_key must be removed"


# ---------------------------------------------------------------------------
# Test: provenance stamping
# ---------------------------------------------------------------------------

class TestProvenanceStamping:
    def test_stamp_adds_source_type_and_fetch_ts(self):
        from app.data_ingestion.data_source_manager import _stamp, SOURCE_API

        df = pd.DataFrame({"a": [1, 2, 3]})
        stamped = _stamp(df, SOURCE_API)

        assert "_source_type" in stamped.columns
        assert "_fetch_ts" in stamped.columns
        assert (stamped["_source_type"] == SOURCE_API).all()

    def test_stamp_does_not_mutate_original(self):
        from app.data_ingestion.data_source_manager import _stamp, SOURCE_CACHE

        df = pd.DataFrame({"x": [10]})
        _stamp(df, SOURCE_CACHE)
        assert "_source_type" not in df.columns

    def test_source_constants_are_distinct(self):
        from app.data_ingestion.data_source_manager import (
            SOURCE_API, SOURCE_CACHE, SOURCE_DERIVED
        )
        sources = {SOURCE_API, SOURCE_CACHE, SOURCE_DERIVED}
        assert len(sources) == 3, "All source constants must be unique"

    def test_no_source_demo_constant(self):
        """There must be no SOURCE_DEMO constant in the data source manager."""
        import app.data_ingestion.data_source_manager as dsm
        assert not hasattr(dsm, "SOURCE_DEMO"), \
            "SOURCE_DEMO must not exist — demo mode is removed"

    def test_source_api_references_afl_data(self):
        """SOURCE_API must reference AFL Data, not Sportradar."""
        from app.data_ingestion.data_source_manager import SOURCE_API
        assert "afl" in SOURCE_API.lower(), \
            f"SOURCE_API should reference AFL Data, got: {SOURCE_API!r}"


# ---------------------------------------------------------------------------
# Test: DataSourceManager requires AFL Data API key
# ---------------------------------------------------------------------------

class TestDataSourceManagerRequiresKey:
    def test_raises_without_afl_data_key(self, monkeypatch):
        """DataSourceManager must raise RuntimeError if no AFL Data key is configured."""
        from app.data_ingestion import data_source_manager
        from app.core import config

        monkeypatch.setattr(config.settings, "afl_data_authkey", "")

        with pytest.raises(RuntimeError, match="AFL_DATA_AUTHKEY"):
            data_source_manager.DataSourceManager()

    def test_no_demo_provider_in_manager(self):
        """DataSourceManager must not have a _demo attribute or load_all_demo method."""
        import inspect
        from app.data_ingestion import data_source_manager

        src = inspect.getsource(data_source_manager)
        assert "load_all_demo" not in src, "load_all_demo() must be removed"
        assert "DemoDataProvider" not in src, "DemoDataProvider must be removed"
        assert "enable_demo_fallback" not in src, "enable_demo_fallback references must be removed"

    def test_manager_status_has_no_demo_available(self, monkeypatch):
        """DataSourceStatus must not have a demo_available field."""
        from app.data_ingestion.data_source_manager import DataSourceStatus
        import dataclasses

        field_names = {f.name for f in dataclasses.fields(DataSourceStatus)}
        assert "demo_available" not in field_names, \
            "DataSourceStatus must not expose demo_available"

    def test_manager_status_has_afl_data_fields(self):
        """DataSourceStatus must have AFL Data specific fields."""
        from app.data_ingestion.data_source_manager import DataSourceStatus
        import dataclasses

        field_names = {f.name for f in dataclasses.fields(DataSourceStatus)}
        assert "afl_data_configured" in field_names, \
            "DataSourceStatus must have afl_data_configured field"


# ---------------------------------------------------------------------------
# Test: demo_loader raises on import
# ---------------------------------------------------------------------------

class TestDemoLoaderRemoved:
    def test_demo_loader_import_raises_immediately(self):
        """Any attempt to import demo_loader must raise ImportError."""
        import sys
        sys.modules.pop("app.data_ingestion.demo_loader", None)

        with pytest.raises(ImportError, match="demo_loader is no longer available"):
            import app.data_ingestion.demo_loader

    def test_demo_loader_error_includes_setup_instructions(self):
        """The ImportError must tell the user how to configure real API keys."""
        import sys
        sys.modules.pop("app.data_ingestion.demo_loader", None)

        try:
            import app.data_ingestion.demo_loader
        except ImportError as e:
            assert "AFL_DATA_AUTHKEY" in str(e), \
                "ImportError must mention AFL_DATA_AUTHKEY"
        else:
            pytest.fail("ImportError was not raised")


# ---------------------------------------------------------------------------
# Test: DataLoader refuses demo mode
# ---------------------------------------------------------------------------

class TestDataLoaderLiveOnly:
    def test_make_afl_provider_raises_without_key(self, monkeypatch):
        """_make_afl_provider must raise RuntimeError if AFL Data key is absent."""
        from app.core import config

        monkeypatch.setattr(config.settings, "afl_data_authkey", "")

        from app.data_ingestion.loader import _make_afl_provider
        with pytest.raises(RuntimeError):
            _make_afl_provider()

    def test_loader_get_source_status_returns_dict(self, monkeypatch):
        """DataLoader.get_source_status() must return a dict with known keys."""
        from app.data_ingestion.loader import DataLoader
        import pandas as pd

        class _FakeAFL:
            @property
            def fixtures_df(self): return pd.DataFrame()
            @property
            def teams_df(self): return pd.DataFrame()
            @property
            def players_df(self): return pd.DataFrame()
            @property
            def team_stats_df(self): return pd.DataFrame()
            @property
            def player_stats_df(self): return pd.DataFrame()

        class _FakeOdds:
            @property
            def odds_df(self): return pd.DataFrame()

        loader = DataLoader(afl_provider=_FakeAFL(), odds_provider=_FakeOdds())
        status = loader.get_source_status()

        assert isinstance(status, dict)
        assert "afl_data" in status
        assert "odds_api" in status

    def test_null_odds_provider_returns_empty_df(self):
        """_NullOddsProvider must return an empty DataFrame for odds_df."""
        from app.data_ingestion.loader import _NullOddsProvider

        provider = _NullOddsProvider()
        df = provider.odds_df
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_null_weather_provider_returns_none(self):
        from app.data_ingestion.loader import _NullWeatherProvider

        provider = _NullWeatherProvider()
        assert provider.get_weather(fixture_id=999) is None
