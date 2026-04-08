"""Tests for DataSourceManager: provenance labeling, mode switching."""
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestDemoDataProvider:
    def test_loads_fixtures(self):
        from app.data_ingestion.data_source_manager import DemoDataProvider
        provider = DemoDataProvider()
        if not provider.is_available():
            pytest.skip("Demo data not available")
        df = provider.get_fixtures()
        assert not df.empty
        assert "_source_type" in df.columns
        assert (df["_source_type"] == "demo").all()

    def test_loads_teams(self):
        from app.data_ingestion.data_source_manager import DemoDataProvider
        provider = DemoDataProvider()
        if not provider.is_available():
            pytest.skip("Demo data not available")
        df = provider.get_teams()
        assert not df.empty

    def test_loads_players(self):
        from app.data_ingestion.data_source_manager import DemoDataProvider
        provider = DemoDataProvider()
        if not provider.is_available():
            pytest.skip("Demo data not available")
        df = provider.get_players()
        assert not df.empty

    def test_loads_player_stats(self):
        from app.data_ingestion.data_source_manager import DemoDataProvider
        provider = DemoDataProvider()
        if not provider.is_available():
            pytest.skip("Demo data not available")
        df = provider.get_player_stats()
        assert not df.empty

    def test_missing_file_returns_empty(self, tmp_path, monkeypatch):
        from app.data_ingestion.data_source_manager import DemoDataProvider
        from app.core import config
        # Point to empty tmp_path
        with patch.object(config.settings, "demo_data_dir", tmp_path):
            provider = DemoDataProvider.__new__(DemoDataProvider)
            provider._base = tmp_path
            df = provider.get_fixtures()
            assert df.empty


class TestProvenanceStamping:
    def test_stamp_adds_columns(self):
        from app.data_ingestion.data_source_manager import _stamp, SOURCE_DEMO
        df = pd.DataFrame({"a": [1, 2, 3]})
        stamped = _stamp(df, SOURCE_DEMO)
        assert "_source_type" in stamped.columns
        assert "_fetch_ts" in stamped.columns
        assert (stamped["_source_type"] == SOURCE_DEMO).all()

    def test_stamp_does_not_mutate_original(self):
        from app.data_ingestion.data_source_manager import _stamp, SOURCE_DEMO
        df = pd.DataFrame({"x": [10]})
        _stamp(df, SOURCE_DEMO)
        assert "_source_type" not in df.columns

    def test_source_constants_distinct(self):
        from app.data_ingestion.data_source_manager import (
            SOURCE_API, SOURCE_CACHE, SOURCE_DEMO, SOURCE_DERIVED
        )
        sources = {SOURCE_API, SOURCE_CACHE, SOURCE_DEMO, SOURCE_DERIVED}
        assert len(sources) == 4


class TestDataSourceManagerStatus:
    def test_status_has_required_keys(self):
        from app.data_ingestion.data_source_manager import DataSourceManager
        mgr = DataSourceManager()
        status = mgr.get_status()
        assert hasattr(status, "mode")
        assert hasattr(status, "effective_mode")
        assert hasattr(status, "demo_available")

    def test_demo_mode_returns_demo_data(self, monkeypatch):
        from app.data_ingestion.data_source_manager import DataSourceManager
        from app.core import config
        # Ensure we're in demo mode
        monkeypatch.setattr(config.settings, "data_mode", "demo")
        monkeypatch.setattr(config.settings, "sportradar_api_key", "")
        mgr = DataSourceManager()
        df = mgr.get_teams()
        # Should get something from demo if available
        if not df.empty:
            assert "_source_type" in df.columns

    def test_load_all_demo_returns_dict(self):
        from app.data_ingestion.data_source_manager import DataSourceManager
        mgr = DataSourceManager()
        data = mgr.load_all_demo()
        assert isinstance(data, dict)
        expected_keys = {"fixtures", "teams", "players", "player_stats", "team_stats", "odds", "venues"}
        assert expected_keys == set(data.keys())

    def test_all_demo_dataframes_have_provenance(self):
        from app.data_ingestion.data_source_manager import DataSourceManager
        mgr = DataSourceManager()
        data = mgr.load_all_demo()
        for name, df in data.items():
            if not df.empty:
                assert "_source_type" in df.columns, f"{name} missing _source_type"
                assert "_fetch_ts" in df.columns, f"{name} missing _fetch_ts"


class TestEffectiveDataMode:
    def test_fallback_to_demo_without_api_key(self, monkeypatch):
        from app.core import config
        monkeypatch.setattr(config.settings, "data_mode", "live")
        monkeypatch.setattr(config.settings, "sportradar_api_key", "")
        monkeypatch.setattr(config.settings, "enable_demo_fallback", True)
        assert config.settings.effective_data_mode == "demo"

    def test_live_mode_with_api_key(self, monkeypatch):
        from app.core import config
        monkeypatch.setattr(config.settings, "data_mode", "live")
        monkeypatch.setattr(config.settings, "sportradar_api_key", "some_key")
        assert config.settings.effective_data_mode == "live"

    def test_demo_mode_stays_demo(self, monkeypatch):
        from app.core import config
        monkeypatch.setattr(config.settings, "data_mode", "demo")
        assert config.settings.effective_data_mode == "demo"
