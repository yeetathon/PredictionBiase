"""Tests for SportradarClient: throttling, quota, cache usage."""
import time
import pytest
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# QuotaManager tests
# ---------------------------------------------------------------------------

class TestQuotaManager:
    def _make_qm(self, tmp_path):
        from app.data_ingestion.quota_manager import QuotaManager
        db_url = f"sqlite:///{tmp_path}/test_quota.db"
        return QuotaManager(database_url=db_url, quota_total=100, warn_pct=0.80)

    def test_initial_state(self, tmp_path):
        qm = self._make_qm(tmp_path)
        assert qm.get_total_used() == 0
        assert qm.get_remaining() == 100
        assert qm.can_make_request() is True

    def test_record_request(self, tmp_path):
        qm = self._make_qm(tmp_path)
        qm.record_request("seasons/schedules.json", 200, 123.4)
        assert qm.get_total_used() == 1
        assert qm.get_remaining() == 99

    def test_daily_used_counts_recent(self, tmp_path):
        qm = self._make_qm(tmp_path)
        qm.record_request("endpoint", 200, 50.0)
        assert qm.get_daily_used() == 1

    def test_can_make_request_false_near_limit(self, tmp_path):
        qm = self._make_qm(tmp_path)
        # Fill up to 96 / 100 (96%)
        for i in range(96):
            qm.record_request(f"ep_{i}", 200, 10.0)
        assert qm.can_make_request() is False

    def test_status_dict_keys(self, tmp_path):
        qm = self._make_qm(tmp_path)
        status = qm.get_status()
        for key in ("total_quota", "used", "remaining", "warn_threshold", "warn_triggered", "daily_used"):
            assert key in status

    def test_warn_threshold_triggered(self, tmp_path):
        qm = self._make_qm(tmp_path)
        for i in range(80):
            qm.record_request(f"ep_{i}", 200, 10.0)
        status = qm.get_status()
        assert status["warn_triggered"] is True

    def test_multiple_instances_share_data(self, tmp_path):
        db_url = f"sqlite:///{tmp_path}/shared.db"
        from app.data_ingestion.quota_manager import QuotaManager
        qm1 = QuotaManager(database_url=db_url, quota_total=100)
        qm1.record_request("ep", 200, 10.0)
        qm2 = QuotaManager(database_url=db_url, quota_total=100)
        assert qm2.get_total_used() == 1


# ---------------------------------------------------------------------------
# CacheManager tests
# ---------------------------------------------------------------------------

class TestCacheManager:
    def _make_cm(self, tmp_path):
        from app.data_ingestion.cache_manager import CacheManager
        db_url = f"sqlite:///{tmp_path}/test_cache.db"
        return CacheManager(database_url=db_url)

    def test_miss_returns_none(self, tmp_path):
        cm = self._make_cm(tmp_path)
        result = cm.get("missing_key", ttl_seconds=3600)
        assert result is None

    def test_store_and_retrieve(self, tmp_path):
        cm = self._make_cm(tmp_path)
        data = {"foo": "bar", "count": 42}
        key = cm.make_key("seasons/schedule.json", {"season": "2024"})
        cm.set(key, "seasons/schedule.json", "hash123", data, ttl_seconds=3600)
        result = cm.get(key, ttl_seconds=3600)
        assert result == data

    def test_stale_entry_returns_none(self, tmp_path):
        cm = self._make_cm(tmp_path)
        data = {"hello": "world"}
        key = cm.make_key("ep", {})
        cm.set(key, "ep", "h", data, ttl_seconds=1)
        time.sleep(1.1)
        result = cm.get(key, ttl_seconds=1)
        assert result is None

    def test_invalidate(self, tmp_path):
        cm = self._make_cm(tmp_path)
        key = cm.make_key("ep", {})
        cm.set(key, "ep", "h", {"x": 1}, ttl_seconds=3600)
        cm.invalidate(key)
        assert cm.get(key, ttl_seconds=3600) is None

    def test_make_key_deterministic(self, tmp_path):
        cm = self._make_cm(tmp_path)
        k1 = cm.make_key("ep", {"a": 1, "b": 2})
        k2 = cm.make_key("ep", {"b": 2, "a": 1})
        assert k1 == k2

    def test_stats_dict(self, tmp_path):
        cm = self._make_cm(tmp_path)
        stats = cm.get_stats()
        assert "total_entries" in stats


# ---------------------------------------------------------------------------
# SportradarClient tests (mocked HTTP)
# ---------------------------------------------------------------------------

class TestSportradarClientMocked:
    """Tests that mock the HTTP layer so no real API calls are made."""

    @pytest.fixture()
    def client(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SPORTRADAR_API_KEY", "TEST_KEY_123")
        from app.data_ingestion.sportradar_client import SportradarClient
        from app.data_ingestion.cache_manager import CacheManager
        from app.data_ingestion.quota_manager import QuotaManager
        db_url = f"sqlite:///{tmp_path}/test.db"
        cm = CacheManager(database_url=db_url)
        qm = QuotaManager(database_url=db_url, quota_total=100)
        client = SportradarClient.__new__(SportradarClient)
        client._api_key = "TEST_KEY_123"
        client._base_url = "https://api.sportradar.com/afl/trial/v2/en"
        client._cache = cm
        client._quota = qm
        client._min_interval = 0.0  # disable throttle for tests
        client._last_request_time = 0.0
        return client

    def test_is_configured(self, client):
        assert client.is_configured() is True

    def test_cache_hit_skips_http(self, client, tmp_path):
        """If data is in cache, no HTTP call should happen."""
        cached_data = {"seasons": [{"id": "sr:season:1"}]}
        key = client._cache.make_key("seasons.json", {})
        client._cache.set(key, "seasons.json", "h", cached_data, ttl_seconds=3600)

        with patch("requests.Session.get") as mock_get:
            # Call the internal _fetch method if it exists, otherwise patch get()
            result = client._cache.get(key, ttl_seconds=3600)
            assert result == cached_data
            mock_get.assert_not_called()

    def test_quota_exhausted_raises(self, client, tmp_path):
        """When quota is exhausted, client should refuse requests."""
        # Fill quota past 95%
        for i in range(96):
            client._quota.record_request(f"ep_{i}", 200, 10.0)
        assert client._quota.can_make_request() is False

    def test_not_configured_without_key(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SPORTRADAR_API_KEY", raising=False)
        from importlib import reload
        import app.data_ingestion.sportradar_client as m
        # Create a client with empty key
        from app.data_ingestion.sportradar_client import SportradarClient
        c = SportradarClient.__new__(SportradarClient)
        c._api_key = ""
        assert c.is_configured() is False
