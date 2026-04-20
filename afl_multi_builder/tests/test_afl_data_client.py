"""Tests for AFLDataClient: caching, retry, authentication."""
import json
import time
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# AFLDataClient file cache tests
# ---------------------------------------------------------------------------

class TestFileCache:
    def _make_cache(self, tmp_path):
        from app.data_ingestion.afl_data_client import _FileCache
        return _FileCache(tmp_path / "cache")

    def test_miss_returns_none(self, tmp_path):
        cache = self._make_cache(tmp_path)
        assert cache.get("nonexistent", ttl_seconds=3600) is None

    def test_store_and_retrieve(self, tmp_path):
        cache = self._make_cache(tmp_path)
        data = {"teams": [{"teamId": 1, "name": "Richmond"}]}
        key = cache.make_key("cfs/afl/teamList", {"year": 2024})
        cache.set(key, data)
        result = cache.get(key, ttl_seconds=3600)
        assert result is not None
        assert result["teams"][0]["name"] == "Richmond"

    def test_stale_entry_returns_none(self, tmp_path):
        cache = self._make_cache(tmp_path)
        key = cache.make_key("ep", {})
        cache.set(key, {"x": 1})
        # Force stale by writing old timestamp
        p = cache._key_path(key)
        raw = json.loads(p.read_text())
        raw["_ts"] = time.time() - 7201  # 2 hours old
        p.write_text(json.dumps(raw))
        assert cache.get(key, ttl_seconds=3600) is None

    def test_make_key_deterministic(self, tmp_path):
        cache = self._make_cache(tmp_path)
        k1 = cache.make_key("ep", {"a": 1, "b": 2})
        k2 = cache.make_key("ep", {"b": 2, "a": 1})
        assert k1 == k2

    def test_make_key_different_endpoints(self, tmp_path):
        cache = self._make_cache(tmp_path)
        k1 = cache.make_key("fixtures", {"year": 2024})
        k2 = cache.make_key("players", {"year": 2024})
        assert k1 != k2


# ---------------------------------------------------------------------------
# AFLDataLoader normalisation tests
# ---------------------------------------------------------------------------

class TestAFLDataLoaderNormalise:
    """Unit tests for normalisation methods — no API calls."""

    def _make_loader(self, monkeypatch):
        monkeypatch.setenv("AFL_DATA_AUTHKEY", "test_key_xyz")
        # Prevent real HTTP calls by patching the client's get method
        from app.data_ingestion.afl_data_loader import AFLDataLoader
        loader = AFLDataLoader.__new__(AFLDataLoader)
        loader._year = 2024
        return loader

    def test_normalise_fixture_basic(self, monkeypatch):
        loader = self._make_loader(monkeypatch)
        from app.data_ingestion.afl_data_loader import AFLDataLoader
        raw = {
            "matchId": "12345",
            "roundNumber": 5,
            "year": 2024,
            "homeTeam": {"teamId": 10, "name": "Richmond Tigers"},
            "awayTeam": {"teamId": 11, "name": "Collingwood"},
            "utcStartTime": "2024-05-10T08:10:00Z",
            "status": "SCHEDULED",
        }
        result = AFLDataLoader._normalise_fixture(loader, raw)
        assert result is not None
        assert result["sport_event_id"] == "12345"
        assert result["round"] == 5
        assert result["season"] == 2024
        assert result["home_team"] == "Richmond Tigers"
        assert result["away_team"] == "Collingwood"
        assert result["status"] == "upcoming"

    def test_normalise_fixture_completed(self, monkeypatch):
        loader = self._make_loader(monkeypatch)
        from app.data_ingestion.afl_data_loader import AFLDataLoader
        raw = {
            "matchId": "99",
            "roundNumber": 3,
            "year": 2024,
            "homeTeam": {"teamId": 1, "name": "Brisbane Lions"},
            "awayTeam": {"teamId": 2, "name": "GWS Giants"},
            "status": "COMPLETED",
            "homeTeamScore": {"totalScore": 95, "goals": 14, "behinds": 11},
            "awayTeamScore": {"totalScore": 72, "goals": 10, "behinds": 12},
        }
        result = AFLDataLoader._normalise_fixture(loader, raw)
        assert result["status"] == "completed"
        assert result["home_score"] == 95
        assert result["away_score"] == 72
        assert result["home_win"] == 1
        assert result["margin"] == 23

    def test_normalise_fixture_no_match_id_returns_none(self, monkeypatch):
        loader = self._make_loader(monkeypatch)
        from app.data_ingestion.afl_data_loader import AFLDataLoader
        raw = {"roundNumber": 5}
        result = AFLDataLoader._normalise_fixture(loader, raw)
        assert result is None

    def test_extract_afl_stats_full(self, monkeypatch):
        loader = self._make_loader(monkeypatch)
        from app.data_ingestion.afl_data_loader import AFLDataLoader
        stats = {
            "goals": 14,
            "behinds": 5,
            "kicks": 120,
            "handballs": 80,
            "disposals": 200,
            "marks": 45,
            "tackles": 38,
            "inside50s": 60,
            "clearances": 35,
            "contestedPossessions": 95,
            "uncontestedPossessions": 105,
            "hitouts": 42,
            "freesFor": 12,
            "freesAgainst": 8,
            "effectiveKicks": 90,
        }
        result = AFLDataLoader._extract_afl_stats(loader, stats)
        assert result["goals"] == 14.0
        assert result["kicks"] == 120.0
        assert result["tackles"] == 38.0
        assert result["clearances"] == 35.0
        assert result["contested_possessions"] == 95.0
        assert result["effective_kicks"] == 90.0

    def test_canonical_status_mapping(self, monkeypatch):
        from app.data_ingestion.afl_data_loader import _canonical_status
        assert _canonical_status("SCHEDULED") == "upcoming"
        assert _canonical_status("COMPLETED") == "completed"
        assert _canonical_status("IN_PROGRESS") == "in_progress"
        assert _canonical_status("BYE") == "bye"
        assert _canonical_status("completed") == "completed"

    def test_extract_fixture_list_shapes(self, monkeypatch):
        loader = self._make_loader(monkeypatch)
        from app.data_ingestion.afl_data_loader import AFLDataLoader

        # Shape 1: {"fixtures": [...]}
        assert AFLDataLoader._extract_fixture_list(loader, {"fixtures": [{"id": 1}]}) == [{"id": 1}]
        # Shape 2: {"matches": [...]}
        assert AFLDataLoader._extract_fixture_list(loader, {"matches": [{"id": 2}]}) == [{"id": 2}]
        # Shape 3: empty
        assert AFLDataLoader._extract_fixture_list(loader, {}) == []


# ---------------------------------------------------------------------------
# AFLDataClient mocked HTTP tests
# ---------------------------------------------------------------------------

class TestAFLDataClientMocked:
    """Tests that mock HTTP so no real API calls are made."""

    @pytest.fixture()
    def client(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AFL_DATA_AUTHKEY", "TEST_KEY_AFL")
        from app.data_ingestion.afl_data_client import AFLDataClient, _FileCache
        c = AFLDataClient.__new__(AFLDataClient)
        c._base_url = "https://api.afl.com.au"
        c._authkey = "TEST_KEY_AFL"
        c._username = "testuser"
        c._password = "testpass"
        c._competition_id = 1
        c._cache = _FileCache(tmp_path / "cache")
        import requests
        c._session = requests.Session()
        return c

    def test_is_configured(self, client):
        assert client.is_configured() is True

    def test_not_configured_empty_key(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AFL_DATA_AUTHKEY", "")
        from app.data_ingestion.afl_data_client import AFLDataClient
        c = AFLDataClient.__new__(AFLDataClient)
        c._authkey = ""
        assert c.is_configured() is False

    def test_cache_hit_skips_http(self, client, tmp_path):
        cached_data = {"teams": [{"teamId": 1, "name": "Richmond"}]}
        key = client._cache.make_key("cfs/afl/teamList", {"authkey": "TEST_KEY_AFL",
                                                           "competitionId": 1, "year": 2024})
        client._cache.set(key, cached_data)
        with patch.object(client._session, "get") as mock_get:
            result = client.get("cfs/afl/teamList",
                                params={"competitionId": 1, "year": 2024},
                                cache_ttl_seconds=3600)
            mock_get.assert_not_called()
        assert result["teams"][0]["name"] == "Richmond"

    def test_get_injects_authkey(self, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"fixtures": []}
        mock_resp.content = b'{"fixtures": []}'
        with patch.object(client._session, "get", return_value=mock_resp) as mock_get:
            client.get("cfs/afl/fixtureList", params={"year": 2024},
                       cache_ttl_seconds=0)
            call_kwargs = mock_get.call_args
            params_used = call_kwargs[1].get("params") or call_kwargs[0][1] if len(call_kwargs[0]) > 1 else {}
            if not params_used and call_kwargs[1]:
                params_used = call_kwargs[1].get("params", {})
            assert "authkey" in params_used
            assert params_used["authkey"] == "TEST_KEY_AFL"
