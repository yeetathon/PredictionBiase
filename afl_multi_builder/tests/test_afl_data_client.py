"""Tests for DSGClient (AFL Data Sports Group API client): caching, retry, auth."""
import json
import time
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# _FileCache tests
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
        # Force stale by writing old timestamp using the internal _path method
        p = cache._path(key)
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
# AFLDataLoader normalisation tests (uses DSG field names)
# ---------------------------------------------------------------------------

class TestAFLDataLoaderNormalise:
    """Unit tests for normalisation methods — no API calls."""

    def _make_loader(self, monkeypatch):
        monkeypatch.setenv("AFL_DATA_AUTHKEY", "test_key_xyz")
        monkeypatch.setenv("AFL_DATA_USERNAME", "testuser")
        from app.data_ingestion.afl_data_loader import AFLDataLoader
        loader = AFLDataLoader.__new__(AFLDataLoader)
        # Inject a minimal mock DSGClient so methods that call self._client work
        mock_client = MagicMock()
        mock_client.current_year.return_value = 2024
        loader._client = mock_client
        loader._competition_id = "1"
        loader._season_id = "2024"
        return loader

    def test_normalise_match_basic(self, monkeypatch):
        """DSG field names: id, round, hometeam/awayteam dicts, status."""
        loader = self._make_loader(monkeypatch)
        from app.data_ingestion.afl_data_loader import AFLDataLoader
        raw = {
            "id": "12345",
            "round": 5,
            "year": 2024,
            "hometeam": {"id": 10, "name": "Richmond Tigers"},
            "awayteam": {"id": 11, "name": "Collingwood"},
            "date": "2024-05-10",
            "time": "08:10:00",
            "status": "fixture",
        }
        result = AFLDataLoader._normalise_match(loader, raw)
        assert result is not None
        assert result["sport_event_id"] == "12345"
        assert result["round"] == 5
        assert result["season"] == 2024
        assert result["home_team"] == "Richmond Tigers"
        assert result["away_team"] == "Collingwood"
        assert result["status"] == "upcoming"

    def test_normalise_match_completed(self, monkeypatch):
        """Completed match with ft_score produces home_win and margin."""
        loader = self._make_loader(monkeypatch)
        from app.data_ingestion.afl_data_loader import AFLDataLoader
        raw = {
            "id": "99",
            "round": 3,
            "year": 2024,
            "hometeam": {"id": 1, "name": "Brisbane Lions"},
            "awayteam": {"id": 2, "name": "GWS Giants"},
            "status": "completed",
            "score": {"ft_score": "95-72"},
        }
        result = AFLDataLoader._normalise_match(loader, raw)
        assert result["status"] == "completed"
        assert result["home_score"] == 95
        assert result["away_score"] == 72
        assert result["home_win"] == 1
        assert result["margin"] == 23

    def test_normalise_match_no_id_returns_none(self, monkeypatch):
        """Match dict without an id field must return None."""
        loader = self._make_loader(monkeypatch)
        from app.data_ingestion.afl_data_loader import AFLDataLoader
        raw = {"round": 5}
        result = AFLDataLoader._normalise_match(loader, raw)
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
            "contested_possessions": 95,
            "uncontested_possessions": 105,
            "hitouts": 42,
            "frees_for": 12,
            "frees_against": 8,
            "effective_kicks": 90,
        }
        result = AFLDataLoader._extract_afl_stats(loader, stats)
        assert result["goals"] == 14.0
        assert result["kicks"] == 120.0
        assert result["tackles"] == 38.0
        assert result["clearances"] == 35.0

    def test_canonical_status_mapping(self, monkeypatch):
        from app.data_ingestion.afl_data_loader import _canonical
        assert _canonical("fixture") == "upcoming"
        assert _canonical("scheduled") == "upcoming"
        assert _canonical("completed") == "completed"
        assert _canonical("finished") == "completed"
        assert _canonical("inprogress") == "in_progress"
        assert _canonical("bye") == "bye"

    def test_extract_list_shapes(self, monkeypatch):
        loader = self._make_loader(monkeypatch)
        from app.data_ingestion.afl_data_loader import AFLDataLoader
        # Key "matches"
        assert AFLDataLoader._extract_list({"matches": [{"id": 1}]}, "matches") == [{"id": 1}]
        # Key "fixtures"
        assert AFLDataLoader._extract_list({"fixtures": [{"id": 2}]}, "fixtures") == [{"id": 2}]
        # Missing key returns empty list
        assert AFLDataLoader._extract_list({}, "matches") == []


# ---------------------------------------------------------------------------
# DSGClient mocked HTTP tests
# ---------------------------------------------------------------------------

class TestDSGClientMocked:
    """Tests that mock HTTP so no real API calls are made."""

    @pytest.fixture()
    def client(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AFL_DATA_AUTHKEY", "TEST_KEY_AFL")
        monkeypatch.setenv("AFL_DATA_USERNAME", "testuser")
        from app.data_ingestion.afl_data_client import DSGClient, _FileCache
        c = DSGClient.__new__(DSGClient)
        c._base = "https://dsg-api.com"
        c._client = "testuser"
        c._authkey = "TEST_KEY_AFL"
        c._cache = _FileCache(tmp_path / "cache")
        import requests
        c._session = requests.Session()
        return c

    def test_is_configured(self, client):
        assert client.is_configured() is True

    def test_not_configured_empty_key(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AFL_DATA_AUTHKEY", "")
        monkeypatch.setenv("AFL_DATA_USERNAME", "testuser")
        from app.data_ingestion.afl_data_client import DSGClient
        c = DSGClient.__new__(DSGClient)
        c._authkey = ""
        c._client = "testuser"
        assert c.is_configured() is False

    def test_cache_hit_skips_http(self, client):
        cached_data = {"teams": [{"id": 1, "name": "Richmond"}]}
        # Cache key excludes authkey — need params with authkey + ftype for key gen
        key = client._cache.make_key(
            "get_teams",
            {"client": "testuser", "authkey": "TEST_KEY_AFL", "ftype": "json",
             "competitionId": 1, "year": 2024},
        )
        client._cache.set(key, cached_data)
        with patch.object(client._session, "get") as mock_get:
            result = client.get(
                "get_teams",
                params={"competitionId": 1, "year": 2024},
                ttl=3600,
            )
            mock_get.assert_not_called()
        assert result["teams"][0]["name"] == "Richmond"

    def test_get_injects_authkey(self, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"matches": []}
        mock_resp.content = b'{"matches": []}'
        with patch.object(client._session, "get", return_value=mock_resp) as mock_get:
            client.get("get_matches", params={"year": 2024}, ttl=0)
            call_kwargs = mock_get.call_args
            params_used = (call_kwargs[1].get("params") or
                           (call_kwargs[0][1] if len(call_kwargs[0]) > 1 else {}))
            assert "authkey" in params_used
            assert params_used["authkey"] == "TEST_KEY_AFL"
