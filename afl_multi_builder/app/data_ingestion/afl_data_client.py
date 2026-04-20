"""
Data Sports Group (DSG) API client for AFL data.

Endpoint pattern:
    https://dsg-api.com/clients/{client}/australian_football/{endpoint}
    ?client={client}&authkey={authkey}&ftype=json

Authentication: client + authkey as query parameters on every request.
No quota limits — unlimited call rate.
Responses cached to file to avoid redundant calls.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from loguru import logger

from app.core.config import settings


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class DSGAPIError(Exception):
    def __init__(self, status_code: int, endpoint: str, message: str):
        self.status_code = status_code
        self.endpoint = endpoint
        self.message = message
        super().__init__(f"DSG API [{status_code}] {endpoint}: {message}")


# ---------------------------------------------------------------------------
# File-based JSON cache (TTL-aware)
# ---------------------------------------------------------------------------

class _FileCache:
    def __init__(self, cache_dir: Path):
        self._dir = cache_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self._dir / f"dsg_{key}.json"

    def get(self, key: str, ttl_seconds: int) -> Optional[dict]:
        p = self._path(key)
        if not p.exists():
            return None
        try:
            raw = json.loads(p.read_text())
            if time.time() - raw.get("_ts", 0) > ttl_seconds:
                return None
            return raw.get("data")
        except Exception:
            return None

    def set(self, key: str, data: dict) -> None:
        try:
            self._path(key).write_text(
                json.dumps({"_ts": time.time(), "data": data}, default=str)
            )
        except Exception as exc:
            logger.debug("DSG cache write failed: {}", exc)

    @staticmethod
    def make_key(endpoint: str, params: dict) -> str:
        raw = endpoint + json.dumps(
            {k: v for k, v in sorted(params.items()) if k not in ("authkey", "client")},
            default=str
        )
        return hashlib.sha1(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# DSG API Client
# ---------------------------------------------------------------------------

class DSGClient:
    """
    HTTP client for the Data Sports Group (DSG) API — Australian Football.

    Every request is authenticated with `client` + `authkey` query parameters
    and requests JSON responses via `ftype=json`.

    AFL Advanced Pack endpoints used:
        get_competitions  — league list
        get_seasons       — season IDs for a competition
        get_rounds        — rounds in a season
        get_matches       — fixtures + results (all rounds)
        get_matches_day   — fixtures for a specific date
        get_matches_updates — live in-progress match updates
        get_teams         — team master list
        get_squad         — team roster / squad
        get_people        — player profiles
        get_tables        — ladder / standings
        get_head2head     — head-to-head historical record
    """

    _SPORT = "australian_football"
    _MAX_RETRIES = 4
    _BACKOFF = 2.0

    def __init__(self):
        self._base = settings.afl_data_base_url.rstrip("/")
        self._client = settings.afl_data_username      # DSG "client" = username
        self._authkey = settings.afl_data_authkey
        self._cache = _FileCache(settings.raw_cache_dir / "dsg")
        self._session = requests.Session()
        self._session.headers.update({
            "Accept": "application/json",
            "User-Agent": "AFL-Multi-Builder/2.0",
        })

    def is_configured(self) -> bool:
        return bool(self._authkey and self._authkey.strip()
                    and self._client and self._client.strip())

    # ------------------------------------------------------------------
    # Core HTTP
    # ------------------------------------------------------------------

    def get(self, endpoint: str, params: Optional[Dict] = None,
            ttl: int = 3600) -> dict:
        """
        GET /clients/{client}/australian_football/{endpoint}
        with auth + json params injected automatically.
        """
        params = dict(params or {})
        params.update({
            "client": self._client,
            "authkey": self._authkey,
            "ftype": "json",
        })

        cache_key = _FileCache.make_key(endpoint, params)
        cached = self._cache.get(cache_key, ttl)
        if cached is not None:
            logger.debug("DSG cache hit: {}", endpoint)
            cached["_source"] = "cache"
            return cached

        url = f"{self._base}/clients/{self._client}/{self._SPORT}/{endpoint}"
        last_exc: Optional[Exception] = None

        for attempt in range(self._MAX_RETRIES):
            try:
                resp = self._session.get(url, params=params, timeout=30)

                if resp.status_code in (429, 500, 502, 503, 504):
                    wait = self._BACKOFF ** attempt
                    logger.warning("DSG {} → {} attempt {}/{}, retry in {:.1f}s",
                                   endpoint, resp.status_code, attempt + 1,
                                   self._MAX_RETRIES, wait)
                    time.sleep(wait)
                    continue

                if resp.status_code == 401:
                    raise DSGAPIError(401, endpoint,
                                      "Unauthorised — check AFL_DATA_AUTHKEY and AFL_DATA_USERNAME")
                if resp.status_code == 403:
                    raise DSGAPIError(403, endpoint,
                                      "Forbidden — check client name, authkey, and API plan permissions")
                if resp.status_code == 404:
                    raise DSGAPIError(404, endpoint, "Endpoint not found")

                resp.raise_for_status()

                # DSG may return XML even when ftype=json is set on some endpoints —
                # detect and handle gracefully.
                content_type = resp.headers.get("Content-Type", "")
                if "xml" in content_type and "json" not in content_type:
                    logger.warning("DSG {} returned XML instead of JSON — "
                                   "endpoint may not support JSON format", endpoint)
                    data = {"_xml_response": resp.text, "_source": "api_live"}
                else:
                    data = resp.json()
                    data["_source"] = "api_live"
                    data["_fetched_at"] = datetime.utcnow().isoformat()

                self._cache.set(cache_key, data)
                logger.debug("DSG live: {} → {} bytes", endpoint, len(resp.content))
                return data

            except DSGAPIError:
                raise
            except requests.RequestException as exc:
                last_exc = exc
                wait = self._BACKOFF ** attempt
                logger.warning("DSG network error {} attempt {}/{}: {} — retry {:.1f}s",
                               endpoint, attempt + 1, self._MAX_RETRIES, exc, wait)
                time.sleep(wait)

        raise DSGAPIError(0, endpoint,
                          f"All {self._MAX_RETRIES} attempts failed: {last_exc}")

    # ------------------------------------------------------------------
    # Competition / season discovery
    # ------------------------------------------------------------------

    def get_competitions(self, ttl: int = 86400 * 7) -> dict:
        """List all competitions (used to find AFL competition ID)."""
        return self.get("get_competitions", ttl=ttl)

    def get_seasons(self, competition_id: str, ttl: int = 86400) -> dict:
        """List seasons for a competition."""
        return self.get("get_seasons",
                        params={"competition_id": competition_id}, ttl=ttl)

    def get_rounds(self, season_id: str, ttl: int = 3600) -> dict:
        """List rounds in a season."""
        return self.get("get_rounds",
                        params={"season_id": season_id}, ttl=ttl)

    # ------------------------------------------------------------------
    # Matches / fixtures
    # ------------------------------------------------------------------

    def get_matches(self, season_id: str, ttl: int = 1800) -> dict:
        """All fixtures + results for a season."""
        return self.get("get_matches",
                        params={"season_id": season_id}, ttl=ttl)

    def get_matches_by_round(self, season_id: str, round_id: str,
                             ttl: int = 1800) -> dict:
        """Fixtures for a specific round."""
        return self.get("get_matches",
                        params={"season_id": season_id, "round_id": round_id},
                        ttl=ttl)

    def get_matches_day(self, date: str, ttl: int = 900) -> dict:
        """Fixtures for a specific date (YYYY-MM-DD)."""
        return self.get("get_matches_day", params={"date": date}, ttl=ttl)

    def get_matches_updates(self, ttl: int = 60) -> dict:
        """Live in-progress match updates (very short TTL)."""
        return self.get("get_matches_updates", ttl=ttl)

    def get_match(self, match_id: str, ttl: int = 86400) -> dict:
        """Full details for a single match (box score, events)."""
        return self.get("get_match", params={"match_id": match_id}, ttl=ttl)

    # ------------------------------------------------------------------
    # Teams / players
    # ------------------------------------------------------------------

    def get_teams(self, season_id: str, ttl: int = 86400 * 7) -> dict:
        """Team master list for a season."""
        return self.get("get_teams", params={"season_id": season_id}, ttl=ttl)

    def get_team(self, team_id: str, ttl: int = 86400) -> dict:
        """Single team profile."""
        return self.get("get_team", params={"team_id": team_id}, ttl=ttl)

    def get_squad(self, team_id: str, season_id: str = "", ttl: int = 86400) -> dict:
        """Team roster / squad."""
        params: Dict[str, Any] = {"team_id": team_id}
        if season_id:
            params["season_id"] = season_id
        return self.get("get_squad", params=params, ttl=ttl)

    def get_people(self, person_id: str, ttl: int = 86400 * 7) -> dict:
        """Player/person profile."""
        return self.get("get_people", params={"person_id": person_id}, ttl=ttl)

    def get_peoples(self, team_id: str = "", season_id: str = "",
                    ttl: int = 86400) -> dict:
        """List of players (optionally filtered by team/season)."""
        params: Dict[str, Any] = {}
        if team_id:
            params["team_id"] = team_id
        if season_id:
            params["season_id"] = season_id
        return self.get("get_peoples", params=params, ttl=ttl)

    # ------------------------------------------------------------------
    # Standings / H2H
    # ------------------------------------------------------------------

    def get_tables(self, season_id: str, ttl: int = 3600) -> dict:
        """Ladder / standings for a season."""
        return self.get("get_tables", params={"season_id": season_id}, ttl=ttl)

    def get_head2head(self, team1_id: str, team2_id: str,
                      ttl: int = 86400) -> dict:
        """Head-to-head historical record between two teams."""
        return self.get("get_head2head",
                        params={"team_id": team1_id, "team2_id": team2_id},
                        ttl=ttl)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def current_year() -> int:
        return datetime.utcnow().year

    def discover_afl_competition(self) -> Optional[str]:
        """
        Auto-discover the AFL competition ID from get_competitions.
        Returns the competition_id string, or None if not found.
        """
        try:
            data = self.get_competitions()
            competitions = (
                data.get("competitions") or
                data.get("competition") or
                data.get("data", {}).get("competitions") or []
            )
            if isinstance(competitions, dict):
                competitions = [competitions]
            for comp in (competitions or []):
                name = (comp.get("name") or comp.get("competition_name") or "").lower()
                if "afl" in name or "australian" in name:
                    cid = comp.get("id") or comp.get("competition_id")
                    if cid:
                        logger.info("DSG: discovered AFL competition_id={}", cid)
                        return str(cid)
        except Exception as exc:
            logger.warning("DSG: competition discovery failed: {}", exc)
        return None

    def discover_current_season(self, competition_id: str) -> Optional[str]:
        """
        Auto-discover the current/most recent season ID.
        Returns season_id string, or None.
        """
        try:
            data = self.get_seasons(competition_id)
            seasons = (
                data.get("seasons") or
                data.get("season") or
                data.get("data", {}).get("seasons") or []
            )
            if isinstance(seasons, dict):
                seasons = [seasons]
            if not seasons:
                return None
            # Sort by year descending, pick the most recent
            def _year(s):
                return int(s.get("year") or s.get("name") or "0")
            seasons_sorted = sorted(seasons, key=_year, reverse=True)
            current = seasons_sorted[0]
            sid = current.get("id") or current.get("season_id")
            if sid:
                logger.info("DSG: discovered current season_id={} year={}",
                            sid, current.get("year"))
                return str(sid)
        except Exception as exc:
            logger.warning("DSG: season discovery failed: {}", exc)
        return None
