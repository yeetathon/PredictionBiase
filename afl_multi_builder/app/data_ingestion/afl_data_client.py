"""
Data Sports Group (DSG) API client for AFL data.

Correct URL pattern (confirmed by DSG):
    https://dsg-api.com/clients/{client_name}/australian_football/{endpoint}
    ?type={type}&id={id}&client={client_name}&authkey={client_authkey}

Authentication: HTTP Basic Auth (client_name, authkey) PLUS the same
credentials repeated as query params — both are required.

All stats (team + player box score) come from get_matches?type=match&id={match_id}.
There is no separate get_match endpoint.
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

    URL structure:
        https://dsg-api.com/clients/{client}/australian_football/{endpoint}
        ?type={type}&id={id}&client={client}&authkey={authkey}

    Authentication: HTTP Basic Auth (client, authkey) + same values as query params.

    Available AFL endpoints (confirmed by DSG):
        get_areas, get_competitions, get_contestants, get_deleted,
        get_disciplines, get_drafts, get_head2head, get_matches,
        get_matches_day, get_matches_updates, get_medals, get_news,
        get_odds, get_peoples, get_peoples_updates, get_player_rankings,
        get_rankings, get_rounds, get_seasons, get_season_venue, get_squad,
        get_tables, get_team, get_trophies, get_venue

    All match data including box scores use get_matches with type=match.
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
            "User-Agent": "AFL-Multi-Builder/3.0",
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

        Correct URL: https://dsg-api.com/clients/{client}/australian_football/{endpoint}
        Auth: HTTP Basic Auth (client, authkey) + both as query params.
        """
        params = dict(params or {})
        params.update({
            "client": self._client,
            "authkey": self._authkey,
        })

        cache_key = _FileCache.make_key(endpoint, params)
        cached = self._cache.get(cache_key, ttl)
        if cached is not None:
            logger.debug("DSG cache hit: {}", endpoint)
            cached["_source"] = "cache"
            return cached

        # Correct path: /clients/{client_name}/australian_football/{endpoint}
        url = f"{self._base}/clients/{self._client}/{self._SPORT}/{endpoint}"
        last_exc: Optional[Exception] = None

        for attempt in range(self._MAX_RETRIES):
            try:
                resp = self._session.get(
                    url,
                    params=params,
                    auth=(self._client, self._authkey),  # HTTP Basic Auth required
                    timeout=30,
                )

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

                content_type = resp.headers.get("Content-Type", "")
                if "xml" in content_type and "json" not in content_type:
                    logger.warning("DSG {} returned XML instead of JSON", endpoint)
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
        """List all competitions."""
        return self.get("get_competitions", ttl=ttl)

    def get_seasons(self, competition_id: str, ttl: int = 86400) -> dict:
        """List seasons for a competition."""
        return self.get("get_seasons",
                        params={"type": "competition", "id": competition_id}, ttl=ttl)

    def get_rounds(self, season_id: str, ttl: int = 3600) -> dict:
        """List rounds in a season."""
        return self.get("get_rounds",
                        params={"type": "season", "id": season_id}, ttl=ttl)

    def get_areas(self, ttl: int = 86400 * 7) -> dict:
        """Geographic areas."""
        return self.get("get_areas", ttl=ttl)

    # ------------------------------------------------------------------
    # Matches / fixtures
    # ------------------------------------------------------------------

    def get_matches(self, season_id: str, ttl: int = 1800) -> dict:
        """All fixtures + results for a season."""
        return self.get("get_matches",
                        params={"type": "season", "id": season_id}, ttl=ttl)

    def get_matches_by_round(self, round_id: str, ttl: int = 1800) -> dict:
        """Fixtures for a specific round."""
        return self.get("get_matches",
                        params={"type": "round", "id": round_id}, ttl=ttl)

    def get_match(self, match_id: str, ttl: int = 86400) -> dict:
        """
        Full match details including box score, team stats, and player stats.
        Uses get_matches?type=match&id={match_id} — confirmed correct endpoint.
        There is no separate get_match endpoint in the DSG API.
        """
        return self.get("get_matches",
                        params={"type": "match", "id": match_id}, ttl=ttl)

    def get_matches_day(self, date: str, ttl: int = 900) -> dict:
        """Fixtures for a specific date (YYYY-MM-DD)."""
        return self.get("get_matches_day",
                        params={"type": "date", "id": date}, ttl=ttl)

    def get_matches_updates(self, ttl: int = 60) -> dict:
        """Live in-progress match updates."""
        return self.get("get_matches_updates", ttl=ttl)

    # ------------------------------------------------------------------
    # Teams / contestants
    # ------------------------------------------------------------------

    def get_contestants(self, season_id: str, ttl: int = 86400 * 7) -> dict:
        """Team/contestant list for a season (replaces get_teams)."""
        return self.get("get_contestants",
                        params={"type": "season", "id": season_id}, ttl=ttl)

    def get_team(self, team_id: str, ttl: int = 86400) -> dict:
        """Single team profile."""
        return self.get("get_team",
                        params={"type": "team", "id": team_id}, ttl=ttl)

    def get_squad(self, team_id: str, season_id: str = "", ttl: int = 86400) -> dict:
        """Team roster / squad."""
        params: Dict[str, Any] = {"type": "team", "id": team_id}
        if season_id:
            params["season_id"] = season_id
        return self.get("get_squad", params=params, ttl=ttl)

    def get_season_venue(self, season_id: str, ttl: int = 86400) -> dict:
        """Venues used in a season."""
        return self.get("get_season_venue",
                        params={"type": "season", "id": season_id}, ttl=ttl)

    def get_venue(self, venue_id: str, ttl: int = 86400 * 7) -> dict:
        """Single venue profile."""
        return self.get("get_venue",
                        params={"type": "venue", "id": venue_id}, ttl=ttl)

    # ------------------------------------------------------------------
    # Players / people
    # ------------------------------------------------------------------

    def get_people(self, person_id: str, ttl: int = 86400 * 7) -> dict:
        """Player/person profile."""
        return self.get("get_peoples",
                        params={"type": "person", "id": person_id}, ttl=ttl)

    def get_peoples(self, team_id: str = "", season_id: str = "",
                    ttl: int = 86400) -> dict:
        """List of players (optionally filtered by team/season)."""
        params: Dict[str, Any] = {}
        if team_id:
            params["type"] = "team"
            params["id"] = team_id
        elif season_id:
            params["type"] = "season"
            params["id"] = season_id
        return self.get("get_peoples", params=params, ttl=ttl)

    def get_peoples_updates(self, ttl: int = 300) -> dict:
        """Live player availability / injury updates."""
        return self.get("get_peoples_updates", ttl=ttl)

    # ------------------------------------------------------------------
    # Rankings
    # ------------------------------------------------------------------

    def get_rankings(self, season_id: str, ttl: int = 3600) -> dict:
        """Team rankings / ladder for a season."""
        return self.get("get_rankings",
                        params={"type": "season", "id": season_id}, ttl=ttl)

    def get_player_rankings(self, season_id: str, ttl: int = 3600) -> dict:
        """Player statistical rankings for a season."""
        return self.get("get_player_rankings",
                        params={"type": "season", "id": season_id}, ttl=ttl)

    def get_tables(self, season_id: str, ttl: int = 3600) -> dict:
        """Ladder / standings for a season."""
        return self.get("get_tables",
                        params={"type": "season", "id": season_id}, ttl=ttl)

    # ------------------------------------------------------------------
    # Odds / news / misc
    # ------------------------------------------------------------------

    def get_odds(self, match_id: str, ttl: int = 300) -> dict:
        """Bookmaker odds for a match."""
        return self.get("get_odds",
                        params={"type": "match", "id": match_id}, ttl=ttl)

    def get_news(self, team_id: str = "", ttl: int = 1800) -> dict:
        """News items, optionally filtered by team."""
        params: Dict[str, Any] = {}
        if team_id:
            params["type"] = "team"
            params["id"] = team_id
        return self.get("get_news", params=params, ttl=ttl)

    def get_head2head(self, team1_id: str, team2_id: str,
                      ttl: int = 86400) -> dict:
        """Head-to-head historical record between two teams."""
        return self.get("get_head2head",
                        params={"type": "team", "id": team1_id,
                                "team2_id": team2_id}, ttl=ttl)

    def get_trophies(self, season_id: str = "", ttl: int = 86400 * 7) -> dict:
        """Trophy / awards data."""
        params: Dict[str, Any] = {}
        if season_id:
            params["type"] = "season"
            params["id"] = season_id
        return self.get("get_trophies", params=params, ttl=ttl)

    def get_disciplines(self, ttl: int = 86400) -> dict:
        """Disciplinary records."""
        return self.get("get_disciplines", ttl=ttl)

    def get_drafts(self, season_id: str = "", ttl: int = 86400) -> dict:
        """Draft picks / history."""
        params: Dict[str, Any] = {}
        if season_id:
            params["type"] = "season"
            params["id"] = season_id
        return self.get("get_drafts", params=params, ttl=ttl)

    def get_deleted(self, ttl: int = 3600) -> dict:
        """Deleted / voided records."""
        return self.get("get_deleted", ttl=ttl)

    def get_medals(self, season_id: str = "", ttl: int = 86400) -> dict:
        """Medal / award records."""
        params: Dict[str, Any] = {}
        if season_id:
            params["type"] = "season"
            params["id"] = season_id
        return self.get("get_medals", params=params, ttl=ttl)

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
