"""
AFL Data Sports Group API HTTP client.

Authentication: authkey as URL parameter (primary) + X-Auth-Key header.
No quota limits — unlimited call rate.
Caches responses to minimise redundant calls.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from loguru import logger

from app.core.config import settings


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class AFLDataAPIError(Exception):
    def __init__(self, status_code: int, endpoint: str, message: str):
        self.status_code = status_code
        self.endpoint = endpoint
        self.message = message
        super().__init__(f"AFL Data API error [{status_code}] {endpoint}: {message}")


# ---------------------------------------------------------------------------
# Simple file-based cache (JSON, TTL-aware)
# ---------------------------------------------------------------------------

class _FileCache:
    """Lightweight JSON file cache with TTL."""

    def __init__(self, cache_dir: Path):
        self._dir = cache_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def _key_path(self, key: str) -> Path:
        return self._dir / f"afl_{key}.json"

    def get(self, key: str, ttl_seconds: int) -> Optional[dict]:
        p = self._key_path(key)
        if not p.exists():
            return None
        try:
            raw = json.loads(p.read_text())
            age = time.time() - raw.get("_ts", 0)
            if age > ttl_seconds:
                return None
            return raw.get("data")
        except Exception:
            return None

    def set(self, key: str, data: dict) -> None:
        try:
            self._key_path(key).write_text(
                json.dumps({"_ts": time.time(), "data": data}, default=str)
            )
        except Exception as exc:
            logger.debug("AFL cache write failed: {}", exc)

    @staticmethod
    def make_key(endpoint: str, params: dict) -> str:
        raw = endpoint + json.dumps(sorted(params.items()), default=str)
        return hashlib.sha1(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# AFL Data Sports Group API Client
# ---------------------------------------------------------------------------

class AFLDataClient:
    """
    HTTP client for the AFL Data Sports Group API.

    Authentication is via the authkey included as a query parameter on every
    request. The username and password are stored for potential OAuth flows.

    The AFL Advanced Pack provides:
      - Fixture lists (upcoming + historical)
      - Live scores and match events
      - Team and squad profiles
      - Box scores with full advanced stats:
          disposals, kicks, handballs, marks, goals, behinds, hitouts,
          tackles, inside 50s, clearances, contested possessions,
          uncontested possessions, score involvements, rebound 50s,
          clangers, frees for/against, 50m penalties, effective disposals,
          goal assists, time on ground, Brownlow votes
    """

    _RETRY_STATUSES = {429, 500, 502, 503, 504}
    _MAX_RETRIES = 4
    _BACKOFF_BASE = 2.0  # seconds

    def __init__(self):
        self._base_url = settings.afl_data_base_url.rstrip("/")
        self._authkey = settings.afl_data_authkey
        self._username = settings.afl_data_username
        self._password = settings.afl_data_password
        self._competition_id = settings.afl_data_competition_id
        self._cache = _FileCache(settings.raw_cache_dir / "afl_data")
        self._session = requests.Session()
        self._session.headers.update({
            "X-Auth-Key": self._authkey,
            "Accept": "application/json",
            "User-Agent": "AFL-Multi-Builder/1.0",
        })
        # Some endpoints may also use Basic auth
        if self._username and self._password:
            self._session.auth = (self._username, self._password)

    def is_configured(self) -> bool:
        return bool(self._authkey and self._authkey.strip())

    # ------------------------------------------------------------------
    # Low-level HTTP
    # ------------------------------------------------------------------

    def get(self, endpoint: str, params: Optional[Dict] = None,
            cache_ttl_seconds: int = 3600) -> dict:
        """
        GET an endpoint, respecting cache TTL.

        Always injects authkey into query params.
        Retries on transient errors with exponential backoff.
        """
        params = dict(params or {})
        params["authkey"] = self._authkey

        cache_key = _FileCache.make_key(endpoint, params)
        cached = self._cache.get(cache_key, cache_ttl_seconds)
        if cached is not None:
            logger.debug("AFL Data cache hit: {}", endpoint)
            cached["_source"] = "cache"
            return cached

        url = f"{self._base_url}/{endpoint.lstrip('/')}"
        last_exc: Optional[Exception] = None

        for attempt in range(self._MAX_RETRIES):
            try:
                resp = self._session.get(url, params=params, timeout=30)

                if resp.status_code in self._RETRY_STATUSES:
                    wait = self._BACKOFF_BASE ** attempt
                    logger.warning(
                        "AFL Data API {} → {} (attempt {}/{}), retry in {:.1f}s",
                        endpoint, resp.status_code, attempt + 1, self._MAX_RETRIES, wait,
                    )
                    time.sleep(wait)
                    continue

                if resp.status_code == 401:
                    raise AFLDataAPIError(401, endpoint,
                                          "Unauthorised — check AFL_DATA_AUTHKEY")
                if resp.status_code == 403:
                    raise AFLDataAPIError(403, endpoint,
                                          "Forbidden — check API plan permissions")
                if resp.status_code == 404:
                    raise AFLDataAPIError(404, endpoint,
                                          "Not found — endpoint may not exist")

                resp.raise_for_status()
                data = resp.json()
                data["_source"] = "api_live"
                data["_fetched_at"] = datetime.utcnow().isoformat()
                self._cache.set(cache_key, data)
                logger.debug("AFL Data API live: {} → {} bytes", endpoint, len(resp.content))
                return data

            except AFLDataAPIError:
                raise
            except requests.RequestException as exc:
                last_exc = exc
                wait = self._BACKOFF_BASE ** attempt
                logger.warning(
                    "AFL Data network error on {} (attempt {}/{}): {} — retry in {:.1f}s",
                    endpoint, attempt + 1, self._MAX_RETRIES, exc, wait,
                )
                time.sleep(wait)

        raise AFLDataAPIError(0, endpoint,
                              f"All {self._MAX_RETRIES} attempts failed: {last_exc}")

    # ------------------------------------------------------------------
    # AFL Data endpoints — Fixtures
    # ------------------------------------------------------------------

    def get_fixture_list(self, year: int, round_no: Optional[int] = None,
                         ttl: int = 3600) -> dict:
        """Return fixture list for a season (or specific round)."""
        params: Dict[str, Any] = {
            "competitionId": self._competition_id,
            "year": year,
        }
        if round_no is not None:
            params["roundId"] = round_no
        return self.get("cfs/afl/fixtureList", params=params, cache_ttl_seconds=ttl)

    def get_scores(self, year: int, round_no: Optional[int] = None,
                   ttl: int = 1800) -> dict:
        """Return match scores for a season (or specific round)."""
        params: Dict[str, Any] = {
            "competitionId": self._competition_id,
            "year": year,
        }
        if round_no is not None:
            params["roundId"] = round_no
        return self.get("cfs/afl/scoreList", params=params, cache_ttl_seconds=ttl)

    def get_match_details(self, match_id: str, ttl: int = 86400) -> dict:
        """Return full box score and match events for a single match."""
        return self.get(
            f"cfs/afl/matchDetails",
            params={"competitionId": self._competition_id, "matchId": match_id},
            cache_ttl_seconds=ttl,
        )

    def get_match_results(self, year: int, round_no: Optional[int] = None,
                          ttl: int = 3600) -> dict:
        """Return completed match results for a season."""
        params: Dict[str, Any] = {
            "competitionId": self._competition_id,
            "year": year,
        }
        if round_no is not None:
            params["roundId"] = round_no
        return self.get("cfs/afl/matchResults", params=params, cache_ttl_seconds=ttl)

    # ------------------------------------------------------------------
    # AFL Data endpoints — Teams and Players
    # ------------------------------------------------------------------

    def get_team_list(self, year: int, ttl: int = 86400 * 7) -> dict:
        """Return all teams for a competition year."""
        return self.get(
            "cfs/afl/teamList",
            params={"competitionId": self._competition_id, "year": year},
            cache_ttl_seconds=ttl,
        )

    def get_squad_list(self, year: int, team_id: Optional[int] = None,
                       ttl: int = 86400) -> dict:
        """Return squad/player roster for a team or all teams."""
        params: Dict[str, Any] = {
            "competitionId": self._competition_id,
            "year": year,
        }
        if team_id is not None:
            params["teamId"] = team_id
        return self.get("cfs/afl/squadList", params=params, cache_ttl_seconds=ttl)

    # ------------------------------------------------------------------
    # AFL Data endpoints — Statistics
    # ------------------------------------------------------------------

    def get_player_stats(self, year: int, round_no: Optional[int] = None,
                         team_id: Optional[int] = None,
                         player_id: Optional[int] = None,
                         ttl: int = 3600) -> dict:
        """
        Return individual player statistics.

        Advanced stats included (AFL Advanced Pack):
          disposals, kicks, handballs, marks, goals, behinds, hitouts,
          tackles, inside_50s, clearances, contested_possessions,
          uncontested_possessions, score_involvements, rebound_50s,
          clangers, frees_for, frees_against, effective_kicks,
          effective_handballs, effective_disposals, goal_assists,
          time_on_ground, contested_marks, marks_inside_50,
          one_percenters, brownlow_votes
        """
        params: Dict[str, Any] = {
            "competitionId": self._competition_id,
            "year": year,
        }
        if round_no is not None:
            params["roundId"] = round_no
        if team_id is not None:
            params["teamId"] = team_id
        if player_id is not None:
            params["playerId"] = player_id
        return self.get("cfs/afl/playerStats", params=params, cache_ttl_seconds=ttl)

    def get_team_stats(self, year: int, round_no: Optional[int] = None,
                       ttl: int = 3600) -> dict:
        """Return team-level aggregated statistics per match."""
        params: Dict[str, Any] = {
            "competitionId": self._competition_id,
            "year": year,
        }
        if round_no is not None:
            params["roundId"] = round_no
        return self.get("cfs/afl/teamStats", params=params, cache_ttl_seconds=ttl)

    def get_ladder(self, year: int, round_no: Optional[int] = None,
                   ttl: int = 3600) -> dict:
        """Return competition ladder/standings."""
        params: Dict[str, Any] = {
            "competitionId": self._competition_id,
            "year": year,
        }
        if round_no is not None:
            params["roundId"] = round_no
        return self.get("cfs/afl/ladderList", params=params, cache_ttl_seconds=ttl)

    def get_live_scores(self, ttl: int = 60) -> dict:
        """Return live in-progress match scores (very short TTL)."""
        return self.get(
            "cfs/afl/liveScores",
            params={"competitionId": self._competition_id},
            cache_ttl_seconds=ttl,
        )

    def get_player_list(self, year: int, team_id: Optional[int] = None,
                        ttl: int = 86400) -> dict:
        """Return full player list (with profiles)."""
        params: Dict[str, Any] = {
            "competitionId": self._competition_id,
            "year": year,
        }
        if team_id is not None:
            params["teamId"] = team_id
        return self.get("cfs/afl/playerList", params=params, cache_ttl_seconds=ttl)

    # ------------------------------------------------------------------
    # Convenience: current year
    # ------------------------------------------------------------------

    @staticmethod
    def current_year() -> int:
        return datetime.utcnow().year
