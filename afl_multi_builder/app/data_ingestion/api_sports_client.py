"""API-Sports AFL HTTP client.

Wraps the API-Sports AFL v1 REST API (https://v1.afl.api-sports.io) with:
- Token-bucket rate limiting (minimum 1.0 s between requests — conservative
  for the free-tier 100 req/day cap)
- Transparent cache layer via CacheManager
- Retry with exponential back-off on 429 / 5xx responses
- Structured logging via loguru at DEBUG / WARNING levels

Cache TTLs
----------
- Fixtures  : 2 hours  (upcoming games may update frequently)
- Teams      : 24 hours (roster data changes slowly)
- Standings  : 24 hours (ladder updates after each round)
- Injuries   : 1 hour   (high churn near game day)
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests
from loguru import logger

from app.core.config import settings
from app.data_ingestion.cache_manager import CacheManager


# ---------------------------------------------------------------------------
# TTL constants (seconds)
# ---------------------------------------------------------------------------

_TTL_FIXTURES: int = 2 * 3600       # 2 hours
_TTL_TEAMS: int = 24 * 3600         # 24 hours
_TTL_STANDINGS: int = 24 * 3600     # 24 hours
_TTL_INJURIES: int = 1 * 3600       # 1 hour

# Status mapping: API-Sports short code → internal canonical status
_STATUS_MAP: Dict[str, str] = {
    "NS": "upcoming",
    "Q1": "in_progress",
    "Q2": "in_progress",
    "Q3": "in_progress",
    "Q4": "in_progress",
    "HT": "in_progress",
    "ET": "in_progress",
    "FT": "completed",
    "AOT": "completed",
    "POST": "postponed",
    "CANC": "cancelled",
    "SUSP": "suspended",
    "ABD": "abandoned",
    "TBD": "upcoming",
    "WO": "walkover",
}


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class APISportsError(Exception):
    """Raised when the API-Sports endpoint returns an unrecoverable error.

    Attributes
    ----------
    status_code : int | None
        HTTP status code of the failing response, or ``None`` if the request
        never reached the server (e.g. network error).
    endpoint : str
        The endpoint path that was requested.
    message : str
        Human-readable description of the failure.
    """

    def __init__(
        self,
        message: str,
        endpoint: str = "",
        status_code: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.endpoint = endpoint
        self.status_code = status_code

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"APISportsError(status_code={self.status_code!r}, "
            f"endpoint={self.endpoint!r}, message={self.message!r})"
        )


# ---------------------------------------------------------------------------
# Token-bucket rate limiter (thread-safe)
# ---------------------------------------------------------------------------

class _TokenBucket:
    """Enforce a minimum inter-request gap.

    The bucket refills at *rate_qps* tokens per second.  Each call to
    ``consume()`` blocks until a token is available.
    """

    def __init__(self, rate_qps: float, min_interval_seconds: float) -> None:
        self._rate = rate_qps
        self._min_interval = min_interval_seconds
        self._tokens: float = 1.0
        self._last_check: float = time.monotonic()
        self._lock = threading.Lock()

    def consume(self) -> None:
        """Block until one token is available, then consume it."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_check
            self._last_check = now
            self._tokens = min(1.0, self._tokens + elapsed * self._rate)
            if self._tokens < 1.0:
                wait = (1.0 - self._tokens) / self._rate
                time.sleep(wait)
                self._tokens = 0.0
            else:
                self._tokens -= 1.0
            # Hard minimum interval between requests
            since_last = time.monotonic() - (self._last_check - elapsed)
            if since_last < self._min_interval:
                time.sleep(self._min_interval - since_last)


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------

class APISportsClient:
    """HTTP client for the API-Sports AFL v1 API.

    Parameters
    ----------
    api_key : str
        API-Sports API key.  Raises ``RuntimeError`` at construction time if
        empty.
    base_url : str
        Base URL, e.g. ``https://v1.afl.api-sports.io``.
    league_id : int
        Numeric AFL league identifier (default from settings).
    cache_manager : CacheManager | None
        Optional response cache.  When provided, successful responses are
        stored and returned on subsequent identical requests.
    timeout : int
        HTTP request timeout in seconds (default 30).
    """

    _RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
    _BACKOFF_SECONDS = (2, 4, 8)

    # Conservative rate for free tier (100 req/day): 1 req/s maximum so we
    # never burst faster than the API can handle, but we do not add artificial
    # daily quotas here — the caller controls when to call.
    _RATE_QPS: float = 1.0
    _MIN_INTERVAL_SECONDS: float = 1.0

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        league_id: Optional[int] = None,
        cache_manager: Optional[CacheManager] = None,
        timeout: int = 30,
    ) -> None:
        resolved_key = api_key if api_key is not None else settings.api_sports_key
        if not resolved_key or not resolved_key.strip():
            raise RuntimeError(
                "APISportsClient requires a valid API-Sports key. "
                "Set the API_SPORTS_KEY environment variable or provide "
                "api_key= at construction time."
            )
        self._api_key: str = resolved_key.strip()
        self._base_url: str = (
            (base_url if base_url is not None else settings.api_sports_base_url).rstrip("/")
        )
        self._league_id: int = (
            league_id if league_id is not None else settings.api_sports_afl_league_id
        )
        self._cache_manager: Optional[CacheManager] = cache_manager
        self._timeout: int = timeout
        self._bucket = _TokenBucket(
            rate_qps=self._RATE_QPS,
            min_interval_seconds=self._MIN_INTERVAL_SECONDS,
        )
        self._session = requests.Session()
        self._session.headers.update(
            {
                "x-apisports-key": self._api_key,
                "Accept": "application/json",
                "User-Agent": "AFL-PredictionBiase/1.0",
            }
        )
        logger.debug(
            "APISportsClient initialised: base_url={} league_id={}",
            self._base_url,
            self._league_id,
        )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def league_id(self) -> int:
        """AFL league identifier used for all parameterised requests."""
        return self._league_id

    # ------------------------------------------------------------------
    # Core request method
    # ------------------------------------------------------------------

    def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        cache_ttl_seconds: int = _TTL_FIXTURES,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """Fetch *endpoint* from the API-Sports AFL API (or cache).

        Parameters
        ----------
        endpoint :
            Path relative to *base_url*, e.g. ``games``.
        params :
            Additional query-string parameters.  The API key is passed via
            request header, NOT here.
        cache_ttl_seconds :
            Maximum age (seconds) of a cached response before re-fetching.
        force_refresh :
            When ``True``, bypass the cache and always hit the live API.

        Returns
        -------
        dict
            Parsed JSON response, augmented with:

            - ``_source``: ``"cache"`` or ``"api_live"``
            - ``_fetch_timestamp``: ISO-8601 UTC string (only on live fetch)

        Raises
        ------
        APISportsError
            If the API returns an error after all retries are exhausted.
        """
        params = dict(params or {})

        # ---- 1. Cache lookup ------------------------------------------------
        cache_key: Optional[str] = None
        if self._cache_manager is not None:
            cache_key = self._cache_manager.make_key(endpoint, params)
            if not force_refresh:
                cached = self._cache_manager.get(cache_key, ttl_seconds=cache_ttl_seconds)
                if cached is not None:
                    logger.debug(
                        "APISports cache HIT endpoint={} key={}", endpoint, cache_key
                    )
                    result = dict(cached)
                    result["_source"] = "cache"
                    return result
                logger.debug(
                    "APISports cache MISS endpoint={} key={}", endpoint, cache_key
                )

        # ---- 2. Rate limit --------------------------------------------------
        self._bucket.consume()

        # ---- 3. HTTP request with retry -------------------------------------
        url = f"{self._base_url}/{endpoint}"

        last_exc: Optional[Exception] = None
        last_status: Optional[int] = None
        t_start = time.monotonic()

        for attempt, backoff in enumerate([0] + list(self._BACKOFF_SECONDS)):
            if backoff > 0:
                logger.debug(
                    "APISports retry attempt={} endpoint={} backoff={}s",
                    attempt,
                    endpoint,
                    backoff,
                )
                time.sleep(backoff)

            try:
                t_req = time.monotonic()
                response = self._session.get(
                    url,
                    params=params,
                    timeout=self._timeout,
                )
                latency_ms = (time.monotonic() - t_req) * 1000.0
                last_status = response.status_code

                logger.debug(
                    "APISports HTTP {} endpoint={} attempt={} latency={:.1f}ms",
                    response.status_code,
                    endpoint,
                    attempt,
                    latency_ms,
                )

                if response.status_code == 200:
                    fetch_ts = datetime.now(timezone.utc).isoformat()
                    try:
                        data: Dict[str, Any] = response.json()
                    except ValueError as parse_exc:
                        raise APISportsError(
                            message=f"Failed to parse JSON response: {parse_exc}",
                            endpoint=endpoint,
                            status_code=200,
                        ) from parse_exc

                    # Persist to cache
                    if self._cache_manager is not None and cache_key is not None:
                        params_hash = self._cache_manager.make_key(endpoint, params)
                        self._cache_manager.set(
                            cache_key=cache_key,
                            endpoint=endpoint,
                            params_hash=params_hash,
                            data=data,
                            ttl_seconds=cache_ttl_seconds,
                        )

                    data["_source"] = "api_live"
                    data["_fetch_timestamp"] = fetch_ts
                    logger.debug(
                        "APISports live response endpoint={} results={}",
                        endpoint,
                        data.get("results", "?"),
                    )
                    return data

                elif response.status_code in self._RETRY_STATUSES:
                    last_exc = APISportsError(
                        message=f"HTTP {response.status_code}: {response.text[:200]}",
                        endpoint=endpoint,
                        status_code=response.status_code,
                    )
                    # Continue to next retry iteration

                else:
                    raise APISportsError(
                        message=(
                            f"HTTP {response.status_code} (non-retryable): "
                            f"{response.text[:200]}"
                        ),
                        endpoint=endpoint,
                        status_code=response.status_code,
                    )

            except APISportsError:
                raise
            except requests.RequestException as req_exc:
                logger.debug(
                    "APISports request error endpoint={} attempt={} error={}",
                    endpoint,
                    attempt,
                    req_exc,
                )
                last_exc = APISportsError(
                    message=f"Network request failed: {req_exc}",
                    endpoint=endpoint,
                    status_code=None,
                )

        # All retries exhausted
        total_ms = (time.monotonic() - t_start) * 1000.0
        logger.warning(
            "APISports all retries exhausted endpoint={} last_status={} total={:.0f}ms",
            endpoint,
            last_status,
            total_ms,
        )
        if last_exc is not None:
            raise last_exc
        raise APISportsError(
            message="All retries exhausted with no specific error recorded",
            endpoint=endpoint,
            status_code=last_status,
        )

    # ------------------------------------------------------------------
    # Domain-level endpoint methods
    # ------------------------------------------------------------------

    def get_leagues(self, force_refresh: bool = False) -> Dict[str, Any]:
        """GET /leagues — return all leagues, highlighting the AFL league.

        Returns
        -------
        dict
            Raw API response with keys ``response`` (list), ``results`` (int),
            plus ``_source`` and optionally ``_fetch_timestamp``.
        """
        result = self.get(
            "leagues",
            cache_ttl_seconds=_TTL_TEAMS,   # League list changes rarely
            force_refresh=force_refresh,
        )
        # Log confirmation that AFL league is present
        for entry in result.get("response", []):
            league = entry.get("league", {})
            if league.get("id") == self._league_id:
                logger.debug(
                    "APISports AFL league confirmed: id={} name={}",
                    league.get("id"),
                    league.get("name"),
                )
                break
        else:
            logger.warning(
                "APISports AFL league id={} not found in /leagues response",
                self._league_id,
            )
        return result

    def get_fixtures(
        self,
        season: int,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """GET /games?league={id}&season={season} — upcoming and past fixtures.

        Parameters
        ----------
        season :
            Four-digit season year, e.g. ``2025``.

        Returns
        -------
        dict
            Raw API response with ``response`` (list of game objects).
        """
        params = {"league": self._league_id, "season": season}
        result = self.get(
            "games",
            params=params,
            cache_ttl_seconds=_TTL_FIXTURES,
            force_refresh=force_refresh,
        )
        logger.debug(
            "APISports get_fixtures season={} count={} source={}",
            season,
            result.get("results", "?"),
            result.get("_source"),
        )
        return result

    def get_teams(
        self,
        season: int,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """GET /teams?league={id}&season={season} — full teams list.

        Parameters
        ----------
        season :
            Four-digit season year.

        Returns
        -------
        dict
            Raw API response with ``response`` (list of team objects).
        """
        params = {"league": self._league_id, "season": season}
        result = self.get(
            "teams",
            params=params,
            cache_ttl_seconds=_TTL_TEAMS,
            force_refresh=force_refresh,
        )
        logger.debug(
            "APISports get_teams season={} count={} source={}",
            season,
            result.get("results", "?"),
            result.get("_source"),
        )
        return result

    def get_standings(
        self,
        season: int,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """GET /standings?league={id}&season={season} — season ladder.

        Parameters
        ----------
        season :
            Four-digit season year.

        Returns
        -------
        dict
            Raw API response with ``response`` (list of standing objects).
        """
        params = {"league": self._league_id, "season": season}
        result = self.get(
            "standings",
            params=params,
            cache_ttl_seconds=_TTL_STANDINGS,
            force_refresh=force_refresh,
        )
        logger.debug(
            "APISports get_standings season={} count={} source={}",
            season,
            result.get("results", "?"),
            result.get("_source"),
        )
        return result

    def get_injuries(
        self,
        season: int,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """GET /injuries?league={id}&season={season} — current injury list.

        Parameters
        ----------
        season :
            Four-digit season year.

        Returns
        -------
        dict
            Raw API response with ``response`` (list of injury objects).
        """
        params = {"league": self._league_id, "season": season}
        result = self.get(
            "injuries",
            params=params,
            cache_ttl_seconds=_TTL_INJURIES,
            force_refresh=force_refresh,
        )
        logger.debug(
            "APISports get_injuries season={} count={} source={}",
            season,
            result.get("results", "?"),
            result.get("_source"),
        )
        return result

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def normalize_fixtures(
        self,
        raw_response: Dict[str, Any],
        season: int,
    ) -> List[Dict[str, Any]]:
        """Map API-Sports game objects to the internal fixture schema.

        Parameters
        ----------
        raw_response :
            The dict returned by :py:meth:`get_fixtures`.
        season :
            The season year used for the original request (embedded into each
            normalised fixture for convenience).

        Returns
        -------
        list[dict]
            Each element contains:

            ============== =================================================
            Field          Description
            ============== =================================================
            fixture_id     ``int`` — game.id from the API
            season         ``int`` — season year
            round          ``str | None`` — round label, e.g. "Round 3"
            home_team_id   ``int | None``
            away_team_id   ``int | None``
            home_team_name ``str | None``
            away_team_name ``str | None``
            scheduled_utc  ``str | None`` — ISO-8601 UTC datetime string
            status         ``str`` — one of upcoming / in_progress /
                           completed / postponed / cancelled / suspended /
                           abandoned / walkover / unknown
            home_score     ``int | None``
            away_score     ``int | None``
            home_win       ``bool | None`` — True if home team won
            margin         ``int | None`` — abs(home_score - away_score)
            total_score    ``int | None`` — home_score + away_score
            venue_name     ``str | None``
            ============== =================================================
        """
        games: List[Any] = raw_response.get("response", [])
        normalised: List[Dict[str, Any]] = []

        for game_obj in games:
            try:
                fixture = self._normalise_single_game(game_obj, season)
            except Exception as exc:  # noqa: BLE001
                game_id = (game_obj or {}).get("game", {}).get("id", "?")
                logger.warning(
                    "APISports normalise_fixtures: skipping game_id={} error={}",
                    game_id,
                    exc,
                )
                continue
            normalised.append(fixture)

        logger.debug(
            "APISports normalize_fixtures: season={} total_in={} normalised={}",
            season,
            len(games),
            len(normalised),
        )
        return normalised

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_single_game(
        game_obj: Dict[str, Any],
        season: int,
    ) -> Dict[str, Any]:
        """Convert a single API-Sports game object to internal schema."""
        game = game_obj.get("game", {})
        teams = game_obj.get("teams", {})
        scores = game_obj.get("scores", {})
        league = game_obj.get("league", {})
        venue = game_obj.get("venue") or {}

        game_id: int = int(game.get("id", 0))

        # Teams
        home_team = teams.get("home") or {}
        away_team = teams.get("away") or {}
        home_team_id: Optional[int] = (
            int(home_team["id"]) if home_team.get("id") is not None else None
        )
        away_team_id: Optional[int] = (
            int(away_team["id"]) if away_team.get("id") is not None else None
        )
        home_team_name: Optional[str] = home_team.get("name")
        away_team_name: Optional[str] = away_team.get("name")

        # Scheduled time — API-Sports uses "date" field in UTC
        scheduled_utc: Optional[str] = game.get("date")
        if scheduled_utc:
            # Normalise to a consistent ISO-8601 UTC format
            try:
                dt = datetime.fromisoformat(scheduled_utc.replace("Z", "+00:00"))
                scheduled_utc = dt.astimezone(timezone.utc).isoformat()
            except ValueError:
                pass  # Keep the raw string if parsing fails

        # Status
        status_obj = game.get("status", {})
        short_code: str = (status_obj.get("short") or "").upper()
        status: str = _STATUS_MAP.get(short_code, "unknown")

        # Scores
        home_score_raw = (scores.get("home") or {}).get("total")
        away_score_raw = (scores.get("away") or {}).get("total")
        home_score: Optional[int] = int(home_score_raw) if home_score_raw is not None else None
        away_score: Optional[int] = int(away_score_raw) if away_score_raw is not None else None

        # Derived fields — only computed for completed games with valid scores
        home_win: Optional[bool] = None
        margin: Optional[int] = None
        total_score: Optional[int] = None
        if home_score is not None and away_score is not None:
            total_score = home_score + away_score
            margin = abs(home_score - away_score)
            if home_score != away_score:
                home_win = home_score > away_score
            else:
                home_win = None  # Draw — rare in AFL but possible

        # Round
        round_label: Optional[str] = league.get("round")

        # Venue
        venue_name: Optional[str] = (
            venue.get("name") if isinstance(venue, dict) else None
        )

        return {
            "fixture_id": game_id,
            "season": season,
            "round": round_label,
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "home_team_name": home_team_name,
            "away_team_name": away_team_name,
            "scheduled_utc": scheduled_utc,
            "status": status,
            "home_score": home_score,
            "away_score": away_score,
            "home_win": home_win,
            "margin": margin,
            "total_score": total_score,
            "venue_name": venue_name,
        }
