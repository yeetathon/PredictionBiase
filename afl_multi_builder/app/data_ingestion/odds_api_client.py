"""The Odds API HTTP client.

Wraps The Odds API v4 (https://api.the-odds-api.com/v4) with:
- Token-bucket rate limiting (conservative for the free-tier 500 req/month cap)
- Transparent cache layer via CacheManager
- Retry with exponential back-off on 429 / 5xx responses
- Structured logging via loguru at DEBUG / WARNING levels
- Quota monitoring via response headers (x-requests-remaining, x-requests-used)

Cache TTLs
----------
- Odds    : 15 minutes  (prices shift constantly)
- Sports  : 1 hour      (sport availability is stable intra-day)
- Scores  : 1 hour      (recent results update a few times per day)
"""
from __future__ import annotations

import re
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from loguru import logger

from app.core.config import settings
from app.data_ingestion.cache_manager import CacheManager


# ---------------------------------------------------------------------------
# TTL constants (seconds)
# ---------------------------------------------------------------------------

_TTL_ODDS: int = 15 * 60      # 15 minutes
_TTL_SPORTS: int = 1 * 3600   # 1 hour
_TTL_SCORES: int = 1 * 3600   # 1 hour

# Market type normalisation: Odds API market key → internal label
_MARKET_MAP: Dict[str, str] = {
    "h2h": "head_to_head",
    "spreads": "spreads",
    "totals": "totals",
}

# Tokens to strip when normalising team names for fuzzy matching
_STRIP_TOKENS = re.compile(
    r"\b(afc|fc|afl|sc|city|united|town|sporting|football|club|"
    r"the|western|eastern|northern|southern|central|port)\b",
    flags=re.IGNORECASE,
)
_WHITESPACE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class OddsAPIError(Exception):
    """Raised when The Odds API returns an unrecoverable error.

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
            f"OddsAPIError(status_code={self.status_code!r}, "
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


class OddsAPIClient:
    """HTTP client for The Odds API v4.

    Parameters
    ----------
    api_key : str | None
        The Odds API key.  Raises ``RuntimeError`` at construction time if
        empty.  Falls back to ``settings.odds_api_key`` when ``None``.
    base_url : str | None
        Base URL for the API, e.g. ``https://api.the-odds-api.com/v4``.
        Falls back to ``settings.odds_api_base_url``.
    sport : str | None
        Sport key for AFL endpoints, e.g. ``aussierules_afl``.
        Falls back to ``settings.odds_api_sport``.
    cache_manager : CacheManager | None
        Optional response cache.  When provided, successful responses are
        stored and returned on subsequent identical requests.
    timeout : int
        HTTP request timeout in seconds (default 30).
    """

    _RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
    _BACKOFF_SECONDS = (2, 4, 8)

    # Very conservative for 500 req/month free tier: cap at 1 req/3 s to avoid
    # burning the quota in any automated loop.
    _RATE_QPS: float = 1.0 / 3.0
    _MIN_INTERVAL_SECONDS: float = 3.0

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        sport: Optional[str] = None,
        cache_manager: Optional[CacheManager] = None,
        timeout: int = 30,
    ) -> None:
        resolved_key = api_key if api_key is not None else settings.odds_api_key
        if not resolved_key or not resolved_key.strip():
            raise RuntimeError(
                "OddsAPIClient requires a valid Odds API key. "
                "Set the ODDS_API_KEY environment variable or provide "
                "api_key= at construction time."
            )
        self._api_key: str = resolved_key.strip()
        self._base_url: str = (
            (base_url if base_url is not None else settings.odds_api_base_url).rstrip("/")
        )
        self._sport: str = sport if sport is not None else settings.odds_api_sport
        self._cache_manager: Optional[CacheManager] = cache_manager
        self._timeout: int = timeout
        self._bucket = _TokenBucket(
            rate_qps=self._RATE_QPS,
            min_interval_seconds=self._MIN_INTERVAL_SECONDS,
        )
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "AFL-PredictionBiase/1.0",
            }
        )
        logger.debug(
            "OddsAPIClient initialised: base_url={} sport={}",
            self._base_url,
            self._sport,
        )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def sport(self) -> str:
        """AFL sport key used for odds/scores endpoints."""
        return self._sport

    # ------------------------------------------------------------------
    # Core request method
    # ------------------------------------------------------------------

    def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        cache_ttl_seconds: int = _TTL_ODDS,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """Fetch *endpoint* from The Odds API (or cache).

        The ``apiKey`` query parameter is injected automatically — do not
        include it in *params*.

        Parameters
        ----------
        endpoint :
            Path relative to *base_url*, e.g.
            ``sports/aussierules_afl/odds``.
        params :
            Additional query-string parameters (without ``apiKey``).
        cache_ttl_seconds :
            Maximum age (seconds) of a cached response before re-fetching.
        force_refresh :
            When ``True``, bypass the cache and always hit the live API.

        Returns
        -------
        dict
            A dict with key ``"data"`` containing the parsed JSON payload
            (list or dict as returned by the API), plus:

            - ``_source``: ``"cache"`` or ``"api_live"``
            - ``_fetch_timestamp``: ISO-8601 UTC string (live responses only)
            - ``_quota_remaining``: int or ``None`` (live responses only)
            - ``_quota_used``: int or ``None`` (live responses only)

        Raises
        ------
        OddsAPIError
            If the API returns an error after all retries are exhausted.
        """
        params = dict(params or {})
        # apiKey must not leak into the cache key; add it only at request time
        request_params = dict(params)
        request_params["apiKey"] = self._api_key

        # ---- 1. Cache lookup ------------------------------------------------
        cache_key: Optional[str] = None
        if self._cache_manager is not None:
            # Build cache key without apiKey so key rotation does not bust cache
            cache_key = self._cache_manager.make_key(endpoint, params)
            if not force_refresh:
                cached = self._cache_manager.get(cache_key, ttl_seconds=cache_ttl_seconds)
                if cached is not None:
                    logger.debug(
                        "OddsAPI cache HIT endpoint={} key={}", endpoint, cache_key
                    )
                    result = dict(cached)
                    result["_source"] = "cache"
                    return result
                logger.debug(
                    "OddsAPI cache MISS endpoint={} key={}", endpoint, cache_key
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
                    "OddsAPI retry attempt={} endpoint={} backoff={}s",
                    attempt,
                    endpoint,
                    backoff,
                )
                time.sleep(backoff)

            try:
                t_req = time.monotonic()
                response = self._session.get(
                    url,
                    params=request_params,
                    timeout=self._timeout,
                )
                latency_ms = (time.monotonic() - t_req) * 1000.0
                last_status = response.status_code

                logger.debug(
                    "OddsAPI HTTP {} endpoint={} attempt={} latency={:.1f}ms",
                    response.status_code,
                    endpoint,
                    attempt,
                    latency_ms,
                )

                # Log quota headers on every response (they are only present on
                # live API responses, not cached ones)
                quota_remaining = self._parse_quota_header(
                    response.headers.get("x-requests-remaining")
                )
                quota_used = self._parse_quota_header(
                    response.headers.get("x-requests-used")
                )
                if quota_remaining is not None or quota_used is not None:
                    logger.info(
                        "OddsAPI quota: used={} remaining={}",
                        quota_used,
                        quota_remaining,
                    )
                    if quota_remaining is not None and quota_remaining <= 20:
                        logger.warning(
                            "OddsAPI quota critically low: only {} requests remaining",
                            quota_remaining,
                        )

                if response.status_code == 200:
                    fetch_ts = datetime.now(timezone.utc).isoformat()
                    try:
                        raw_payload = response.json()
                    except ValueError as parse_exc:
                        raise OddsAPIError(
                            message=f"Failed to parse JSON response: {parse_exc}",
                            endpoint=endpoint,
                            status_code=200,
                        ) from parse_exc

                    # Wrap list responses so the cache always stores a dict
                    if isinstance(raw_payload, list):
                        data: Dict[str, Any] = {"data": raw_payload}
                    else:
                        data = dict(raw_payload)
                        data.setdefault("data", raw_payload)

                    # Persist to cache (before adding volatile metadata)
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
                    data["_quota_remaining"] = quota_remaining
                    data["_quota_used"] = quota_used
                    logger.debug(
                        "OddsAPI live response endpoint={} source=api_live",
                        endpoint,
                    )
                    return data

                elif response.status_code in self._RETRY_STATUSES:
                    last_exc = OddsAPIError(
                        message=f"HTTP {response.status_code}: {response.text[:200]}",
                        endpoint=endpoint,
                        status_code=response.status_code,
                    )
                    # Continue to next retry iteration

                else:
                    raise OddsAPIError(
                        message=(
                            f"HTTP {response.status_code} (non-retryable): "
                            f"{response.text[:200]}"
                        ),
                        endpoint=endpoint,
                        status_code=response.status_code,
                    )

            except OddsAPIError:
                raise
            except requests.RequestException as req_exc:
                logger.debug(
                    "OddsAPI request error endpoint={} attempt={} error={}",
                    endpoint,
                    attempt,
                    req_exc,
                )
                last_exc = OddsAPIError(
                    message=f"Network request failed: {req_exc}",
                    endpoint=endpoint,
                    status_code=None,
                )

        # All retries exhausted
        total_ms = (time.monotonic() - t_start) * 1000.0
        logger.warning(
            "OddsAPI all retries exhausted endpoint={} last_status={} total={:.0f}ms",
            endpoint,
            last_status,
            total_ms,
        )
        if last_exc is not None:
            raise last_exc
        raise OddsAPIError(
            message="All retries exhausted with no specific error recorded",
            endpoint=endpoint,
            status_code=last_status,
        )

    # ------------------------------------------------------------------
    # Domain-level endpoint methods
    # ------------------------------------------------------------------

    def get_sports(self, force_refresh: bool = False) -> Dict[str, Any]:
        """GET /sports — list all available sports.

        Logs a confirmation when ``aussierules_afl`` is found in the response.

        Returns
        -------
        dict
            ``data`` key holds a list of sport objects.  Each object has at
            minimum ``key``, ``title``, ``active`` fields.
        """
        result = self.get(
            "sports",
            cache_ttl_seconds=_TTL_SPORTS,
            force_refresh=force_refresh,
        )
        sports_list: List[Dict[str, Any]] = result.get("data", [])
        afl_found = any(s.get("key") == self._sport for s in sports_list)
        if afl_found:
            logger.debug(
                "OddsAPI sport confirmed: key={}", self._sport
            )
        else:
            logger.warning(
                "OddsAPI sport key={} not found in /sports response "
                "(total sports={})",
                self._sport,
                len(sports_list),
            )
        return result

    def get_odds(
        self,
        regions: str = "au",
        markets: str = "h2h,spreads",
        bookmakers: Optional[str] = None,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """GET /sports/{sport}/odds — live and upcoming match odds.

        Parameters
        ----------
        regions :
            Comma-separated region codes, e.g. ``"au"`` or ``"au,us"``.
        markets :
            Comma-separated market keys, e.g. ``"h2h,spreads"``.
        bookmakers :
            Optional comma-separated bookmaker slugs to restrict results.
            Defaults to ``settings.odds_api_bookmakers_list`` joined as a
            string.

        Returns
        -------
        dict
            ``data`` key holds a list of event objects, each containing
            ``bookmakers`` with nested ``markets`` and ``outcomes``.
        """
        if bookmakers is None:
            bookmakers = ",".join(settings.odds_api_bookmakers_list)

        params: Dict[str, Any] = {
            "regions": regions,
            "markets": markets,
        }
        if bookmakers:
            params["bookmakers"] = bookmakers

        result = self.get(
            f"sports/{self._sport}/odds",
            params=params,
            cache_ttl_seconds=_TTL_ODDS,
            force_refresh=force_refresh,
        )
        logger.debug(
            "OddsAPI get_odds regions={} markets={} events={} source={}",
            regions,
            markets,
            len(result.get("data", [])),
            result.get("_source"),
        )
        return result

    def get_scores(
        self,
        days_from: int = 1,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """GET /sports/{sport}/scores — recent results with scores.

        Parameters
        ----------
        days_from :
            Number of days in the past to include completed results (1–3).

        Returns
        -------
        dict
            ``data`` key holds a list of score objects with ``scores`` nested
            arrays for home and away.
        """
        params: Dict[str, Any] = {"daysFrom": days_from}
        result = self.get(
            f"sports/{self._sport}/scores",
            params=params,
            cache_ttl_seconds=_TTL_SCORES,
            force_refresh=force_refresh,
        )
        logger.debug(
            "OddsAPI get_scores days_from={} events={} source={}",
            days_from,
            len(result.get("data", [])),
            result.get("_source"),
        )
        return result

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def normalize_odds(
        self,
        raw_response: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Flatten Odds API event objects into a list of per-outcome rows.

        Parameters
        ----------
        raw_response :
            The dict returned by :py:meth:`get_odds`.

        Returns
        -------
        list[dict]
            Each element contains:

            ============== =================================================
            Field          Description
            ============== =================================================
            fixture_label  ``str`` — ``"Team A vs Team B"``
            home_team      ``str`` — home_team field from the API
            away_team      ``str`` — away_team field from the API
            commence_time  ``str`` — ISO-8601 UTC commence time
            bookmaker      ``str`` — bookmaker key / slug
            market_type    ``str`` — ``"head_to_head"`` / ``"spreads"`` /
                           ``"totals"`` (mapped from API key)
            selection      ``str`` — outcome name (team name or Over/Under)
            decimal_odds   ``float`` — decimal price
            last_update    ``str`` — ISO-8601 UTC last update time for this
                           market from this bookmaker
            ============== =================================================
        """
        events: List[Dict[str, Any]] = raw_response.get("data", [])
        rows: List[Dict[str, Any]] = []

        for event in events:
            home_team: str = event.get("home_team", "")
            away_team: str = event.get("away_team", "")
            commence_time: str = event.get("commence_time", "")
            fixture_label: str = f"{home_team} vs {away_team}"

            for bookmaker_obj in event.get("bookmakers", []):
                bookmaker_key: str = bookmaker_obj.get("key", "")
                for market_obj in bookmaker_obj.get("markets", []):
                    market_key: str = market_obj.get("key", "")
                    market_type: str = _MARKET_MAP.get(market_key, market_key)
                    last_update: str = market_obj.get("last_update", "")

                    for outcome in market_obj.get("outcomes", []):
                        selection: str = outcome.get("name", "")
                        price_raw = outcome.get("price")
                        try:
                            decimal_odds = float(price_raw)
                        except (TypeError, ValueError):
                            logger.warning(
                                "OddsAPI normalize_odds: unparseable price={} "
                                "event={} bookmaker={} market={}",
                                price_raw,
                                fixture_label,
                                bookmaker_key,
                                market_key,
                            )
                            continue

                        rows.append(
                            {
                                "fixture_label": fixture_label,
                                "home_team": home_team,
                                "away_team": away_team,
                                "commence_time": commence_time,
                                "bookmaker": bookmaker_key,
                                "market_type": market_type,
                                "selection": selection,
                                "decimal_odds": decimal_odds,
                                "last_update": last_update,
                            }
                        )

        logger.debug(
            "OddsAPI normalize_odds: events={} rows={}",
            len(events),
            len(rows),
        )
        return rows

    def match_to_fixture_id(
        self,
        home_team: str,
        away_team: str,
        fixtures_df: pd.DataFrame,
    ) -> Optional[int]:
        """Fuzzy-match team names to a fixture_id in *fixtures_df*.

        The DataFrame must have columns ``home_team_name``, ``away_team_name``
        and ``fixture_id``.  Matching is performed after lowercasing and
        stripping common suffixes/tokens (AFC, FC, etc.) plus punctuation.
        A match is found when both normalised team names appear as substrings
        of the corresponding normalised DataFrame column values (or vice
        versa).

        Parameters
        ----------
        home_team :
            Home team name from The Odds API.
        away_team :
            Away team name from The Odds API.
        fixtures_df :
            DataFrame of known fixtures, typically produced by
            :py:meth:`APISportsClient.normalize_fixtures`.

        Returns
        -------
        int or None
            The matching ``fixture_id``, or ``None`` if no confident match
            was found.
        """
        if fixtures_df.empty:
            logger.debug("OddsAPI match_to_fixture_id: fixtures_df is empty")
            return None

        norm_home = _normalise_team_name(home_team)
        norm_away = _normalise_team_name(away_team)

        for _, row in fixtures_df.iterrows():
            df_home = _normalise_team_name(str(row.get("home_team_name", "")))
            df_away = _normalise_team_name(str(row.get("away_team_name", "")))

            home_match = _names_match(norm_home, df_home)
            away_match = _names_match(norm_away, df_away)

            if home_match and away_match:
                fixture_id = row.get("fixture_id")
                logger.debug(
                    "OddsAPI match_to_fixture_id: matched '{}' vs '{}' → fixture_id={}",
                    home_team,
                    away_team,
                    fixture_id,
                )
                try:
                    return int(fixture_id)
                except (TypeError, ValueError):
                    logger.warning(
                        "OddsAPI match_to_fixture_id: fixture_id={} is not castable to int",
                        fixture_id,
                    )
                    return None

        logger.debug(
            "OddsAPI match_to_fixture_id: no match found for '{}' vs '{}'",
            home_team,
            away_team,
        )
        return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_quota_header(value: Optional[str]) -> Optional[int]:
        """Parse an integer from a quota response header value.

        Returns ``None`` if the header is absent or unparseable.
        """
        if value is None:
            return None
        try:
            return int(value)
        except ValueError:
            return None


# ---------------------------------------------------------------------------
# Module-level helpers for team name normalisation
# ---------------------------------------------------------------------------


def _normalise_team_name(name: str) -> str:
    """Return a normalised lowercase token string for fuzzy matching.

    Strips common football club suffixes/prefixes, punctuation, and extra
    whitespace so that e.g. ``"Port Adelaide FC"`` and ``"Port Adelaide"``
    normalise to the same string.
    """
    s = name.lower()
    # Remove punctuation (keep letters, digits, spaces)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    # Strip common tokens
    s = _STRIP_TOKENS.sub(" ", s)
    # Collapse whitespace
    s = _WHITESPACE.sub(" ", s).strip()
    return s


def _names_match(a: str, b: str) -> bool:
    """Return True when *a* and *b* are a confident substring match.

    The shorter string must be a substring of the longer string, or they
    must be equal.  Both strings should already be normalised via
    :func:`_normalise_team_name`.
    """
    if not a or not b:
        return False
    if a == b:
        return True
    # Substring matching in both directions
    return a in b or b in a
