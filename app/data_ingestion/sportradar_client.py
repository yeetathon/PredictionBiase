"""Sportradar AFL API HTTP client.

Wraps the Sportradar AFL v2 REST API with:
- Token-bucket rate limiting (minimum 1.1 s between requests)
- Transparent cache layer via CacheManager
- Quota enforcement via QuotaManager
- Retry with exponential back-off on 429 / 5xx responses
- Structured logging at DEBUG / WARNING levels
"""
from __future__ import annotations

import time
import threading
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class SportradarAPIError(Exception):
    """Raised when the Sportradar API returns an unrecoverable error.

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
            f"SportradarAPIError(status_code={self.status_code!r}, "
            f"endpoint={self.endpoint!r}, message={self.message!r})"
        )


# ---------------------------------------------------------------------------
# Token-bucket rate limiter (thread-safe)
# ---------------------------------------------------------------------------

class _TokenBucket:
    """Simple token-bucket that enforces a minimum inter-request gap.

    The bucket refills at *rate_qps* tokens per second.  Each call to
    ``consume()`` blocks until a token is available.
    """

    def __init__(self, rate_qps: float, min_interval_seconds: float = 1.1) -> None:
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
            # Refill proportionally to elapsed time, capped at 1 token
            self._tokens = min(1.0, self._tokens + elapsed * self._rate)
            if self._tokens < 1.0:
                # Wait until we have a token
                wait = (1.0 - self._tokens) / self._rate
                time.sleep(wait)
                self._tokens = 0.0
            else:
                self._tokens -= 1.0
            # Additionally enforce the hard minimum interval
            since_last = time.monotonic() - (self._last_check - elapsed)
            if since_last < self._min_interval:
                time.sleep(self._min_interval - since_last)


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------

class SportradarClient:
    """HTTP client for the Sportradar AFL v2 API.

    Parameters
    ----------
    api_key : str
        Sportradar API key.  An empty string means the client is not
        configured; :py:meth:`is_configured` returns ``False``.
    base_url : str
        Base URL including locale suffix, e.g.
        ``https://api.sportradar.com/afl/trial/v2/en``.
    rate_limit_qps : float
        Maximum requests per second (default 1.0).  A hard minimum of
        1.1 seconds between requests is also enforced.
    quota_manager : QuotaManager | None
        Optional quota tracker.  When provided, every response is
        recorded and quota exhaustion stops further requests.
    cache_manager : CacheManager | None
        Optional response cache.  When provided, successful responses
        are stored and returned on subsequent identical requests.
    timeout : int
        HTTP request timeout in seconds (default 30).
    """

    _RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
    _BACKOFF_SECONDS = (2, 4, 8)

    def __init__(
        self,
        api_key: str,
        base_url: str,
        rate_limit_qps: float = 1.0,
        quota_manager: Optional[Any] = None,
        cache_manager: Optional[Any] = None,
        timeout: int = 30,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._quota_manager = quota_manager
        self._cache_manager = cache_manager
        self._timeout = timeout
        self._bucket = _TokenBucket(
            rate_qps=rate_limit_qps,
            min_interval_seconds=max(1.1, 1.0 / rate_limit_qps) if rate_limit_qps > 0 else 1.1,
        )
        self._session = requests.Session()
        self._session.headers.update({
            "Accept": "application/json",
            "User-Agent": "AFL-PredictionBiase/1.0",
            # Sportradar v3 uses x-api-key header auth (NOT a query param)
            "x-api-key": api_key,
        })

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def is_configured(self) -> bool:
        """Return ``True`` when an API key has been provided."""
        return bool(self._api_key and self._api_key.strip())

    # ------------------------------------------------------------------
    # Core request method
    # ------------------------------------------------------------------

    def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        cache_ttl_seconds: int = 21600,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """Fetch *endpoint* from the Sportradar API (or cache).

        Parameters
        ----------
        endpoint :
            Path relative to *base_url*, e.g.
            ``seasons/sr:season:118700/schedules.json``.
        params :
            Additional query-string parameters.  ``api_key`` is injected
            automatically and must NOT be included here.
        cache_ttl_seconds :
            Maximum age (seconds) of a cached response before it is
            considered stale and re-fetched.  Defaults to 6 hours.
        force_refresh :
            When ``True``, bypass the cache and always fetch from the API.

        Returns
        -------
        dict
            Parsed JSON response, augmented with:
            - ``_source``: ``"cache"`` or ``"api_live"``
            - ``_fetch_timestamp``: ISO-8601 UTC string (only on live fetch)

        Raises
        ------
        SportradarAPIError
            If the API returns an error after all retries are exhausted,
            or if the quota is exceeded.
        """
        params = dict(params or {})

        # ---- 1. Cache lookup ------------------------------------------------
        if self._cache_manager is not None and not force_refresh:
            cache_key = self._cache_manager.make_key(endpoint, params)
            cached = self._cache_manager.get(cache_key, ttl_seconds=cache_ttl_seconds)
            if cached is not None:
                logger.debug(
                    "Cache HIT endpoint=%s key=%s", endpoint, cache_key
                )
                result = dict(cached)
                result["_source"] = "cache"
                return result
            logger.debug("Cache MISS endpoint=%s key=%s", endpoint, cache_key)
        else:
            cache_key = (
                self._cache_manager.make_key(endpoint, params)
                if self._cache_manager is not None
                else None
            )

        # ---- 2. Quota check -------------------------------------------------
        if self._quota_manager is not None:
            if not self._quota_manager.can_make_request():
                status = self._quota_manager.get_status()
                logger.warning(
                    "Quota exhausted: used=%d/%d",
                    status.get("used", "?"),
                    status.get("total_quota", "?"),
                )
                raise SportradarAPIError(
                    message=(
                        f"API quota exhausted: used={status.get('used')} of "
                        f"{status.get('total_quota')}"
                    ),
                    endpoint=endpoint,
                    status_code=None,
                )
            # Warn if approaching limit
            status = self._quota_manager.get_status()
            if status.get("warn_triggered"):
                logger.warning(
                    "API quota warning: used=%d/%d remaining=%d",
                    status.get("used", 0),
                    status.get("total_quota", 0),
                    status.get("remaining", 0),
                )

        # ---- 3. Rate limit --------------------------------------------------
        self._bucket.consume()

        # ---- 4. HTTP request with retry -------------------------------------
        # v3 uses x-api-key header (set in __init__); do NOT inject api_key as
        # a query param — it is not recognised and causes 403 on v3 endpoints.
        url = f"{self._base_url}/{endpoint}"
        request_params = dict(params)  # no api_key in params for v3

        last_exc: Optional[Exception] = None
        last_status: Optional[int] = None
        t_start = time.monotonic()

        for attempt, backoff in enumerate([0] + list(self._BACKOFF_SECONDS)):
            if backoff > 0:
                logger.debug(
                    "Retry attempt=%d endpoint=%s backoff=%ds",
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
                    "HTTP %d endpoint=%s attempt=%d latency=%.1fms",
                    response.status_code,
                    endpoint,
                    attempt,
                    latency_ms,
                )

                if response.status_code == 200:
                    # Success path
                    fetch_ts = datetime.now(timezone.utc).isoformat()
                    try:
                        data: Dict[str, Any] = response.json()
                    except ValueError as parse_exc:
                        raise SportradarAPIError(
                            message=f"Failed to parse JSON response: {parse_exc}",
                            endpoint=endpoint,
                            status_code=200,
                        ) from parse_exc

                    # Record quota usage
                    if self._quota_manager is not None:
                        self._quota_manager.record_request(
                            endpoint=endpoint,
                            response_code=200,
                            latency_ms=latency_ms,
                        )

                    # Store in cache
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
                    return data

                elif response.status_code in self._RETRY_STATUSES:
                    # Record failed request in quota ledger
                    if self._quota_manager is not None:
                        self._quota_manager.record_request(
                            endpoint=endpoint,
                            response_code=response.status_code,
                            latency_ms=latency_ms,
                        )
                    last_exc = SportradarAPIError(
                        message=f"HTTP {response.status_code}: {response.text[:200]}",
                        endpoint=endpoint,
                        status_code=response.status_code,
                    )
                    # Continue to next retry
                else:
                    # Non-retryable error (4xx other than 429)
                    if self._quota_manager is not None:
                        self._quota_manager.record_request(
                            endpoint=endpoint,
                            response_code=response.status_code,
                            latency_ms=latency_ms,
                        )

                    if response.status_code == 403:
                        # Log everything so the cause is obvious in the terminal
                        logger.error(
                            "Sportradar 403 Forbidden — endpoint=%s full_url=%s",
                            endpoint, response.url,
                        )
                        logger.error(
                            "Sportradar 403 response body: %s", response.text[:500]
                        )
                        raise SportradarAPIError(
                            message=(
                                "Sportradar returned 403 Forbidden.\n"
                                f"  Endpoint called : {endpoint}\n"
                                f"  Full URL        : {response.url}\n"
                                "Likely causes:\n"
                                "  1. API key is invalid or expired — check SPORTRADAR_API_KEY in .env\n"
                                "  2. Endpoint not available on your subscription tier\n"
                                "  3. Wrong base URL — trial keys require "
                                "https://api.sportradar.com/afl/trial/v3\n"
                                f"  Response: {response.text[:200]}"
                            ),
                            endpoint=endpoint,
                            status_code=403,
                        )

                    raise SportradarAPIError(
                        message=(
                            f"HTTP {response.status_code} (non-retryable): "
                            f"{response.text[:200]}"
                        ),
                        endpoint=endpoint,
                        status_code=response.status_code,
                    )

            except SportradarAPIError:
                raise
            except requests.RequestException as req_exc:
                latency_ms = (time.monotonic() - t_start) * 1000.0
                logger.debug(
                    "Request error endpoint=%s attempt=%d error=%s",
                    endpoint,
                    attempt,
                    req_exc,
                )
                last_exc = SportradarAPIError(
                    message=f"Request failed: {req_exc}",
                    endpoint=endpoint,
                    status_code=None,
                )

        # All retries exhausted
        total_latency_ms = (time.monotonic() - t_start) * 1000.0
        logger.warning(
            "All retries exhausted endpoint=%s last_status=%s total_latency=%.0fms",
            endpoint,
            last_status,
            total_latency_ms,
        )
        if last_exc is not None:
            raise last_exc
        raise SportradarAPIError(
            message="All retries exhausted with no specific error recorded",
            endpoint=endpoint,
            status_code=last_status,
        )
