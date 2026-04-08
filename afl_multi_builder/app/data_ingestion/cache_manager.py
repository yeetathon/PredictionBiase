"""Cache manager for Sportradar API responses.

Stores raw JSON API responses in a SQLite table ``api_response_cache``.
Provides get/set semantics with TTL-based staleness checks, hit-count
tracking, deterministic cache-key generation and bulk invalidation.
Thread-safe via a threading.Lock.
"""
import hashlib
import json
import sqlite3
import threading
import time
from typing import Any, Dict, Optional

from loguru import logger


def _db_path_from_url(database_url: str) -> str:
    """Extract filesystem path from a SQLite URL."""
    if database_url.startswith("sqlite:///"):
        return database_url[len("sqlite:///"):]
    return database_url


class CacheManager:
    """Manages a persistent SQLite cache for raw Sportradar API JSON responses.

    Parameters
    ----------
    database_url:
        SQLAlchemy-style SQLite URL.  If omitted the value from
        ``app.core.config.settings.database_url`` is used.
    default_ttl_seconds:
        Fallback TTL used when the caller does not supply one.
    """

    _CREATE_TABLE_SQL = """
        CREATE TABLE IF NOT EXISTS api_response_cache (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            cache_key       TEXT    NOT NULL UNIQUE,
            endpoint        TEXT    NOT NULL,
            params_hash     TEXT    NOT NULL,
            response_json   TEXT    NOT NULL,
            fetch_timestamp REAL    NOT NULL,
            ttl_seconds     INTEGER NOT NULL,
            hit_count       INTEGER NOT NULL DEFAULT 0
        )
    """

    _GET_SQL = (
        "SELECT response_json, fetch_timestamp, ttl_seconds, hit_count "
        "FROM api_response_cache WHERE cache_key = ?"
    )
    _UPDATE_HIT_SQL = (
        "UPDATE api_response_cache SET hit_count = hit_count + 1 "
        "WHERE cache_key = ?"
    )
    _UPSERT_SQL = """
        INSERT INTO api_response_cache
            (cache_key, endpoint, params_hash, response_json,
             fetch_timestamp, ttl_seconds, hit_count)
        VALUES (?, ?, ?, ?, ?, ?, 0)
        ON CONFLICT(cache_key) DO UPDATE SET
            endpoint        = excluded.endpoint,
            params_hash     = excluded.params_hash,
            response_json   = excluded.response_json,
            fetch_timestamp = excluded.fetch_timestamp,
            ttl_seconds     = excluded.ttl_seconds
    """
    _DELETE_ONE_SQL = "DELETE FROM api_response_cache WHERE cache_key = ?"
    _DELETE_ALL_SQL = "DELETE FROM api_response_cache"
    _STATS_SQL = (
        "SELECT COUNT(*), COALESCE(SUM(hit_count),0) FROM api_response_cache"
    )

    def __init__(
        self,
        database_url: Optional[str] = None,
        default_ttl_seconds: int = 21600,
    ) -> None:
        if database_url is None:
            from app.core.config import settings as _settings  # type: ignore[import]
            database_url = _settings.database_url
            default_ttl_seconds = getattr(
                _settings, "cache_ttl_hours", 6
            ) * 3600

        self._db_path = _db_path_from_url(database_url)
        self.default_ttl_seconds = default_ttl_seconds
        self._lock = threading.Lock()
        self._init_db()
        logger.debug(
            "CacheManager initialised: db={} default_ttl={}s",
            self._db_path,
            self.default_ttl_seconds,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(self._CREATE_TABLE_SQL)
                conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def make_key(endpoint: str, params: Optional[Dict[str, Any]] = None) -> str:
        """Build a deterministic, stable cache key from endpoint and params.

        Parameters
        ----------
        endpoint:
            The API endpoint path, e.g. ``seasons/sr:season:118700/schedules.json``.
        params:
            Optional query-string parameters dict.  Keys are sorted before
            hashing so parameter order does not matter.

        Returns
        -------
        str
            A 40-character SHA-1 hex digest that uniquely identifies the request.
        """
        normalised_params: Dict[str, Any] = {}
        if params:
            # Exclude api_key from the cache key to avoid key-rotation cache misses
            normalised_params = {
                k: v for k, v in sorted(params.items()) if k != "api_key"
            }
        raw = json.dumps({"endpoint": endpoint, "params": normalised_params}, sort_keys=True)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def is_stale(self, cache_key: str, ttl_seconds: Optional[int] = None) -> bool:
        """Return ``True`` when the cached entry is absent or older than *ttl_seconds*.

        Parameters
        ----------
        cache_key:
            Key previously returned by :py:meth:`make_key`.
        ttl_seconds:
            Age limit in seconds.  Falls back to ``default_ttl_seconds`` when
            ``None``.
        """
        effective_ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT fetch_timestamp FROM api_response_cache WHERE cache_key = ?",
                    (cache_key,),
                ).fetchone()
        if row is None:
            return True
        age = time.time() - float(row[0])
        return age > effective_ttl

    def get(
        self,
        cache_key: str,
        ttl_seconds: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve a cached response if it exists and is not stale.

        On a cache hit the ``hit_count`` for the entry is atomically
        incremented.

        Parameters
        ----------
        cache_key:
            Key returned by :py:meth:`make_key`.
        ttl_seconds:
            Override the stored TTL for staleness evaluation.

        Returns
        -------
        dict or None
            The parsed JSON payload, or ``None`` on a miss / stale entry.
        """
        effective_ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(self._GET_SQL, (cache_key,)).fetchone()
                if row is None:
                    logger.debug("Cache miss (absent): key={}", cache_key)
                    return None
                response_json_str, fetch_timestamp, stored_ttl, hit_count = row
                age = time.time() - float(fetch_timestamp)
                check_ttl = effective_ttl if ttl_seconds is not None else int(stored_ttl)
                if age > check_ttl:
                    logger.debug(
                        "Cache miss (stale): key={} age={:.0f}s ttl={}s",
                        cache_key,
                        age,
                        check_ttl,
                    )
                    return None
                # Valid hit — increment counter
                conn.execute(self._UPDATE_HIT_SQL, (cache_key,))
                conn.commit()

        logger.debug(
            "Cache hit: key={} age={:.0f}s hits={}",
            cache_key,
            age,
            hit_count + 1,
        )
        try:
            return json.loads(response_json_str)
        except json.JSONDecodeError as exc:
            logger.error("Cache entry contains invalid JSON for key={}: {}", cache_key, exc)
            return None

    def set(
        self,
        cache_key: str,
        endpoint: str,
        params_hash: str,
        data: Dict[str, Any],
        ttl_seconds: Optional[int] = None,
    ) -> None:
        """Upsert a response payload into the cache.

        Parameters
        ----------
        cache_key:
            Unique identifier for this request.
        endpoint:
            Human-readable endpoint path stored for diagnostic purposes.
        params_hash:
            Hex digest of the request parameters (for debugging).
        data:
            Parsed JSON dict to store.  Will be serialised to a JSON string.
        ttl_seconds:
            How long the entry is considered fresh.
        """
        effective_ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds
        response_json_str = json.dumps(data, separators=(",", ":"))
        fetch_ts = time.time()
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    self._UPSERT_SQL,
                    (cache_key, endpoint, params_hash, response_json_str, fetch_ts, effective_ttl),
                )
                conn.commit()
        logger.debug("Cache set: key={} endpoint={} ttl={}s", cache_key, endpoint, effective_ttl)

    def invalidate(self, cache_key: Optional[str] = None) -> None:
        """Delete one or all entries from the cache.

        Parameters
        ----------
        cache_key:
            If supplied, only that entry is removed.  If ``None``, the
            entire cache table is cleared.
        """
        with self._lock:
            with self._connect() as conn:
                if cache_key is None:
                    conn.execute(self._DELETE_ALL_SQL)
                    logger.info("Cache invalidated: all entries removed")
                else:
                    conn.execute(self._DELETE_ONE_SQL, (cache_key,))
                    logger.debug("Cache invalidated: key={}", cache_key)
                conn.commit()

    def get_stats(self) -> Dict[str, Any]:
        """Return cache statistics.

        Returns
        -------
        dict
            ``total_entries`` – number of rows in the cache table.
            ``total_hits``    – cumulative hit counter across all entries.
            ``stale_entries`` – count of entries whose age exceeds their stored TTL.
        """
        now = time.time()
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(self._STATS_SQL).fetchone()
                total_entries = int(row[0]) if row else 0
                total_hits = int(row[1]) if row else 0
                stale_count = conn.execute(
                    "SELECT COUNT(*) FROM api_response_cache "
                    "WHERE (? - fetch_timestamp) > ttl_seconds",
                    (now,),
                ).fetchone()
                stale_entries = int(stale_count[0]) if stale_count else 0

        return {
            "total_entries": total_entries,
            "total_hits": total_hits,
            "stale_entries": stale_entries,
        }
