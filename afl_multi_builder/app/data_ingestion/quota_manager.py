"""Quota manager for Sportradar API usage tracking.

Tracks every API request in a SQLite table ``quota_ledger`` and exposes
helpers for checking remaining quota, daily usage and warn thresholds.
Thread-safe via a threading.Lock.
"""
import sqlite3
import threading
import time
from datetime import datetime, timezone
from typing import Optional

from loguru import logger


def _db_path_from_url(database_url: str) -> str:
    """Extract filesystem path from a SQLite URL.

    Handles both ``sqlite:///./relative/path.db`` and
    ``sqlite:////absolute/path.db`` conventions.
    """
    if database_url.startswith("sqlite:///"):
        raw = database_url[len("sqlite:///"):]
        # An absolute path produces sqlite:////abs/... which strips to /abs/...
        # A relative path like ./data/x.db strips to ./data/x.db
        return raw
    # Fallback: treat the whole value as a path
    return database_url


class QuotaManager:
    """Tracks Sportradar API quota usage in the ``quota_ledger`` SQLite table.

    Parameters
    ----------
    database_url:
        SQLAlchemy-style SQLite URL, e.g. ``sqlite:///./data/afl_multi_builder.db``.
        The filesystem path is derived from the URL automatically.
    quota_total:
        Maximum number of API calls permitted by the current subscription plan.
    warn_pct:
        Fraction of ``quota_total`` at which a warning is emitted (e.g. 0.80).
    """

    _CREATE_TABLE_SQL = """
        CREATE TABLE IF NOT EXISTS quota_ledger (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     REAL    NOT NULL,
            endpoint      TEXT    NOT NULL,
            response_code INTEGER NOT NULL,
            latency_ms    REAL    NOT NULL
        )
    """

    _COUNT_ALL_SQL = "SELECT COUNT(*) FROM quota_ledger"
    _COUNT_SINCE_SQL = "SELECT COUNT(*) FROM quota_ledger WHERE timestamp >= ?"
    _INSERT_SQL = (
        "INSERT INTO quota_ledger (timestamp, endpoint, response_code, latency_ms) "
        "VALUES (?, ?, ?, ?)"
    )

    def __init__(
        self,
        database_url: Optional[str] = None,
        quota_total: int = 1000,
        warn_pct: float = 0.80,
    ) -> None:
        # Defer settings import to avoid import-time side-effects
        if database_url is None:
            from app.core.config import settings as _settings  # type: ignore[import]
            database_url = _settings.database_url
            quota_total = getattr(_settings, "api_quota_total", quota_total)
            warn_pct = getattr(_settings, "api_quota_warn_pct", warn_pct)

        self._db_path = _db_path_from_url(database_url)
        self.quota_total = quota_total
        self.warn_pct = warn_pct
        self._lock = threading.Lock()
        self._init_db()
        logger.debug(
            "QuotaManager initialised: db={} quota_total={} warn_pct={}",
            self._db_path,
            self.quota_total,
            self.warn_pct,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Return a new SQLite connection with WAL mode enabled."""
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        """Create the quota_ledger table if it does not already exist."""
        with self._lock:
            with self._connect() as conn:
                conn.execute(self._CREATE_TABLE_SQL)
                conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_request(
        self,
        endpoint: str,
        response_code: int,
        latency_ms: float,
    ) -> None:
        """Persist a single API request record to the ledger.

        Parameters
        ----------
        endpoint:
            The API endpoint path (without base URL), e.g. ``seasons/schedules.json``.
        response_code:
            HTTP response status code, e.g. 200, 429.
        latency_ms:
            Round-trip request latency in milliseconds.
        """
        ts = time.time()
        with self._lock:
            with self._connect() as conn:
                conn.execute(self._INSERT_SQL, (ts, endpoint, response_code, latency_ms))
                conn.commit()

        total = self.get_total_used()
        warn_threshold = int(self.quota_total * self.warn_pct)
        if total >= warn_threshold:
            logger.warning(
                "API quota warning: used={}/{} ({:.1f}%)",
                total,
                self.quota_total,
                (total / self.quota_total) * 100,
            )

    def get_total_used(self) -> int:
        """Return the total number of API requests ever recorded in the ledger."""
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(self._COUNT_ALL_SQL).fetchone()
                return int(row[0]) if row else 0

    def get_remaining(self) -> int:
        """Return how many API calls remain before the quota is exhausted."""
        return max(0, self.quota_total - self.get_total_used())

    def can_make_request(self) -> bool:
        """Return ``True`` when remaining quota exceeds 5 % of the total.

        The 5 % buffer prevents the very last calls from being blocked
        partway through a multi-step workflow.
        """
        return self.get_remaining() > self.quota_total * 0.05

    def get_daily_used(self) -> int:
        """Return the count of requests made in the last 24 hours."""
        since = time.time() - 86400.0
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(self._COUNT_SINCE_SQL, (since,)).fetchone()
                return int(row[0]) if row else 0

    def get_status(self) -> dict:
        """Return a status snapshot dictionary.

        Keys
        ----
        total_quota : int
            The configured maximum quota.
        used : int
            Total requests recorded in the ledger.
        remaining : int
            Calls still available (clamped to 0).
        warn_threshold : int
            The absolute count at which warnings are triggered.
        warn_triggered : bool
            Whether the current usage has crossed the warn threshold.
        daily_used : int
            Number of requests in the last 24 hours.
        """
        used = self.get_total_used()
        remaining = max(0, self.quota_total - used)
        warn_threshold = int(self.quota_total * self.warn_pct)
        return {
            "total_quota": self.quota_total,
            "used": used,
            "remaining": remaining,
            "warn_threshold": warn_threshold,
            "warn_triggered": used >= warn_threshold,
            "daily_used": self.get_daily_used(),
        }
