"""Live-only data source manager: AFL Data Sports Group API → cache.

Every DataFrame returned carries ``_source_type`` and ``_fetch_ts`` columns
so downstream code can distinguish provenance.

NO demo mode. NO fallback to CSV files. If live data is unavailable → fail
with a clear, actionable error message.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
from loguru import logger

from app.core.config import settings


# ---------------------------------------------------------------------------
# Provenance constants
# ---------------------------------------------------------------------------
SOURCE_API = "afl_data_api"
SOURCE_CACHE = "afl_data_cache"
SOURCE_DERIVED = "derived"


def _stamp(df: pd.DataFrame, source_type: str) -> pd.DataFrame:
    """Add provenance columns to a DataFrame and return a copy."""
    df = df.copy()
    df["_source_type"] = source_type
    df["_fetch_ts"] = datetime.utcnow().isoformat()
    return df


# ---------------------------------------------------------------------------
# AFLDataProvider
# ---------------------------------------------------------------------------

class AFLDataProvider:
    """
    Fetches and normalises AFL Data Sports Group API data.

    Uses AFLDataClient (handles caching + retry) and AFLDataLoader
    to produce clean DataFrames.
    """

    def __init__(self) -> None:
        from app.data_ingestion.afl_data_loader import AFLDataLoader
        self._loader = AFLDataLoader()

    def get_fixtures(self) -> pd.DataFrame:
        df = self._loader.fixtures_df.copy()
        source = SOURCE_CACHE  # AFLDataLoader uses internal cache
        return _stamp(df, source)

    def get_upcoming_fixtures(self) -> pd.DataFrame:
        df = self._loader.load_upcoming_fixtures_df()
        return _stamp(df, SOURCE_CACHE)

    def get_teams(self) -> pd.DataFrame:
        df = self._loader.teams_df.copy()
        return _stamp(df, SOURCE_CACHE)

    def get_team_stats(self) -> pd.DataFrame:
        df = self._loader.team_stats_df.copy()
        return _stamp(df, SOURCE_CACHE)

    def get_player_stats(self) -> pd.DataFrame:
        df = self._loader.player_stats_df.copy()
        return _stamp(df, SOURCE_CACHE)

    def get_players(self) -> pd.DataFrame:
        df = self._loader.players_df.copy()
        return _stamp(df, SOURCE_CACHE)

    def is_available(self) -> bool:
        return self._loader._client.is_configured()


# ---------------------------------------------------------------------------
# DataSourceStatus
# ---------------------------------------------------------------------------

@dataclass
class DataSourceStatus:
    mode: str
    effective_mode: str
    afl_data_configured: bool
    afl_data_available: bool
    quota_status: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# DataSourceManager
# ---------------------------------------------------------------------------

class DataSourceManager:
    """Orchestrates the live AFL Data Sports Group data source.

    Priority (based on ``settings.effective_data_mode``):
      1. ``live``   — fetch from AFL Data API (uses cache, respects TTL)
      2. ``cache``  — only use cached responses

    If the data source is unavailable → raises a clear RuntimeError.
    There is no demo or CSV fallback. Ever.
    """

    def __init__(self) -> None:
        self._provider: Optional[AFLDataProvider] = None
        if settings.is_afl_data_configured:
            try:
                self._provider = AFLDataProvider()
            except Exception as exc:
                logger.error("Could not initialise AFLDataProvider: {}", exc)
                raise RuntimeError(
                    f"AFL Data provider failed to initialise: {exc}\n"
                    "Check your AFL_DATA_AUTHKEY setting in .env."
                ) from exc
        else:
            raise RuntimeError(
                "AFL_DATA_AUTHKEY is not configured. "
                "Live predictions require a valid AFL Data Sports Group API key.\n"
                "Add AFL_DATA_AUTHKEY=<your_key> to your .env file."
            )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> DataSourceStatus:
        return DataSourceStatus(
            mode=settings.data_mode,
            effective_mode=settings.effective_data_mode,
            afl_data_configured=settings.is_afl_data_configured,
            afl_data_available=self._provider is not None and self._provider.is_available(),
        )

    # ------------------------------------------------------------------
    # Fixtures / schedule
    # ------------------------------------------------------------------

    def get_upcoming_fixtures(self, season_id: Optional[str] = None) -> pd.DataFrame:
        """Return upcoming fixtures from AFL Data Sports Group API."""
        try:
            df = self._provider.get_upcoming_fixtures()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to fetch upcoming fixtures from AFL Data API: {exc}\n"
                "Check AFL_DATA_AUTHKEY and network connectivity."
            ) from exc

        if df.empty:
            raise RuntimeError(
                "AFL Data API returned an empty fixture list. "
                "The season may not have started yet, or the competition ID may be wrong.\n"
                f"Current AFL_DATA_COMPETITION_ID={settings.afl_data_competition_id}"
            )
        logger.info("Loaded {} upcoming fixtures from AFL Data API", len(df))
        return df

    def get_completed_fixtures(
        self,
        season_id: Optional[str] = None,
        lookback_days: int = 30,
    ) -> pd.DataFrame:
        """Return recently completed fixtures."""
        try:
            df = self._provider.get_fixtures()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to fetch completed fixtures from AFL Data API: {exc}"
            ) from exc

        if df.empty:
            return df

        if "status" in df.columns:
            return df[df["status"].isin(["completed"])].copy()
        return df

    # ------------------------------------------------------------------
    # Teams / players
    # ------------------------------------------------------------------

    def get_teams(self) -> pd.DataFrame:
        try:
            df = self._provider.get_teams()
            if not df.empty:
                return df
        except Exception as exc:
            raise RuntimeError(
                f"Failed to fetch teams from AFL Data API: {exc}"
            ) from exc
        raise RuntimeError("AFL Data API returned an empty teams response.")

    def get_players(self) -> pd.DataFrame:
        try:
            return self._provider.get_players()
        except Exception as exc:
            logger.warning("get_players failed: {} — returning empty DataFrame", exc)
            return pd.DataFrame()

    def get_player_stats(self) -> pd.DataFrame:
        try:
            return self._provider.get_player_stats()
        except Exception as exc:
            logger.warning("get_player_stats failed: {} — player markets disabled", exc)
            return pd.DataFrame()

    def get_team_stats(self) -> pd.DataFrame:
        try:
            return self._provider.get_team_stats()
        except Exception as exc:
            logger.warning("get_team_stats failed: {}", exc)
            return pd.DataFrame()

    def get_odds(self) -> pd.DataFrame:
        """Live bookmaker odds from The Odds API."""
        if not settings.is_odds_api_configured:
            logger.warning(
                "ODDS_API_KEY not configured — live odds unavailable."
            )
            return pd.DataFrame()
        try:
            from app.data_ingestion.odds_provider import OddsAPIProvider
            provider = OddsAPIProvider()
            return provider.odds_df
        except Exception as exc:
            logger.error("Failed to fetch live odds: {}", exc)
            return pd.DataFrame()


# Singleton for import convenience
_manager: Optional[DataSourceManager] = None


def get_data_source_manager() -> DataSourceManager:
    global _manager
    if _manager is None:
        _manager = DataSourceManager()
    return _manager
