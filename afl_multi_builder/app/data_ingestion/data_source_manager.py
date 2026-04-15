"""Live-only data source manager: live API → cache.

Every DataFrame returned carries ``_source_type`` and ``_fetch_ts`` columns
so downstream code can distinguish provenance.

NO demo mode. NO fallback to CSV files. If live data is unavailable → fail
with a clear, actionable error message.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from app.core.config import settings


# ---------------------------------------------------------------------------
# Provenance constants
# ---------------------------------------------------------------------------
SOURCE_API = "sportradar_api"
SOURCE_CACHE = "sportradar_cache"
SOURCE_DERIVED = "derived"


def _stamp(df: pd.DataFrame, source_type: str) -> pd.DataFrame:
    """Add provenance columns to a DataFrame and return a copy."""
    df = df.copy()
    df["_source_type"] = source_type
    df["_fetch_ts"] = datetime.utcnow().isoformat()
    return df


# ---------------------------------------------------------------------------
# SportradarDataProvider
# ---------------------------------------------------------------------------

class SportradarDataProvider:
    """Fetches and normalises Sportradar AFL API data.

    Uses the SportradarClient (handles rate-limiting / caching / quota) and
    SportradarNormalizer to produce clean DataFrames.
    """

    def __init__(self) -> None:
        from app.data_ingestion.sportradar_client import SportradarClient
        from app.data_ingestion.sportradar_normalizer import SportradarNormalizer

        self._client = SportradarClient()
        self._norm = SportradarNormalizer()

    # ------------------------------------------------------------------
    # High-level fetch methods
    # ------------------------------------------------------------------

    def get_seasons(self) -> pd.DataFrame:
        competition_id = settings.sportradar_afl_competition_id
        endpoint = f"competitions/{competition_id}/seasons.json"
        data = self._client.get(endpoint, cache_ttl_seconds=86400)
        rows = self._norm.normalize_seasons(data)
        src = SOURCE_API if data.get("_source") == "api_live" else SOURCE_CACHE
        return _stamp(pd.DataFrame(rows), src)

    def get_schedule(self, season_id: str) -> pd.DataFrame:
        endpoint = f"seasons/{season_id}/schedules.json"
        data = self._client.get(endpoint, cache_ttl_seconds=settings.cache_ttl_upcoming_hours * 3600)
        rows = self._norm.normalize_schedule(data)
        src = SOURCE_API if data.get("_source") == "api_live" else SOURCE_CACHE
        return _stamp(pd.DataFrame(rows), src)

    def get_match_summary(self, sport_event_id: str) -> pd.DataFrame:
        endpoint = f"sport_events/{sport_event_id}/summary.json"
        data = self._client.get(endpoint, cache_ttl_seconds=settings.cache_ttl_results_hours * 3600)
        rows = self._norm.normalize_match_summary(data)
        src = SOURCE_API if data.get("_source") == "api_live" else SOURCE_CACHE
        return _stamp(pd.DataFrame(rows) if rows else pd.DataFrame(), src)

    def get_teams(self) -> pd.DataFrame:
        competition_id = settings.sportradar_afl_competition_id
        endpoint = f"competitions/{competition_id}/teams.json"
        data = self._client.get(endpoint, cache_ttl_seconds=86400 * 7)
        rows = self._norm.normalize_teams(data)
        src = SOURCE_API if data.get("_source") == "api_live" else SOURCE_CACHE
        return _stamp(pd.DataFrame(rows), src)

    def is_available(self) -> bool:
        return self._client.is_configured()


# ---------------------------------------------------------------------------
# DataSourceStatus
# ---------------------------------------------------------------------------

@dataclass
class DataSourceStatus:
    mode: str
    effective_mode: str
    sportradar_configured: bool
    sportradar_available: bool
    quota_status: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# DataSourceManager
# ---------------------------------------------------------------------------

class DataSourceManager:
    """Orchestrates the 2-tier live data source hierarchy.

    Priority (based on ``settings.effective_data_mode``):
      1. ``live``   — fetch from Sportradar API (uses cache, respects TTL)
      2. ``cache``  — only use cached Sportradar responses

    If the requested source is unavailable → raises a clear RuntimeError.
    There is no demo or CSV fallback. Ever.
    """

    def __init__(self) -> None:
        self._sportradar: Optional[SportradarDataProvider] = None
        if settings.is_sportradar_configured:
            try:
                self._sportradar = SportradarDataProvider()
            except Exception as exc:
                logger.error("Could not initialise SportradarDataProvider: {}", exc)
                raise RuntimeError(
                    f"Sportradar data provider failed to initialise: {exc}\n"
                    "Check your SPORTRADAR_API_KEY and SPORTRADAR_BASE_URL settings."
                ) from exc
        else:
            raise RuntimeError(
                "SPORTRADAR_API_KEY is not configured. "
                "Live predictions require a valid Sportradar API key.\n"
                "Add SPORTRADAR_API_KEY=<your_key> to your .env file.\n"
                "Get a key at https://developer.sportradar.com/"
            )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> DataSourceStatus:
        quota = None
        if self._sportradar is not None:
            try:
                from app.data_ingestion.quota_manager import QuotaManager
                quota = QuotaManager().get_status()
            except Exception:
                pass

        return DataSourceStatus(
            mode=settings.data_mode,
            effective_mode=settings.effective_data_mode,
            sportradar_configured=settings.is_sportradar_configured,
            sportradar_available=self._sportradar is not None and self._sportradar.is_available(),
            quota_status=quota,
        )

    # ------------------------------------------------------------------
    # Fixtures / schedule
    # ------------------------------------------------------------------

    def get_upcoming_fixtures(self, season_id: Optional[str] = None) -> pd.DataFrame:
        """Return upcoming fixtures for the current season from Sportradar."""
        sid = season_id or settings.sportradar_afl_season_id
        if not sid:
            raise RuntimeError(
                "SPORTRADAR_AFL_SEASON_ID is not configured. "
                "Set it in your .env file to fetch upcoming fixtures.\n"
                "Find the current season ID via the Sportradar seasons endpoint."
            )

        try:
            df = self._sportradar.get_schedule(sid)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to fetch schedule for season {sid} from Sportradar: {exc}\n"
                "Check API connectivity, API key, and season ID."
            ) from exc

        if df.empty:
            raise RuntimeError(
                f"Sportradar returned an empty schedule for season {sid}. "
                "The season may be over, or the season ID may be incorrect.\n"
                "Verify SPORTRADAR_AFL_SEASON_ID in your .env file."
            )

        if "status" in df.columns:
            upcoming = df[df["status"].isin(["not_started", "upcoming"])]
            if not upcoming.empty:
                logger.info("Loaded {} upcoming fixtures from Sportradar", len(upcoming))
                return upcoming
            # If no "upcoming" rows, return full schedule with a warning
            logger.warning(
                "No fixtures with status 'not_started'/'upcoming' in season {}. "
                "Returning full schedule ({} rows) — check if season is active.",
                sid, len(df),
            )

        logger.info("Loaded {} fixtures from Sportradar (season={})", len(df), sid)
        return df

    def get_completed_fixtures(
        self,
        season_id: Optional[str] = None,
        lookback_days: int = 30,
    ) -> pd.DataFrame:
        """Return recently completed fixtures from Sportradar."""
        sid = season_id or settings.sportradar_afl_season_id
        if not sid:
            raise RuntimeError(
                "SPORTRADAR_AFL_SEASON_ID is not configured. "
                "Set it in your .env file."
            )

        try:
            df = self._sportradar.get_schedule(sid)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to fetch completed fixtures for season {sid}: {exc}"
            ) from exc

        if df.empty:
            return df

        if "status" in df.columns:
            return df[df["status"].isin(["closed", "completed", "ended"])]
        return df

    def get_match_summary(self, sport_event_id: str) -> pd.DataFrame:
        """Return detailed match summary for one fixture from Sportradar."""
        try:
            return self._sportradar.get_match_summary(sport_event_id)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to fetch match summary for {sport_event_id}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Teams / players
    # ------------------------------------------------------------------

    def get_teams(self) -> pd.DataFrame:
        try:
            df = self._sportradar.get_teams()
            if not df.empty:
                return df
        except Exception as exc:
            raise RuntimeError(
                f"Failed to fetch teams from Sportradar: {exc}\n"
                "Check API key and connectivity."
            ) from exc

        raise RuntimeError(
            "Sportradar returned an empty teams response. "
            "Check your competition ID and API plan permissions."
        )

    def get_players(self) -> pd.DataFrame:
        """
        Player roster data.

        The Sportradar trial plan does not expose individual player roster
        endpoints. If the live API supports it, it will be returned; otherwise
        an empty DataFrame is returned and player-props markets are disabled.
        """
        logger.warning(
            "Player roster endpoint not available via Sportradar trial plan. "
            "Player-prop markets will be disabled."
        )
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_player_stats(self) -> pd.DataFrame:
        """
        Player statistics.

        These must come from live match summaries fetched via Sportradar.
        If not available → returns empty DataFrame (player markets disabled).
        """
        logger.warning(
            "Player stats not available from Sportradar trial plan. "
            "Player-prop markets will be disabled for this session."
        )
        return pd.DataFrame()

    def get_team_stats(self) -> pd.DataFrame:
        """
        Team statistics derived from completed match summaries.
        """
        logger.warning(
            "Direct team stats endpoint not available via current plan. "
            "Team stats will be derived from match summaries."
        )
        return pd.DataFrame()

    def get_odds(self) -> pd.DataFrame:
        """
        Live bookmaker odds from The Odds API.

        Returns empty DataFrame if ODDS_API_KEY is not configured —
        head-to-head legs will use Elo-derived probabilities only.
        """
        if not settings.is_odds_api_configured:
            logger.warning(
                "ODDS_API_KEY not configured — live odds unavailable. "
                "Edge calculations will use model probability only."
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
