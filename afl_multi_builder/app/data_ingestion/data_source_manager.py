"""3-tier data source manager: live API → cache → demo CSV.

Every DataFrame returned carries ``_source_type`` and ``_fetch_ts`` columns
so downstream code can distinguish provenance.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from app.core.config import settings


# ---------------------------------------------------------------------------
# Provenance constants
# ---------------------------------------------------------------------------
SOURCE_API = "sportradar_api"
SOURCE_CACHE = "sportradar_cache"
SOURCE_DEMO = "demo"
SOURCE_DERIVED = "derived"


def _stamp(df: pd.DataFrame, source_type: str) -> pd.DataFrame:
    """Add provenance columns to a DataFrame in-place and return it."""
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
        return _stamp(pd.DataFrame(rows), SOURCE_API if data.get("_source") == "api_live" else SOURCE_CACHE)

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
# DemoDataProvider
# ---------------------------------------------------------------------------

class DemoDataProvider:
    """Loads demo CSV files from ``settings.demo_data_dir``."""

    def __init__(self) -> None:
        self._base = settings.demo_data_dir

    def _read(self, filename: str) -> pd.DataFrame:
        path = self._base / filename
        if not path.exists():
            logger.warning("Demo file not found: {}", path)
            return pd.DataFrame()
        df = pd.read_csv(path)
        return _stamp(df, SOURCE_DEMO)

    def get_fixtures(self) -> pd.DataFrame:
        return self._read("fixtures.csv")

    def get_teams(self) -> pd.DataFrame:
        return self._read("teams.csv")

    def get_players(self) -> pd.DataFrame:
        return self._read("players.csv")

    def get_player_stats(self) -> pd.DataFrame:
        return self._read("player_stats.csv")

    def get_team_stats(self) -> pd.DataFrame:
        return self._read("team_stats.csv")

    def get_odds(self) -> pd.DataFrame:
        return self._read("odds.csv")

    def get_venues(self) -> pd.DataFrame:
        return self._read("venues.csv")

    def is_available(self) -> bool:
        return (self._base / "fixtures.csv").exists()


# ---------------------------------------------------------------------------
# DataSourceManager
# ---------------------------------------------------------------------------

@dataclass
class DataSourceStatus:
    mode: str
    effective_mode: str
    sportradar_configured: bool
    sportradar_available: bool
    demo_available: bool
    quota_status: Optional[Dict[str, Any]] = None


class DataSourceManager:
    """Orchestrates the 3-tier data source hierarchy.

    Priority (based on ``settings.effective_data_mode``):
      1. ``live``   — fetch from Sportradar API (uses cache, respects TTL)
      2. ``cache``  — only use cached Sportradar responses
      3. ``demo``   — local demo CSV files

    If the requested source is unavailable and ``enable_demo_fallback`` is True,
    automatically falls back to demo.
    """

    def __init__(self) -> None:
        self._demo = DemoDataProvider()
        self._sportradar: Optional[SportradarDataProvider] = None
        if settings.is_sportradar_configured:
            try:
                self._sportradar = SportradarDataProvider()
            except Exception as exc:
                logger.warning("Could not initialise SportradarDataProvider: {}", exc)

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
            demo_available=self._demo.is_available(),
            quota_status=quota,
        )

    # ------------------------------------------------------------------
    # Fixtures / schedule
    # ------------------------------------------------------------------

    def get_upcoming_fixtures(self, season_id: Optional[str] = None) -> pd.DataFrame:
        """Return upcoming fixtures for the current season."""
        mode = settings.effective_data_mode

        if mode in ("live", "cache") and self._sportradar:
            sid = season_id or settings.sportradar_afl_season_id
            if sid:
                try:
                    df = self._sportradar.get_schedule(sid)
                    upcoming = df[df.get("status", pd.Series(["upcoming"] * len(df))) == "upcoming"]
                    if not upcoming.empty:
                        logger.info("Loaded {} upcoming fixtures from Sportradar", len(upcoming))
                        return upcoming
                except Exception as exc:
                    logger.warning("Sportradar schedule fetch failed: {}", exc)

        if settings.enable_demo_fallback or mode == "demo":
            logger.info("Using demo fixtures")
            df = self._demo.get_fixtures()
            if "status" in df.columns:
                return df[df["status"] == "upcoming"]
            return df

        return pd.DataFrame()

    def get_completed_fixtures(
        self,
        season_id: Optional[str] = None,
        lookback_days: int = 30,
    ) -> pd.DataFrame:
        """Return recently completed fixtures."""
        mode = settings.effective_data_mode

        if mode in ("live", "cache") and self._sportradar:
            sid = season_id or settings.sportradar_afl_season_id
            if sid:
                try:
                    df = self._sportradar.get_schedule(sid)
                    completed = df[df.get("status", pd.Series()) == "closed"]
                    if not completed.empty:
                        return completed
                except Exception as exc:
                    logger.warning("Sportradar completed fixtures fetch failed: {}", exc)

        if settings.enable_demo_fallback or mode == "demo":
            df = self._demo.get_fixtures()
            if "status" in df.columns:
                return df[df["status"] == "completed"]
            return df

        return pd.DataFrame()

    def get_match_summary(self, sport_event_id: str) -> pd.DataFrame:
        """Return detailed match summary (stats, timeline) for one fixture."""
        mode = settings.effective_data_mode

        if mode in ("live", "cache") and self._sportradar:
            try:
                return self._sportradar.get_match_summary(sport_event_id)
            except Exception as exc:
                logger.warning("Match summary fetch failed for {}: {}", sport_event_id, exc)

        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Teams / players
    # ------------------------------------------------------------------

    def get_teams(self) -> pd.DataFrame:
        mode = settings.effective_data_mode

        if mode in ("live", "cache") and self._sportradar:
            try:
                df = self._sportradar.get_teams()
                if not df.empty:
                    return df
            except Exception as exc:
                logger.warning("Teams fetch failed: {}", exc)

        return self._demo.get_teams()

    def get_players(self) -> pd.DataFrame:
        """Players are only available from demo data in trial API."""
        return self._demo.get_players()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_player_stats(self) -> pd.DataFrame:
        """Player stats — demo only for now (trial API has limited stat endpoints)."""
        return self._demo.get_player_stats()

    def get_team_stats(self) -> pd.DataFrame:
        return self._demo.get_team_stats()

    def get_odds(self) -> pd.DataFrame:
        """Odds data — demo only (odds not included in Sportradar trial)."""
        return self._demo.get_odds()

    # ------------------------------------------------------------------
    # Bulk load for pipeline
    # ------------------------------------------------------------------

    def load_all_demo(self) -> Dict[str, pd.DataFrame]:
        """Return all demo DataFrames keyed by name. Used by pipeline in demo mode."""
        return {
            "fixtures": self._demo.get_fixtures(),
            "teams": self._demo.get_teams(),
            "players": self._demo.get_players(),
            "player_stats": self._demo.get_player_stats(),
            "team_stats": self._demo.get_team_stats(),
            "odds": self._demo.get_odds(),
            "venues": self._demo.get_venues(),
        }


# Singleton for import convenience
_manager: Optional[DataSourceManager] = None


def get_data_source_manager() -> DataSourceManager:
    global _manager
    if _manager is None:
        _manager = DataSourceManager()
    return _manager
