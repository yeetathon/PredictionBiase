"""
Main data loader: aggregates provider data into DataFrames for modelling.

Data source hierarchy:
  1. AFL Data Sports Group API  → fixtures, schedule, scores, full AFL stats (primary)
  2. The Odds API               → real bookmaker odds for edge calculation (when configured)
  3. Edge Intelligence          → scraped news/injury signals (when scraping enabled)

Tests inject providers explicitly:
    DataLoader(afl_provider=MockAFLDataProvider(...))
"""
from __future__ import annotations

import pandas as pd
from typing import Optional
from loguru import logger

from app.core.config import settings


# ---------------------------------------------------------------------------
# Null providers (used when an optional source is not configured)
# ---------------------------------------------------------------------------

class _NullOddsProvider:
    @property
    def odds_df(self) -> pd.DataFrame:
        return pd.DataFrame()

    def get_odds(self, *args, **kwargs):
        return []

    def get_best_odds(self, *args, **kwargs):
        return None


class _NullWeatherProvider:
    @property
    def weather_df(self) -> pd.DataFrame:
        return pd.DataFrame()

    def get_weather(self, *args, **kwargs):
        return None


class _NullInjuryProvider:
    @property
    def injuries_df(self) -> pd.DataFrame:
        return pd.DataFrame()

    def get_injuries(self, *args, **kwargs):
        return []


# ---------------------------------------------------------------------------
# Provider factories
# ---------------------------------------------------------------------------

def _make_afl_provider():
    """
    Build the primary AFL data provider.
    AFL Data Sports Group API is required; raises RuntimeError if not configured.
    """
    if not settings.is_afl_data_configured:
        raise RuntimeError(
            "AFL_DATA_AUTHKEY is not configured. "
            "Add AFL_DATA_AUTHKEY=<your_key> to your .env file."
        )
    from app.data_ingestion.afl_data_loader import AFLDataLoader
    provider = AFLDataLoader()
    logger.info("DataLoader: using AFLDataLoader (mode=%s)", settings.effective_data_mode)
    return provider


def _make_odds_provider(fixtures_df: Optional[pd.DataFrame] = None):
    """
    Build the odds provider.
    Uses The Odds API when configured; falls back to null provider.
    """
    if settings.is_odds_api_configured:
        try:
            from app.data_ingestion.odds_provider import OddsAPIProvider
            provider = OddsAPIProvider(fixtures_df=fixtures_df)
            logger.info("DataLoader: using OddsAPIProvider (live bookmaker odds)")
            return provider
        except Exception as e:
            logger.warning("DataLoader: OddsAPIProvider init failed: %s — using null odds", e)
    else:
        logger.info("DataLoader: ODDS_API_KEY not configured — running without market odds")
    return _NullOddsProvider()


def _make_edge_provider():
    """Build the edge intelligence provider if scraping is enabled."""
    if settings.enable_scraping:
        try:
            from app.data_ingestion.edge_intelligence import EdgeIntelligenceService
            svc = EdgeIntelligenceService()
            logger.info("DataLoader: EdgeIntelligenceService enabled")
            return svc
        except Exception as e:
            logger.warning("DataLoader: EdgeIntelligenceService init failed: %s", e)
    return None


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------

class DataLoader:
    """
    Aggregates data from all providers into analysis-ready DataFrames.

    In production: uses AFLDataLoader (primary) + OddsAPIProvider + EdgeIntelligence.
    In tests: inject providers explicitly.
    """

    def __init__(
        self,
        afl_provider=None,
        odds_provider=None,
        weather_provider=None,
        injury_provider=None,
        edge_provider=None,
    ):
        self.afl = afl_provider or _make_afl_provider()
        self.weather = weather_provider or _NullWeatherProvider()
        self.injuries = injury_provider or _NullInjuryProvider()

        # Odds provider needs fixture data to map team names → fixture IDs
        if odds_provider is not None:
            self.odds = odds_provider
        else:
            # Pass fixtures_df so OddsAPIProvider can match games to fixture IDs
            try:
                fx = self.afl.fixtures_df if hasattr(self.afl, "fixtures_df") else pd.DataFrame()
            except Exception:
                fx = pd.DataFrame()
            self.odds = _make_odds_provider(fixtures_df=fx)

        # Edge intelligence (optional scraped signals)
        self.edge = edge_provider if edge_provider is not None else _make_edge_provider()

    # ------------------------------------------------------------------
    # Core load methods
    # ------------------------------------------------------------------

    def load_fixtures_df(self) -> pd.DataFrame:
        """All fixtures (completed + upcoming) — used by feature engineering."""
        return self.afl.fixtures_df.copy()

    def load_upcoming_fixtures_df(self) -> pd.DataFrame:
        """
        Future fixtures only (scheduled_utc > now).
        Delegates to provider's strict datetime filter when available.
        """
        if hasattr(self.afl, "load_upcoming_fixtures_df"):
            return self.afl.load_upcoming_fixtures_df()

        # Fallback for test-injected demo providers
        fx = self.afl.fixtures_df.copy()
        fx["status"] = fx["status"].str.strip()
        return fx[fx["status"] == "upcoming"]

    def load_team_stats_df(self) -> pd.DataFrame:
        return self.afl.team_stats_df.copy()

    def load_player_stats_df(self) -> pd.DataFrame:
        return self.afl.player_stats_df.copy()

    def load_odds_df(self) -> pd.DataFrame:
        """
        Load bookmaker odds. Returns real Odds API data when configured,
        empty DataFrame otherwise. Logs source clearly.
        """
        df = self.odds.odds_df.copy()
        if not df.empty:
            logger.info(
                "DataLoader.load_odds_df: %d rows from %s",
                len(df), type(self.odds).__name__,
            )
        else:
            logger.debug("DataLoader.load_odds_df: no odds available")
        return df

    def load_weather_df(self) -> pd.DataFrame:
        return self.weather.weather_df.copy()

    def load_injuries_df(self) -> pd.DataFrame:
        return self.injuries.injuries_df.copy()

    def load_teams_df(self) -> pd.DataFrame:
        return self.afl.teams_df.copy()

    def load_players_df(self) -> pd.DataFrame:
        return self.afl.players_df.copy()

    def load_edge_signals(self, home_team: str = "", away_team: str = "") -> list:
        """
        Load scraped edge intelligence signals for a fixture.
        Returns list of EdgeSignal objects (or empty list if scraping disabled).
        """
        if self.edge is None:
            return []
        try:
            if home_team or away_team:
                return self.edge.get_fixture_signals(home_team, away_team)
            return self.edge.fetch_signals()
        except Exception as e:
            logger.warning("DataLoader.load_edge_signals failed: %s", e)
            return []

    def load_all_edge_signals(self) -> list:
        """Load all current edge signals."""
        if self.edge is None:
            return []
        try:
            return self.edge.fetch_signals()
        except Exception as e:
            logger.warning("DataLoader.load_all_edge_signals failed: %s", e)
            return []

    # ------------------------------------------------------------------
    # Data source status
    # ------------------------------------------------------------------

    def get_source_status(self) -> dict:
        """Return a dict summarising which data sources are active."""
        status = {
            "afl_data": {
                "active": isinstance(self.afl, object) and hasattr(self.afl, "_client"),
                "type": type(self.afl).__name__,
            },
            "odds_api": {
                "active": settings.is_odds_api_configured,
                "type": type(self.odds).__name__,
                "has_data": not self.odds.odds_df.empty if hasattr(self.odds, "odds_df") else False,
            },
            "edge_intelligence": {
                "active": self.edge is not None,
                "type": type(self.edge).__name__ if self.edge else "disabled",
            },
        }
        if hasattr(self.odds, "quota_status"):
            status["odds_api"]["quota"] = self.odds.quota_status
        return status

    # ------------------------------------------------------------------
    # Combined helpers
    # ------------------------------------------------------------------

    def load_combined_match_data(self) -> pd.DataFrame:
        """Wide fixture-level DataFrame with team stats joined in."""
        fixtures = self.load_fixtures_df()
        team_stats = self.load_team_stats_df()
        weather = self.load_weather_df()

        if fixtures.empty:
            return pd.DataFrame()

        if not team_stats.empty:
            home_stats = team_stats[team_stats["is_home"] == 1].copy()
            away_stats = team_stats[team_stats["is_home"] == 0].copy()
            stat_cols = [
                c for c in team_stats.columns
                if c not in ["stat_id", "fixture_id", "team_id", "is_home"]
            ]
            home_stats = home_stats.rename(
                columns={c: f"home_{c}" for c in stat_cols}
            )[["fixture_id"] + [f"home_{c}" for c in stat_cols]]
            away_stats = away_stats.rename(
                columns={c: f"away_{c}" for c in stat_cols}
            )[["fixture_id"] + [f"away_{c}" for c in stat_cols]]
            fixtures = fixtures.merge(home_stats, on="fixture_id", how="left")
            fixtures = fixtures.merge(away_stats, on="fixture_id", how="left")

        if not weather.empty:
            wx_cols = ["fixture_id", "temperature_c", "humidity_pct", "wind_speed_kmh", "conditions"]
            wx_sub = weather[[c for c in wx_cols if c in weather.columns]]
            fixtures = fixtures.merge(wx_sub, on="fixture_id", how="left")

        return fixtures

    def get_team_history(self, team_id: int) -> pd.DataFrame:
        ts = self.load_team_stats_df()
        fixtures = self.load_fixtures_df()
        if ts.empty:
            return pd.DataFrame()
        ts = ts[ts["team_id"] == team_id].copy()
        ts = ts.merge(
            fixtures[["fixture_id", "season", "round", "date", "home_team_id", "away_team_id"]],
            on="fixture_id", how="left",
        )
        ts["is_home_game"] = (ts["team_id"] == ts["home_team_id"]).astype(int)
        return ts.sort_values(["season", "round"]).reset_index(drop=True)

    def get_player_history(self, player_id: int) -> pd.DataFrame:
        ps = self.load_player_stats_df()
        if ps.empty:
            return pd.DataFrame()
        fixtures = self.load_fixtures_df()
        ps = ps[ps["player_id"] == player_id].copy()
        if ps.empty:
            return ps
        ps = ps.merge(
            fixtures[["fixture_id", "season", "round", "date"]], on="fixture_id", how="left",
        )
        return ps.sort_values(["season", "round"]).reset_index(drop=True)

