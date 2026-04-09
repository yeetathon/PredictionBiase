"""
Main data loader: aggregates provider data into DataFrames for modelling.

In live/cache mode the AFL provider is SportradarLoader (requires API key).
Demo mode is not supported in production — constructing DataLoader without
an explicit afl_provider when no API key is configured raises RuntimeError.

Tests that need deterministic data should inject DemoAFLDataProvider
explicitly:
    from app.data_ingestion.demo_loader import DemoAFLDataProvider
    loader = DataLoader(afl_provider=DemoAFLDataProvider(demo_data_path))
"""
from __future__ import annotations

import pandas as pd
from typing import Optional
from loguru import logger

from app.core.config import settings


# ---------------------------------------------------------------------------
# Null providers (returned when API has no odds/weather/injuries)
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
# Provider factory
# ---------------------------------------------------------------------------

def _make_afl_provider():
    """
    Return the appropriate AFL data provider based on effective_data_mode.

    live / cache → SportradarLoader (raises if no API key)
    demo         → RuntimeError (demo mode is not supported in production)
    """
    mode = settings.effective_data_mode
    if mode in ("live", "cache"):
        from app.data_ingestion.sportradar_loader import SportradarLoader
        return SportradarLoader()

    raise RuntimeError(
        f"Data mode is '{mode}' but SPORTRADAR_API_KEY is not configured. "
        "Set SPORTRADAR_API_KEY in your .env file. "
        "Demo mode has been removed from production — live API data is required."
    )


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------

class DataLoader:
    """
    Aggregates data from all providers into analysis-ready DataFrames.

    Inject providers to override the default (used by tests):
        DataLoader(afl_provider=DemoAFLDataProvider(path))
    """

    def __init__(
        self,
        afl_provider=None,
        odds_provider=None,
        weather_provider=None,
        injury_provider=None,
    ):
        self.afl = afl_provider or _make_afl_provider()
        self.odds = odds_provider or _NullOddsProvider()
        self.weather = weather_provider or _NullWeatherProvider()
        self.injuries = injury_provider or _NullInjuryProvider()

    # ------------------------------------------------------------------
    # Core load methods
    # ------------------------------------------------------------------

    def load_fixtures_df(self) -> pd.DataFrame:
        """All fixtures (completed + upcoming) — used by feature engineering."""
        return self.afl.fixtures_df.copy()

    def load_upcoming_fixtures_df(self) -> pd.DataFrame:
        """
        Future fixtures only. Delegates to provider's strict datetime filter
        if available; otherwise filters by status == 'upcoming'.

        Raises NoUpcomingFixturesError (from sportradar_loader) if the live
        provider finds no future games.
        """
        if hasattr(self.afl, "load_upcoming_fixtures_df"):
            return self.afl.load_upcoming_fixtures_df()

        # Fallback for test-injected demo providers (status field is reliable)
        fx = self.afl.fixtures_df.copy()
        fx["status"] = fx["status"].str.strip()
        upcoming = fx[fx["status"] == "upcoming"]
        return upcoming

    def load_team_stats_df(self) -> pd.DataFrame:
        return self.afl.team_stats_df.copy()

    def load_player_stats_df(self) -> pd.DataFrame:
        return self.afl.player_stats_df.copy()

    def load_odds_df(self) -> pd.DataFrame:
        return self.odds.odds_df.copy()

    def load_weather_df(self) -> pd.DataFrame:
        return self.weather.weather_df.copy()

    def load_injuries_df(self) -> pd.DataFrame:
        return self.injuries.injuries_df.copy()

    def load_teams_df(self) -> pd.DataFrame:
        return self.afl.teams_df.copy()

    def load_players_df(self) -> pd.DataFrame:
        return self.afl.players_df.copy()

    # ------------------------------------------------------------------
    # Combined / joined helpers
    # ------------------------------------------------------------------

    def load_combined_match_data(self) -> pd.DataFrame:
        """
        Wide fixture-level DataFrame with home_/away_ team stats joined in.
        Used by feature engineering.
        """
        fixtures = self.load_fixtures_df()
        team_stats = self.load_team_stats_df()
        weather = self.load_weather_df()

        if fixtures.empty:
            return pd.DataFrame()

        # Pivot team stats to wide format (home_ and away_ prefixes)
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
            weather_cols = [
                "fixture_id", "temperature_c", "humidity_pct",
                "wind_speed_kmh", "conditions",
            ]
            wx_sub = weather[[c for c in weather_cols if c in weather.columns]]
            fixtures = fixtures.merge(wx_sub, on="fixture_id", how="left")

        return fixtures

    def get_team_history(self, team_id: int) -> pd.DataFrame:
        """Sorted historical team stats for a team."""
        ts = self.load_team_stats_df()
        fixtures = self.load_fixtures_df()
        if ts.empty:
            return pd.DataFrame()
        ts = ts[ts["team_id"] == team_id].copy()
        ts = ts.merge(
            fixtures[["fixture_id", "season", "round", "date",
                       "home_team_id", "away_team_id"]],
            on="fixture_id", how="left",
        )
        ts["is_home_game"] = (ts["team_id"] == ts["home_team_id"]).astype(int)
        ts = ts.sort_values(["season", "round"]).reset_index(drop=True)
        return ts

    def get_player_history(self, player_id: int) -> pd.DataFrame:
        """Sorted historical stats for a player (empty when no player data)."""
        ps = self.load_player_stats_df()
        if ps.empty:
            return pd.DataFrame()
        fixtures = self.load_fixtures_df()
        ps = ps[ps["player_id"] == player_id].copy()
        if ps.empty:
            return ps
        ps = ps.merge(
            fixtures[["fixture_id", "season", "round", "date"]],
            on="fixture_id", how="left",
        )
        ps = ps.sort_values(["season", "round"]).reset_index(drop=True)
        return ps
