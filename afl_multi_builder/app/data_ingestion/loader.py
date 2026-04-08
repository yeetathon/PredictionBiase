"""Main data loader: aggregates data from providers into DataFrames for modelling."""
import pandas as pd
import numpy as np
from typing import Optional, Tuple
from loguru import logger

from app.data_ingestion.demo_loader import (
    DemoAFLDataProvider,
    DemoOddsProvider,
    DemoWeatherProvider,
    DemoInjuryProvider,
)


class DataLoader:
    """
    Aggregates data from all providers into analysis-ready DataFrames.
    Inject different providers to switch between demo and live data.
    """

    def __init__(
        self,
        afl_provider=None,
        odds_provider=None,
        weather_provider=None,
        injury_provider=None,
    ):
        self.afl = afl_provider or DemoAFLDataProvider()
        self.odds = odds_provider or DemoOddsProvider()
        self.weather = weather_provider or DemoWeatherProvider()
        self.injuries = injury_provider or DemoInjuryProvider()

    def load_fixtures_df(self) -> pd.DataFrame:
        """Load fixtures from CSV directly."""
        return self.afl.fixtures_df.copy()

    def load_team_stats_df(self) -> pd.DataFrame:
        """Load team stats."""
        return self.afl.team_stats_df.copy()

    def load_player_stats_df(self) -> pd.DataFrame:
        """Load player stats."""
        return self.afl.player_stats_df.copy()

    def load_odds_df(self) -> pd.DataFrame:
        """Load odds."""
        return self.odds.odds_df.copy()

    def load_weather_df(self) -> pd.DataFrame:
        """Load weather."""
        return self.weather.weather_df.copy()

    def load_injuries_df(self) -> pd.DataFrame:
        """Load injuries."""
        return self.injuries.injuries_df.copy()

    def load_teams_df(self) -> pd.DataFrame:
        """Load teams."""
        return self.afl.teams_df.copy()

    def load_players_df(self) -> pd.DataFrame:
        """Load players."""
        return self.afl.players_df.copy()

    def load_combined_match_data(self) -> pd.DataFrame:
        """
        Build a wide fixture-level DataFrame with team stats joined in.
        Used for feature engineering.
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

        # Join weather
        if not weather.empty:
            weather_cols = [
                "fixture_id", "temperature_c", "humidity_pct",
                "wind_speed_kmh", "conditions"
            ]
            weather_sub = weather[[c for c in weather_cols if c in weather.columns]]
            fixtures = fixtures.merge(weather_sub, on="fixture_id", how="left")

        return fixtures

    def get_player_history(self, player_id: int) -> pd.DataFrame:
        """Return sorted historical stats for a player."""
        ps = self.load_player_stats_df()
        if ps.empty:
            return pd.DataFrame()
        fixtures = self.load_fixtures_df()
        ps = ps[ps["player_id"] == player_id].copy()
        if ps.empty:
            return ps
        ps = ps.merge(
            fixtures[["fixture_id", "season", "round", "date"]],
            on="fixture_id",
            how="left",
        )
        ps = ps.sort_values(["season", "round"]).reset_index(drop=True)
        return ps

    def get_team_history(self, team_id: int) -> pd.DataFrame:
        """Return sorted historical team stats for a team."""
        ts = self.load_team_stats_df()
        fixtures = self.load_fixtures_df()
        if ts.empty:
            return pd.DataFrame()
        ts = ts[ts["team_id"] == team_id].copy()
        ts = ts.merge(
            fixtures[["fixture_id", "season", "round", "date", "home_team_id", "away_team_id"]],
            on="fixture_id",
            how="left",
        )
        ts["is_home_game"] = (ts["team_id"] == ts["home_team_id"]).astype(int)
        ts = ts.sort_values(["season", "round"]).reset_index(drop=True)
        return ts
