"""Demo data loader - implements provider interfaces using local CSV files."""
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional
from loguru import logger

from app.data_ingestion.providers import (
    AFLDataProvider, OddsProvider, WeatherProvider, InjuryProvider,
    FixtureData, OddsData, WeatherData, InjuryData,
)
from app.core.config import settings


class DemoAFLDataProvider(AFLDataProvider):
    """Loads AFL data from local demo CSV files."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or settings.demo_data_dir
        self._fixtures: Optional[pd.DataFrame] = None
        self._teams: Optional[pd.DataFrame] = None
        self._players: Optional[pd.DataFrame] = None
        self._team_stats: Optional[pd.DataFrame] = None
        self._player_stats: Optional[pd.DataFrame] = None

    def _load(self, name: str) -> pd.DataFrame:
        path = self.data_dir / f"{name}.csv"
        if not path.exists():
            logger.warning(f"Demo file not found: {path}")
            return pd.DataFrame()
        return pd.read_csv(path)

    @property
    def fixtures_df(self) -> pd.DataFrame:
        if self._fixtures is None:
            self._fixtures = self._load("fixtures")
        return self._fixtures

    @property
    def teams_df(self) -> pd.DataFrame:
        if self._teams is None:
            self._teams = self._load("teams")
        return self._teams

    @property
    def players_df(self) -> pd.DataFrame:
        if self._players is None:
            self._players = self._load("players")
        return self._players

    @property
    def team_stats_df(self) -> pd.DataFrame:
        if self._team_stats is None:
            self._team_stats = self._load("team_stats")
        return self._team_stats

    @property
    def player_stats_df(self) -> pd.DataFrame:
        if self._player_stats is None:
            self._player_stats = self._load("player_stats")
        return self._player_stats

    def get_fixtures(self, season: int, round: Optional[int] = None) -> List[FixtureData]:
        df = self.fixtures_df
        df = df[df["season"] == season]
        if round is not None:
            df = df[df["round"] == round]
        result = []
        for _, row in df.iterrows():
            def si(v):
                try:
                    if pd.isna(v):
                        return None
                    return int(v)
                except (ValueError, TypeError):
                    return None
            result.append(FixtureData(
                fixture_id=int(row["fixture_id"]),
                season=int(row["season"]),
                round=int(row["round"]),
                home_team_id=int(row["home_team_id"]),
                away_team_id=int(row["away_team_id"]),
                venue_id=int(row["venue_id"]),
                date=str(row["date"]),
                status=str(row.get("status", "upcoming")).strip(),
                home_score=si(row.get("result_home_score")),
                away_score=si(row.get("result_away_score")),
            ))
        return result

    def get_team_stats(self, fixture_id: int) -> List[Dict[str, Any]]:
        df = self.team_stats_df
        df = df[df["fixture_id"] == fixture_id]
        return df.to_dict("records")

    def get_player_stats(self, fixture_id: int) -> List[Dict[str, Any]]:
        df = self.player_stats_df
        if df.empty:
            return []
        df = df[df["fixture_id"] == fixture_id]
        return df.to_dict("records")

    def get_teams(self) -> List[Dict[str, Any]]:
        return self.teams_df.to_dict("records")

    def get_players(self, team_id: Optional[int] = None) -> List[Dict[str, Any]]:
        df = self.players_df
        if team_id is not None:
            df = df[df["team_id"] == team_id]
        return df.to_dict("records")


class DemoOddsProvider(OddsProvider):
    """Loads odds from local demo CSV files."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or settings.demo_data_dir
        self._odds: Optional[pd.DataFrame] = None

    @property
    def odds_df(self) -> pd.DataFrame:
        if self._odds is None:
            path = self.data_dir / "odds.csv"
            if not path.exists():
                logger.warning(f"Odds file not found: {path}")
                self._odds = pd.DataFrame()
            else:
                self._odds = pd.read_csv(path)
        return self._odds

    def get_odds(self, fixture_id: int, market_type: Optional[str] = None) -> List[OddsData]:
        df = self.odds_df
        if df.empty:
            return []
        df = df[df["fixture_id"] == fixture_id]
        if market_type:
            df = df[df["market_type"] == market_type]
        result = []
        for _, row in df.iterrows():
            result.append(OddsData(
                fixture_id=int(row["fixture_id"]),
                market_type=str(row["market_type"]),
                selection=str(row["selection"]),
                bookmaker=str(row["bookmaker"]),
                decimal_odds=float(row["decimal_odds"]),
                timestamp=str(row.get("timestamp", "")),
                status=str(row.get("status", "active")),
            ))
        return result

    def get_best_odds(self, fixture_id: int, selection: str) -> Optional[OddsData]:
        df = self.odds_df
        if df.empty:
            return None
        df = df[(df["fixture_id"] == fixture_id) & (df["selection"] == selection)]
        if df.empty:
            return None
        idx = df["decimal_odds"].idxmax()
        row = df.loc[idx]
        return OddsData(
            fixture_id=int(row["fixture_id"]),
            market_type=str(row["market_type"]),
            selection=str(row["selection"]),
            bookmaker=str(row["bookmaker"]),
            decimal_odds=float(row["decimal_odds"]),
            timestamp=str(row.get("timestamp", "")),
            status=str(row.get("status", "active")),
        )


class DemoWeatherProvider(WeatherProvider):
    """Loads weather data from local demo CSV files."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or settings.demo_data_dir
        self._weather: Optional[pd.DataFrame] = None

    @property
    def weather_df(self) -> pd.DataFrame:
        if self._weather is None:
            path = self.data_dir / "weather.csv"
            if not path.exists():
                logger.warning(f"Weather file not found: {path}")
                self._weather = pd.DataFrame()
            else:
                self._weather = pd.read_csv(path)
        return self._weather

    def get_weather(self, fixture_id: int) -> Optional[WeatherData]:
        df = self.weather_df
        if df.empty:
            return None
        row = df[df["fixture_id"] == fixture_id]
        if row.empty:
            return None
        r = row.iloc[0]
        return WeatherData(
            fixture_id=int(r["fixture_id"]),
            venue_id=int(r["venue_id"]),
            temperature_c=float(r.get("temperature_c", 18)),
            humidity_pct=float(r.get("humidity_pct", 60)),
            wind_speed_kmh=float(r.get("wind_speed_kmh", 10)),
            conditions=str(r.get("conditions", "clear")),
        )


class DemoInjuryProvider(InjuryProvider):
    """Loads injury data from local demo CSV files."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or settings.demo_data_dir
        self._injuries: Optional[pd.DataFrame] = None

    @property
    def injuries_df(self) -> pd.DataFrame:
        if self._injuries is None:
            path = self.data_dir / "injuries.csv"
            if not path.exists():
                logger.warning(f"Injuries file not found: {path}")
                self._injuries = pd.DataFrame()
            else:
                self._injuries = pd.read_csv(path)
        return self._injuries

    def get_injuries(
        self,
        fixture_id: Optional[int] = None,
        team_id: Optional[int] = None,
    ) -> List[InjuryData]:
        df = self.injuries_df
        if df.empty:
            return []
        if fixture_id is not None:
            df = df[df["fixture_id"] == fixture_id]
        if team_id is not None:
            df = df[df["team_id"] == team_id]
        result = []
        for _, row in df.iterrows():
            result.append(InjuryData(
                player_id=int(row["player_id"]),
                team_id=int(row["team_id"]),
                fixture_id=int(row["fixture_id"]),
                status=str(row["status"]),
                injury_type=str(row.get("injury_type", "")) or None,
                notes=str(row.get("notes", "")) or None,
            ))
        return result
