"""Abstract provider interfaces for data sources.

These define the contracts for AFL data, odds, weather, and injury providers.
The demo implementations use local CSV files.
Real connectors (AFL API, Betfair, etc.) can implement these interfaces.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class FixtureData:
    fixture_id: int
    season: int
    round: int
    home_team_id: int
    away_team_id: int
    venue_id: int
    date: str
    status: str
    home_score: Optional[int] = None
    away_score: Optional[int] = None


@dataclass
class OddsData:
    fixture_id: int
    market_type: str
    selection: str
    bookmaker: str
    decimal_odds: float
    timestamp: str
    status: str = "active"


@dataclass
class WeatherData:
    fixture_id: int
    venue_id: int
    temperature_c: float
    humidity_pct: float
    wind_speed_kmh: float
    conditions: str


@dataclass
class InjuryData:
    player_id: int
    team_id: int
    fixture_id: int
    status: str  # Available / Doubtful / Ruled Out
    injury_type: Optional[str] = None
    notes: Optional[str] = None


class AFLDataProvider(ABC):
    """Interface for AFL match and player data."""

    @abstractmethod
    def get_fixtures(self, season: int, round: Optional[int] = None) -> List[FixtureData]:
        """Fetch fixtures for a season/round."""
        ...

    @abstractmethod
    def get_team_stats(self, fixture_id: int) -> List[Dict[str, Any]]:
        """Fetch team stats for a fixture."""
        ...

    @abstractmethod
    def get_player_stats(self, fixture_id: int) -> List[Dict[str, Any]]:
        """Fetch player stats for a fixture."""
        ...

    @abstractmethod
    def get_teams(self) -> List[Dict[str, Any]]:
        """Fetch all teams."""
        ...

    @abstractmethod
    def get_players(self, team_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Fetch players, optionally filtered by team."""
        ...


class OddsProvider(ABC):
    """Interface for bookmaker odds data."""

    @abstractmethod
    def get_odds(
        self,
        fixture_id: int,
        market_type: Optional[str] = None,
    ) -> List[OddsData]:
        """Fetch odds for a fixture."""
        ...

    @abstractmethod
    def get_best_odds(self, fixture_id: int, selection: str) -> Optional[OddsData]:
        """Get best available odds for a selection across bookmakers."""
        ...


class WeatherProvider(ABC):
    """Interface for weather data."""

    @abstractmethod
    def get_weather(self, fixture_id: int) -> Optional[WeatherData]:
        """Fetch weather data for a fixture."""
        ...


class InjuryProvider(ABC):
    """Interface for injury/availability data."""

    @abstractmethod
    def get_injuries(
        self,
        fixture_id: Optional[int] = None,
        team_id: Optional[int] = None,
    ) -> List[InjuryData]:
        """Fetch injury reports."""
        ...
