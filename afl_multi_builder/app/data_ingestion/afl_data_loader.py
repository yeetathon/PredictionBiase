"""
AFL Data Sports Group data loader.

Primary data provider for all AFL fixture, team, player, and statistics data.
Returns clean, analysis-ready DataFrames.

Internal schema (consistent with rest of AFL Multi Builder system):
  fixtures_df    — one row per match
  team_stats_df  — one row per team per match (rich AFL stats)
  player_stats_df— one row per player per match
  teams_df       — master team list
  players_df     — master player list
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from app.core.config import settings
from app.data_ingestion.afl_data_client import AFLDataClient, AFLDataAPIError


# ---------------------------------------------------------------------------
# Column name mapping: AFL Data API field → internal name
# ---------------------------------------------------------------------------

_FIXTURE_COL_MAP = {
    "matchId": "sport_event_id",
    "roundId": "round",
    "roundNumber": "round",
    "utcStartTime": "scheduled_utc",
    "startTime": "scheduled_utc",
    "localStartTime": "date",
    "homeTeam.teamId": "home_team_id",
    "awayTeam.teamId": "away_team_id",
    "homeTeam.name": "home_team",
    "awayTeam.name": "away_team",
    "venue.name": "venue",
    "status": "status",
    "year": "season",
    "compSeason.year": "season",
    "homeTeamScore.totalScore": "home_score",
    "awayTeamScore.totalScore": "away_score",
}

_TEAM_STAT_COLS = [
    # Scoring
    "goals", "behinds", "score",
    # Possessions
    "kicks", "handballs", "disposals",
    "effective_kicks", "effective_handballs", "effective_disposals",
    "clangers",
    # Contested
    "contested_possessions", "uncontested_possessions",
    "contested_marks",
    # Ground ball
    "clearances", "centre_clearances", "stoppage_clearances",
    "tackles", "inside_50s", "rebound_50s",
    # Marks
    "marks", "marks_inside_50",
    # Ruck
    "hitouts", "hitouts_to_advantage",
    # Score
    "score_involvements", "goal_assists",
    # Frees
    "frees_for", "frees_against",
    # Other
    "one_percenters", "bounces", "time_on_ground_pct",
]

_PLAYER_STAT_COLS = _TEAM_STAT_COLS + [
    "brownlow_votes", "supercoach_score", "rating_points",
]

_STATUS_MAP = {
    # AFL Data API status strings → internal canonical names
    "SCHEDULED": "upcoming",
    "UNCONFIRMED_TEAMS": "upcoming",
    "CONFIRMED_TEAMS": "upcoming",
    "IN_PROGRESS": "in_progress",
    "BREAK": "in_progress",
    "COMPLETED": "completed",
    "CONCLUDED": "completed",
    "BYE": "bye",
    "POSTPONED": "postponed",
    # Lower-case variants
    "scheduled": "upcoming",
    "confirmed": "upcoming",
    "in_progress": "in_progress",
    "completed": "completed",
    "bye": "bye",
}


def _extract(d: dict, *keys: str, default=None):
    """Traverse nested dict by key path, e.g. _extract(d, 'home', 'team', 'id')."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d


def _safe_int(v, default: int = 0) -> int:
    try:
        return int(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _safe_float(v, default: float = 0.0) -> float:
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _parse_utc(s) -> str:
    """Return ISO UTC string from various date formats."""
    if not s:
        return ""
    s = str(s)
    # Replace Z with +00:00 for fromisoformat
    s = s.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        return s


def _canonical_status(raw: str) -> str:
    return _STATUS_MAP.get(str(raw).strip(), "unknown")


# ---------------------------------------------------------------------------
# AFL Data Sports Group Loader
# ---------------------------------------------------------------------------

class AFLDataLoader:
    """
    Loads AFL fixture, team, and player data from the AFL Data Sports Group API.

    Exposes the same interface as the rest of the prediction pipeline expects:
    fixtures_df, team_stats_df, player_stats_df, teams_df, players_df.

    Properties:
      fixtures_df     — pd.DataFrame of all known fixtures (completed + upcoming)
      team_stats_df   — pd.DataFrame of per-team stats per match
      player_stats_df — pd.DataFrame of per-player stats per match
      teams_df        — pd.DataFrame of team master data
      players_df      — pd.DataFrame of player master data

    All DataFrames are lazily populated on first access and then cached in
    memory for the lifetime of the loader instance.
    """

    def __init__(self):
        if not settings.is_afl_data_configured:
            raise RuntimeError(
                "AFL_DATA_AUTHKEY is not configured. "
                "Add AFL_DATA_AUTHKEY=<key> to your .env file."
            )
        self._client = AFLDataClient()
        self._year = self._client.current_year()

        # Lazy-loaded caches
        self._fixtures_df: Optional[pd.DataFrame] = None
        self._team_stats_df: Optional[pd.DataFrame] = None
        self._player_stats_df: Optional[pd.DataFrame] = None
        self._teams_df: Optional[pd.DataFrame] = None
        self._players_df: Optional[pd.DataFrame] = None

        logger.info("AFLDataLoader initialised (year={})", self._year)

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def fixtures_df(self) -> pd.DataFrame:
        if self._fixtures_df is None:
            self._fixtures_df = self._load_fixtures()
        return self._fixtures_df

    @property
    def team_stats_df(self) -> pd.DataFrame:
        if self._team_stats_df is None:
            self._team_stats_df = self._load_team_stats()
        return self._team_stats_df

    @property
    def player_stats_df(self) -> pd.DataFrame:
        if self._player_stats_df is None:
            self._player_stats_df = self._load_player_stats()
        return self._player_stats_df

    @property
    def teams_df(self) -> pd.DataFrame:
        if self._teams_df is None:
            self._teams_df = self._load_teams()
        return self._teams_df

    @property
    def players_df(self) -> pd.DataFrame:
        if self._players_df is None:
            self._players_df = self._load_players()
        return self._players_df

    def load_upcoming_fixtures_df(self) -> pd.DataFrame:
        """Return only future fixtures (status == upcoming, scheduled_utc > now)."""
        fx = self.fixtures_df.copy()
        now_str = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
        if "status" in fx.columns:
            upcoming = fx[fx["status"] == "upcoming"].copy()
        else:
            upcoming = fx.copy()

        if "scheduled_utc" in upcoming.columns:
            upcoming = upcoming[upcoming["scheduled_utc"].apply(
                lambda s: _parse_utc(s) >= now_str if s else False
            )].copy()

        logger.info("AFLDataLoader: {} upcoming fixtures (from {})", len(upcoming), len(fx))
        return upcoming

    # ------------------------------------------------------------------
    # Fixture loading
    # ------------------------------------------------------------------

    def _load_fixtures(self) -> pd.DataFrame:
        """Fetch and normalise all fixtures for the current season."""
        rows: List[Dict] = []
        try:
            data = self._client.get_fixture_list(
                year=self._year,
                ttl=settings.cache_ttl_upcoming_hours * 3600,
            )
            raw_fixtures = self._extract_fixture_list(data)
            for f in raw_fixtures:
                row = self._normalise_fixture(f)
                if row:
                    rows.append(row)
            logger.info("AFLDataLoader: loaded {} fixtures for {}", len(rows), self._year)
        except AFLDataAPIError as exc:
            logger.error("AFLDataLoader._load_fixtures failed: {}", exc)
        except Exception as exc:
            logger.error("AFLDataLoader._load_fixtures unexpected error: {}", exc)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df = self._ensure_fixture_schema(df)
        return df.sort_values(["season", "round", "scheduled_utc"],
                               na_position="last").reset_index(drop=True)

    def _extract_fixture_list(self, data: dict) -> List[dict]:
        """Handle multiple AFL Data API response shapes."""
        # Shape 1: {"fixtures": [...]}
        if "fixtures" in data:
            return data["fixtures"] or []
        # Shape 2: {"matches": [...]}
        if "matches" in data:
            return data["matches"] or []
        # Shape 3: {"fixtureList": [...]}
        if "fixtureList" in data:
            return data["fixtureList"] or []
        # Shape 4: top-level array
        if isinstance(data, list):
            return data
        # Shape 5: nested {"data": {"fixtures": [...]}}
        if "data" in data and isinstance(data["data"], dict):
            return self._extract_fixture_list(data["data"])
        return []

    def _normalise_fixture(self, f: dict) -> Optional[Dict]:
        """Convert one raw AFL Data fixture dict to internal schema."""
        # Match ID
        match_id = (
            f.get("matchId") or f.get("id") or f.get("gameNumber") or ""
        )
        if not match_id:
            return None

        # Round
        round_no = (
            _safe_int(f.get("roundNumber"))
            or _safe_int(_extract(f, "round", "roundNumber"))
            or _safe_int(f.get("round"))
        )

        # Season year
        season = (
            _safe_int(f.get("year"))
            or _safe_int(_extract(f, "compSeason", "year"))
            or self._year
        )

        # Teams
        home = f.get("homeTeam") or {}
        away = f.get("awayTeam") or {}
        home_id = _safe_int(home.get("teamId") or home.get("id"))
        away_id = _safe_int(away.get("teamId") or away.get("id"))
        home_name = home.get("name") or home.get("nickname") or ""
        away_name = away.get("name") or away.get("nickname") or ""

        # Scheduled time
        scheduled_utc = _parse_utc(
            f.get("utcStartTime") or f.get("startTime") or f.get("date") or ""
        )

        # Status
        raw_status = f.get("status") or f.get("matchStatus") or "SCHEDULED"
        status = _canonical_status(raw_status)

        # Scores
        home_score_obj = f.get("homeTeamScore") or {}
        away_score_obj = f.get("awayTeamScore") or {}
        home_score = (
            _safe_int(home_score_obj.get("totalScore"))
            or _safe_int(f.get("homeScore"))
        )
        away_score = (
            _safe_int(away_score_obj.get("totalScore"))
            or _safe_int(f.get("awayScore"))
        )

        # Goals + behinds
        home_goals = _safe_int(home_score_obj.get("goals") or f.get("homeGoals"))
        home_behinds = _safe_int(home_score_obj.get("behinds") or f.get("homeBehinds"))
        away_goals = _safe_int(away_score_obj.get("goals") or f.get("awayGoals"))
        away_behinds = _safe_int(away_score_obj.get("behinds") or f.get("awayBehinds"))

        # Venue
        venue = _extract(f, "venue", "name") or f.get("venueName") or ""

        # Derived
        home_win = None
        margin = None
        total_score = None
        if status == "completed" and (home_score or away_score):
            home_win = 1 if home_score > away_score else 0
            margin = home_score - away_score
            total_score = home_score + away_score

        return {
            "fixture_id": _safe_int(match_id) if str(match_id).isdigit() else abs(hash(str(match_id))) % (10 ** 9),
            "sport_event_id": str(match_id),
            "season": season,
            "round": round_no,
            "scheduled_utc": scheduled_utc,
            "date": scheduled_utc[:10] if scheduled_utc else "",
            "status": status,
            "home_team_id": home_id,
            "away_team_id": away_id,
            "home_team": home_name,
            "away_team": away_name,
            "venue": venue,
            "home_score": home_score,
            "away_score": away_score,
            "home_goals": home_goals,
            "home_behinds": home_behinds,
            "away_goals": away_goals,
            "away_behinds": away_behinds,
            "home_win": home_win,
            "margin": margin,
            "total_score": total_score,
        }

    @staticmethod
    def _ensure_fixture_schema(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all required columns exist with correct types."""
        int_cols = ["fixture_id", "season", "round", "home_team_id", "away_team_id",
                    "home_score", "away_score", "home_goals", "home_behinds",
                    "away_goals", "away_behinds"]
        for col in int_cols:
            if col not in df.columns:
                df[col] = 0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

        str_cols = ["sport_event_id", "status", "home_team", "away_team",
                    "venue", "date", "scheduled_utc"]
        for col in str_cols:
            if col not in df.columns:
                df[col] = ""
            df[col] = df[col].fillna("").astype(str)

        return df

    # ------------------------------------------------------------------
    # Team stats loading
    # ------------------------------------------------------------------

    def _load_team_stats(self) -> pd.DataFrame:
        """Fetch per-team stats for all completed matches this season."""
        rows: List[Dict] = []
        try:
            # Get team stats for current season
            data = self._client.get_team_stats(
                year=self._year,
                ttl=settings.cache_ttl_results_hours * 3600,
            )
            raw = self._extract_stats_list(data, "teamStats")
            for entry in raw:
                extracted = self._normalise_team_stats(entry)
                if extracted:
                    rows.extend(extracted)
            logger.info("AFLDataLoader: loaded {} team-stat rows for {}", len(rows), self._year)
        except AFLDataAPIError as exc:
            logger.warning("AFLDataLoader._load_team_stats failed: {}", exc)
        except Exception as exc:
            logger.warning("AFLDataLoader._load_team_stats unexpected: {}", exc)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        return self._ensure_stats_schema(df, stat_cols=_TEAM_STAT_COLS)

    def _normalise_team_stats(self, entry: dict) -> Optional[List[Dict]]:
        """Convert one match's team stats entry into [home_row, away_row]."""
        match_id = entry.get("matchId") or entry.get("id")
        if not match_id:
            return None
        fixture_id = _safe_int(match_id) if str(match_id).isdigit() else abs(hash(str(match_id))) % (10 ** 9)

        result = []
        for is_home, side_key in ((1, "homeTeam"), (0, "awayTeam")):
            side = entry.get(side_key) or {}
            team_id = _safe_int(side.get("teamId") or side.get("id"))
            stats = side.get("stats") or side.get("teamStats") or {}
            if not team_id:
                continue
            row: Dict[str, Any] = {
                "stat_id": f"{fixture_id}_{team_id}",
                "fixture_id": fixture_id,
                "team_id": team_id,
                "is_home": is_home,
            }
            row.update(self._extract_afl_stats(stats))
            result.append(row)

        return result if result else None

    # ------------------------------------------------------------------
    # Player stats loading
    # ------------------------------------------------------------------

    def _load_player_stats(self) -> pd.DataFrame:
        """Fetch per-player stats for all completed matches this season."""
        rows: List[Dict] = []
        try:
            data = self._client.get_player_stats(
                year=self._year,
                ttl=settings.cache_ttl_results_hours * 3600,
            )
            raw = self._extract_stats_list(data, "playerStats")
            for entry in raw:
                row = self._normalise_player_stats(entry)
                if row:
                    rows.append(row)
            logger.info("AFLDataLoader: loaded {} player-stat rows for {}", len(rows), self._year)
        except AFLDataAPIError as exc:
            logger.warning("AFLDataLoader._load_player_stats failed: {}", exc)
        except Exception as exc:
            logger.warning("AFLDataLoader._load_player_stats unexpected: {}", exc)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        return self._ensure_stats_schema(df, stat_cols=_PLAYER_STAT_COLS)

    def _normalise_player_stats(self, entry: dict) -> Optional[Dict]:
        """Convert one player-match stats record to internal schema."""
        match_id = entry.get("matchId") or entry.get("gameId")
        player = entry.get("player") or {}
        player_id = _safe_int(player.get("playerId") or player.get("id") or entry.get("playerId"))
        team_id = _safe_int(entry.get("teamId") or _extract(entry, "team", "teamId"))

        if not match_id or not player_id:
            return None

        fixture_id = _safe_int(match_id) if str(match_id).isdigit() else abs(hash(str(match_id))) % (10 ** 9)
        stats = entry.get("stats") or entry.get("playerStats") or entry

        row: Dict[str, Any] = {
            "stat_id": f"{fixture_id}_{player_id}",
            "fixture_id": fixture_id,
            "player_id": player_id,
            "team_id": team_id,
            "player_name": (
                player.get("displayName") or player.get("name")
                or f"{player.get('firstName', '')} {player.get('surname', '')}".strip()
                or entry.get("playerName", "")
            ),
            "jumper_number": _safe_int(player.get("jumperNumber")),
            "position": player.get("position") or entry.get("position", ""),
        }
        row.update(self._extract_afl_stats(stats))
        return row

    # ------------------------------------------------------------------
    # Team and player master lists
    # ------------------------------------------------------------------

    def _load_teams(self) -> pd.DataFrame:
        """Fetch team master list."""
        rows: List[Dict] = []
        try:
            data = self._client.get_team_list(year=self._year)
            teams = (data.get("teams") or data.get("teamList") or
                     data.get("data", {}).get("teams", []) or [])
            for t in teams:
                rows.append({
                    "team_id": _safe_int(t.get("teamId") or t.get("id")),
                    "name": t.get("name") or t.get("teamName") or "",
                    "nickname": t.get("nickname") or t.get("shortName") or "",
                    "abbreviation": t.get("abbreviation") or t.get("code") or "",
                    "state": t.get("state") or "",
                    "venue": _extract(t, "venue", "name") or t.get("groundName") or "",
                })
        except Exception as exc:
            logger.warning("AFLDataLoader._load_teams failed: {}", exc)

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def _load_players(self) -> pd.DataFrame:
        """Fetch player master list with profiles."""
        rows: List[Dict] = []
        try:
            data = self._client.get_player_list(year=self._year)
            players = (data.get("players") or data.get("playerList") or
                       data.get("data", {}).get("players", []) or [])
            for p in players:
                player_info = p.get("player") or p
                rows.append({
                    "player_id": _safe_int(player_info.get("playerId") or player_info.get("id")),
                    "team_id": _safe_int(p.get("teamId") or _extract(p, "team", "teamId")),
                    "first_name": player_info.get("firstName") or "",
                    "surname": player_info.get("surname") or player_info.get("lastName") or "",
                    "display_name": player_info.get("displayName") or player_info.get("name") or "",
                    "jumper_number": _safe_int(player_info.get("jumperNumber")),
                    "position": player_info.get("position") or p.get("position") or "",
                    "dob": player_info.get("dateOfBirth") or "",
                    "height_cm": _safe_float(player_info.get("heightInCm")),
                    "weight_kg": _safe_float(player_info.get("weightInKg")),
                })
        except Exception as exc:
            logger.warning("AFLDataLoader._load_players failed: {}", exc)

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    # ------------------------------------------------------------------
    # Stats extraction helpers
    # ------------------------------------------------------------------

    def _extract_afl_stats(self, stats: dict) -> Dict[str, float]:
        """
        Extract all AFL advanced statistics from a stats dict.

        Handles multiple possible key names for each stat (AFL Data API
        uses camelCase or snake_case depending on endpoint version).
        """
        def pick(*keys, d=None) -> float:
            d = d or stats
            for k in keys:
                v = d.get(k)
                if v is not None:
                    return _safe_float(v)
            return 0.0

        return {
            # Scoring
            "goals": pick("goals"),
            "behinds": pick("behinds"),
            "score": pick("score", "totalScore", "total_score"),
            # Basic possessions
            "kicks": pick("kicks"),
            "handballs": pick("handballs"),
            "disposals": pick("disposals"),
            # Effective possessions
            "effective_kicks": pick("effectiveKicks", "effective_kicks"),
            "effective_handballs": pick("effectiveHandballs", "effective_handballs"),
            "effective_disposals": pick("effectiveDisposals", "effective_disposals"),
            "clangers": pick("clangers"),
            # Marks
            "marks": pick("marks"),
            "contested_marks": pick("contestedMarks", "contested_marks"),
            "marks_inside_50": pick("marksInside50", "marks_inside_50"),
            # Contested/uncontested
            "contested_possessions": pick("contestedPossessions", "contested_possessions"),
            "uncontested_possessions": pick("uncontestedPossessions", "uncontested_possessions"),
            # Ground ball / pressure
            "clearances": pick("clearances", "totalClearances", "total_clearances"),
            "centre_clearances": pick("centreClearances", "centre_clearances"),
            "stoppage_clearances": pick("stoppageClearances", "stoppage_clearances"),
            "tackles": pick("tackles"),
            "inside_50s": pick("inside50s", "insideFifties", "inside_50s"),
            "rebound_50s": pick("rebound50s", "reboundFifties", "rebound_50s"),
            # Ruck
            "hitouts": pick("hitouts"),
            "hitouts_to_advantage": pick("hitoutsToAdvantage", "hitouts_to_advantage"),
            # Score involvement
            "score_involvements": pick("scoreInvolvements", "score_involvements"),
            "goal_assists": pick("goalAssists", "goal_assists"),
            # Frees
            "frees_for": pick("freesFor", "frees_for"),
            "frees_against": pick("freesAgainst", "frees_against"),
            # Other
            "one_percenters": pick("onePercenters", "one_percenters"),
            "bounces": pick("bounces"),
            "time_on_ground_pct": pick("timeOnGroundPercentage", "timeOnGround",
                                       "time_on_ground_pct"),
            # Player-only (zero for team rows)
            "brownlow_votes": pick("brownlowVotes", "brownlow_votes"),
            "supercoach_score": pick("supercoachScore", "supercoach_score"),
            "rating_points": pick("ratingPoints", "rating_points"),
        }

    @staticmethod
    def _extract_stats_list(data: dict, primary_key: str) -> List[dict]:
        """Extract list of stat records from AFL Data API response."""
        if primary_key in data:
            return data[primary_key] or []
        if "data" in data and isinstance(data["data"], dict):
            if primary_key in data["data"]:
                return data["data"][primary_key] or []
        if "matches" in data:
            return data["matches"] or []
        if isinstance(data, list):
            return data
        return []

    @staticmethod
    def _ensure_stats_schema(df: pd.DataFrame, stat_cols: List[str]) -> pd.DataFrame:
        """Ensure all stat columns exist and are numeric."""
        for col in stat_cols:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        return df
