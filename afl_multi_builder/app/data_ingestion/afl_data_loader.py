"""
Data Sports Group (DSG) AFL data loader.

Fetches fixture, team, player, and statistics data from the DSG API
and returns clean, analysis-ready DataFrames.

Internal schema (consistent with rest of AFL Multi Builder system):
  fixtures_df     — one row per match
  team_stats_df   — one row per team per match (advanced AFL stats)
  player_stats_df — one row per player per match
  teams_df        — master team list
  players_df      — master player list
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from app.core.config import settings
from app.data_ingestion.afl_data_client import DSGClient, DSGAPIError


# ---------------------------------------------------------------------------
# DSG status → internal canonical status
# ---------------------------------------------------------------------------

_STATUS_MAP = {
    # Common DSG status strings
    "fixture": "upcoming",
    "scheduled": "upcoming",
    "not_started": "upcoming",
    "tbd": "upcoming",
    "inprogress": "in_progress",
    "in_progress": "in_progress",
    "live": "in_progress",
    "halftime": "in_progress",
    "finished": "completed",
    "complete": "completed",
    "completed": "completed",
    "full_time": "completed",
    "ft": "completed",
    "aet": "completed",
    "postponed": "postponed",
    "cancelled": "cancelled",
    "suspended": "in_progress",
    "awarded": "completed",
    "bye": "bye",
}

_COMPLETED = {"completed", "finished", "full_time", "ft", "aet", "awarded"}


def _canonical(raw: str) -> str:
    return _STATUS_MAP.get(str(raw).lower().strip(), "unknown")


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
    if not s:
        return ""
    s = str(s).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        return str(s)


def _get(d: dict, *keys, default=None):
    """Safe nested dict access."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d


def _fixture_id_from_str(s) -> int:
    """Convert match_id string to int fixture_id (hash if non-numeric)."""
    try:
        return int(s)
    except (TypeError, ValueError):
        return abs(hash(str(s))) % (10 ** 9)


# ---------------------------------------------------------------------------
# DSG AFL Loader
# ---------------------------------------------------------------------------

class AFLDataLoader:
    """
    Loads AFL fixture, team, and player data from the DSG API.

    Auto-discovers the AFL competition ID and current season on first use.
    Results are cached in-memory for the loader's lifetime to avoid duplicate
    API calls within a single pipeline run.

    Exposes:
        fixtures_df      — all matches (upcoming + completed)
        team_stats_df    — per-team aggregated stats per match
        player_stats_df  — per-player stats per match
        teams_df         — team master list
        players_df       — player master list
    """

    def __init__(self):
        if not settings.is_afl_data_configured:
            raise RuntimeError(
                "AFL_DATA_AUTHKEY is not configured. "
                "Add AFL_DATA_AUTHKEY and AFL_DATA_USERNAME to your .env file."
            )
        self._client = DSGClient()

        # Discover competition + season (config can override)
        self._competition_id: str = str(settings.afl_data_competition_id)
        self._season_id: str = settings.afl_data_season_id

        if not self._season_id:
            self._season_id = self._client.discover_current_season(self._competition_id) or ""

        # In-memory caches
        self._fixtures_df: Optional[pd.DataFrame] = None
        self._team_stats_df: Optional[pd.DataFrame] = None
        self._player_stats_df: Optional[pd.DataFrame] = None
        self._teams_df: Optional[pd.DataFrame] = None
        self._players_df: Optional[pd.DataFrame] = None

        logger.info("AFLDataLoader init: competition_id={} season_id={}",
                    self._competition_id, self._season_id)

    # ------------------------------------------------------------------
    # Public properties (lazy)
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
        """Return fixtures with status == upcoming and scheduled_utc > now."""
        fx = self.fixtures_df.copy()
        now_iso = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()
        upcoming = fx[fx["status"] == "upcoming"].copy()
        if "scheduled_utc" in upcoming.columns:
            upcoming = upcoming[upcoming["scheduled_utc"].apply(
                lambda s: bool(s) and _parse_utc(s) >= now_iso
            )]
        logger.info("AFLDataLoader: {} upcoming fixtures (total {})",
                    len(upcoming), len(fx))
        return upcoming.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Fixture loading
    # ------------------------------------------------------------------

    def _load_fixtures(self) -> pd.DataFrame:
        rows: List[Dict] = []
        if not self._season_id:
            logger.warning("AFLDataLoader: no season_id — cannot load fixtures")
            return pd.DataFrame()

        try:
            data = self._client.get_matches(
                season_id=self._season_id,
                ttl=settings.cache_ttl_upcoming_hours * 3600,
            )
            matches = self._extract_list(data, "matches", "match")
            for m in matches:
                row = self._normalise_match(m)
                if row:
                    rows.append(row)
            logger.info("AFLDataLoader: {} fixtures loaded (season={})",
                        len(rows), self._season_id)
        except DSGAPIError as exc:
            logger.error("AFLDataLoader._load_fixtures DSG error: {}", exc)
        except Exception as exc:
            logger.error("AFLDataLoader._load_fixtures error: {}", exc)

        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df = self._cast_fixture_cols(df)
        return df.sort_values(["season", "round", "scheduled_utc"],
                              na_position="last").reset_index(drop=True)

    def _normalise_match(self, m: dict) -> Optional[Dict]:
        """Normalise one DSG match object to internal schema."""
        match_id = m.get("id") or m.get("match_id")
        if not match_id:
            return None

        # Round
        round_no = _safe_int(m.get("round") or _get(m, "round_info", "round"))

        # Season / year
        season = _safe_int(
            m.get("year") or _get(m, "season", "year") or
            self._client.current_year()
        )

        # Teams — DSG uses hometeam / awayteam dicts
        home = m.get("hometeam") or m.get("home_team") or {}
        away = m.get("awayteam") or m.get("away_team") or {}
        home_id = _safe_int(home.get("id") or home.get("team_id"))
        away_id = _safe_int(away.get("id") or away.get("team_id"))
        home_name = home.get("name") or home.get("team_name") or ""
        away_name = away.get("name") or away.get("team_name") or ""

        # Date / time — DSG typically has separate date + time fields
        date_str = m.get("date") or m.get("match_date") or ""
        time_str = m.get("time") or m.get("match_time") or "00:00:00"
        if date_str:
            scheduled_utc = _parse_utc(f"{date_str}T{time_str}")
        else:
            scheduled_utc = ""

        # Status
        raw_status = m.get("status") or m.get("match_status") or "fixture"
        status = _canonical(raw_status)

        # Scores — DSG uses ht_score, ft_score strings like "95-72"
        # or nested score objects
        home_score, away_score = 0, 0
        home_goals, home_behinds = 0, 0
        away_goals, away_behinds = 0, 0

        score = m.get("score") or {}
        ft = (score.get("ft_score") or score.get("full_time") or
              m.get("ft_score") or "")
        if ft and "-" in str(ft):
            parts = str(ft).split("-")
            home_score = _safe_int(parts[0])
            away_score = _safe_int(parts[1]) if len(parts) > 1 else 0

        # Some DSG AFL feeds expose goals/behinds separately
        home_goals = _safe_int(score.get("home_goals") or m.get("home_goals"))
        home_behinds = _safe_int(score.get("home_behinds") or m.get("home_behinds"))
        away_goals = _safe_int(score.get("away_goals") or m.get("away_goals"))
        away_behinds = _safe_int(score.get("away_behinds") or m.get("away_behinds"))

        # Reconstruct score from goals/behinds if ft_score missing
        if not home_score and home_goals:
            home_score = home_goals * 6 + home_behinds
        if not away_score and away_goals:
            away_score = away_goals * 6 + away_behinds

        # Venue
        venue = _get(m, "venue", "name") or m.get("venue") or m.get("stadium") or ""
        if isinstance(venue, dict):
            venue = venue.get("name") or venue.get("venue_name") or ""

        # Derived
        home_win = margin = total_score = None
        if status == "completed" and (home_score or away_score):
            home_win = 1 if home_score > away_score else 0
            margin = home_score - away_score
            total_score = home_score + away_score

        return {
            "fixture_id": _fixture_id_from_str(match_id),
            "sport_event_id": str(match_id),
            "season": season,
            "round": round_no,
            "scheduled_utc": scheduled_utc,
            "date": date_str,
            "status": status,
            "home_team_id": home_id,
            "away_team_id": away_id,
            "home_team": home_name,
            "away_team": away_name,
            "venue": str(venue),
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
    def _cast_fixture_cols(df: pd.DataFrame) -> pd.DataFrame:
        for col in ["fixture_id", "season", "round", "home_team_id", "away_team_id",
                    "home_score", "away_score", "home_goals", "home_behinds",
                    "away_goals", "away_behinds"]:
            if col not in df.columns:
                df[col] = 0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        for col in ["sport_event_id", "status", "home_team", "away_team",
                    "venue", "date", "scheduled_utc"]:
            if col not in df.columns:
                df[col] = ""
            df[col] = df[col].fillna("").astype(str)
        return df

    # ------------------------------------------------------------------
    # Team stats — derived from individual match data
    # ------------------------------------------------------------------

    def _load_team_stats(self) -> pd.DataFrame:
        """
        Build team stats DataFrame from completed match details.
        DSG returns match-level stats inside the get_match endpoint.
        We iterate completed fixtures and pull box score stats.
        """
        rows: List[Dict] = []
        fx = self.fixtures_df
        if fx.empty:
            return pd.DataFrame()

        completed = fx[fx["status"] == "completed"].head(50)  # cap to avoid slow startup
        for _, row in completed.iterrows():
            match_id = row.get("sport_event_id", "")
            if not match_id:
                continue
            try:
                data = self._client.get_match(match_id,
                                              ttl=settings.cache_ttl_results_hours * 3600)
                extracted = self._extract_team_stats(data, int(row["fixture_id"]))
                rows.extend(extracted)
            except Exception as exc:
                logger.debug("team_stats for match {}: {}", match_id, exc)

        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        return self._cast_stat_cols(df)

    def _extract_team_stats(self, data: dict, fixture_id: int) -> List[Dict]:
        rows = []
        for is_home, side_key in ((1, "hometeam"), (0, "awayteam")):
            side = (data.get(side_key) or
                    data.get("match", {}).get(side_key) or {})
            team_id = _safe_int(side.get("id") or side.get("team_id"))
            if not team_id:
                continue
            stats = side.get("statistics") or side.get("stats") or {}
            row: Dict[str, Any] = {
                "stat_id": f"{fixture_id}_{team_id}",
                "fixture_id": fixture_id,
                "team_id": team_id,
                "is_home": is_home,
            }
            row.update(self._extract_afl_stats(stats, side))
            rows.append(row)
        return rows

    # ------------------------------------------------------------------
    # Player stats
    # ------------------------------------------------------------------

    def _load_player_stats(self) -> pd.DataFrame:
        rows: List[Dict] = []
        fx = self.fixtures_df
        if fx.empty:
            return pd.DataFrame()

        completed = fx[fx["status"] == "completed"].head(50)
        for _, row in completed.iterrows():
            match_id = row.get("sport_event_id", "")
            if not match_id:
                continue
            try:
                data = self._client.get_match(match_id,
                                              ttl=settings.cache_ttl_results_hours * 3600)
                players = self._extract_player_stats(data, int(row["fixture_id"]))
                rows.extend(players)
            except Exception as exc:
                logger.debug("player_stats for match {}: {}", match_id, exc)

        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        return self._cast_stat_cols(df)

    def _extract_player_stats(self, data: dict, fixture_id: int) -> List[Dict]:
        rows = []
        match = data.get("match") or data
        for side_key in ("hometeam", "awayteam"):
            side = match.get(side_key) or data.get(side_key) or {}
            team_id = _safe_int(side.get("id") or side.get("team_id"))
            players = (side.get("players") or side.get("player") or
                       side.get("lineups") or [])
            if isinstance(players, dict):
                players = [players]
            for p in players:
                player = p.get("player") or p
                pid = _safe_int(player.get("id") or player.get("person_id"))
                if not pid:
                    continue
                stats = p.get("statistics") or p.get("stats") or {}
                row: Dict[str, Any] = {
                    "stat_id": f"{fixture_id}_{pid}",
                    "fixture_id": fixture_id,
                    "player_id": pid,
                    "team_id": team_id,
                    "player_name": (
                        player.get("name") or player.get("display_name") or
                        f"{player.get('firstname', '')} {player.get('lastname', '')}".strip()
                    ),
                    "position": p.get("position") or player.get("position") or "",
                    "jumper_number": _safe_int(player.get("shirt_number") or
                                               player.get("jumper")),
                }
                row.update(self._extract_afl_stats(stats, p))
                rows.append(row)
        return rows

    # ------------------------------------------------------------------
    # Teams / players master lists
    # ------------------------------------------------------------------

    def _load_teams(self) -> pd.DataFrame:
        rows: List[Dict] = []
        if not self._season_id:
            return pd.DataFrame()
        try:
            # get_teams does not exist in DSG — correct endpoint is get_contestants
            data = self._client.get_contestants(season_id=self._season_id)
            teams = self._extract_list(data, "contestants", "contestant", "teams", "team")
            for t in teams:
                rows.append({
                    "team_id": _safe_int(t.get("id") or t.get("team_id")),
                    "name": t.get("name") or t.get("team_name") or "",
                    "short_name": t.get("short_name") or t.get("abbr") or "",
                    "country": t.get("country") or t.get("area") or "",
                    "venue": _get(t, "venue", "name") or t.get("stadium") or "",
                })
        except Exception as exc:
            logger.warning("AFLDataLoader._load_teams: {}", exc)
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def _load_players(self) -> pd.DataFrame:
        rows: List[Dict] = []
        try:
            data = self._client.get_peoples(season_id=self._season_id)
            players = self._extract_list(data, "peoples", "people", "players", "player")
            for p in players:
                person = p.get("person") or p.get("player") or p
                rows.append({
                    "player_id": _safe_int(person.get("id") or person.get("person_id")),
                    "team_id": _safe_int(p.get("team_id") or _get(p, "team", "id")),
                    "first_name": person.get("firstname") or person.get("first_name") or "",
                    "surname": person.get("lastname") or person.get("surname") or "",
                    "display_name": person.get("name") or person.get("display_name") or "",
                    "position": p.get("position") or person.get("position") or "",
                    "jumper_number": _safe_int(person.get("shirt_number") or person.get("jumper")),
                    "dob": person.get("date_of_birth") or person.get("dob") or "",
                    "height_cm": _safe_float(person.get("height")),
                    "weight_kg": _safe_float(person.get("weight")),
                })
        except Exception as exc:
            logger.warning("AFLDataLoader._load_players: {}", exc)
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    # ------------------------------------------------------------------
    # Stats extraction — handles DSG stat naming conventions
    # ------------------------------------------------------------------

    def _extract_afl_stats(self, stats: dict, context: dict = None) -> Dict[str, float]:
        """Extract all AFL advanced statistics from a DSG stats dict."""
        ctx = context or {}

        def pick(*keys) -> float:
            for k in keys:
                v = stats.get(k) or ctx.get(k)
                if v is not None:
                    return _safe_float(v)
            return 0.0

        return {
            "goals": pick("goals"),
            "behinds": pick("behinds", "points"),
            "score": pick("score", "total_score", "points_total"),
            "kicks": pick("kicks"),
            "handballs": pick("handballs"),
            "disposals": pick("disposals"),
            "effective_kicks": pick("effective_kicks", "effectivekicks"),
            "effective_handballs": pick("effective_handballs", "effectivehandballs"),
            "effective_disposals": pick("effective_disposals", "effectivedisposals"),
            "clangers": pick("clangers"),
            "marks": pick("marks"),
            "contested_marks": pick("contested_marks", "contestedmarks"),
            "marks_inside_50": pick("marks_inside_50", "marksinside50"),
            "contested_possessions": pick("contested_possessions", "contestedpossessions"),
            "uncontested_possessions": pick("uncontested_possessions", "uncontestedpossessions"),
            "clearances": pick("clearances", "total_clearances"),
            "centre_clearances": pick("centre_clearances", "centreclearances"),
            "stoppage_clearances": pick("stoppage_clearances", "stoppageclearances"),
            "tackles": pick("tackles"),
            "inside_50s": pick("inside_50s", "inside50s", "inside_fifties"),
            "rebound_50s": pick("rebound_50s", "rebound50s", "rebound_fifties"),
            "hitouts": pick("hitouts"),
            "hitouts_to_advantage": pick("hitouts_to_advantage", "hitoutstoadvantage"),
            "score_involvements": pick("score_involvements", "scoreinvolvements"),
            "goal_assists": pick("goal_assists", "goalassists"),
            "frees_for": pick("frees_for", "freesfor", "free_kicks_for"),
            "frees_against": pick("frees_against", "freesagainst", "free_kicks_against"),
            "one_percenters": pick("one_percenters", "onepercenters"),
            "bounces": pick("bounces"),
            "time_on_ground_pct": pick("time_on_ground", "timesonground", "tog"),
            "brownlow_votes": pick("brownlow_votes", "brownlowvotes"),
            "supercoach_score": pick("supercoach_score", "supercoach"),
            "rating_points": pick("rating_points", "ratingpoints"),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_list(data: dict, *keys) -> List[dict]:
        """Try multiple keys to find the list of records in a DSG response."""
        for key in keys:
            val = data.get(key)
            if val is not None:
                if isinstance(val, list):
                    return val
                if isinstance(val, dict):
                    return [val]
        # Try nested under "data"
        inner = data.get("data") or {}
        if isinstance(inner, dict):
            for key in keys:
                val = inner.get(key)
                if val is not None:
                    if isinstance(val, list):
                        return val
                    if isinstance(val, dict):
                        return [val]
        return []

    @staticmethod
    def _cast_stat_cols(df: pd.DataFrame) -> pd.DataFrame:
        stat_cols = [
            "goals", "behinds", "score", "kicks", "handballs", "disposals",
            "effective_kicks", "effective_handballs", "effective_disposals",
            "clangers", "marks", "contested_marks", "marks_inside_50",
            "contested_possessions", "uncontested_possessions",
            "clearances", "centre_clearances", "stoppage_clearances",
            "tackles", "inside_50s", "rebound_50s", "hitouts",
            "hitouts_to_advantage", "score_involvements", "goal_assists",
            "frees_for", "frees_against", "one_percenters", "bounces",
            "time_on_ground_pct", "brownlow_votes", "supercoach_score",
            "rating_points",
        ]
        for col in stat_cols:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        return df
