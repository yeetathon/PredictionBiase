"""
SportradarLoader — live-only AFL data provider.

Fetches the full season schedule from Sportradar and exposes DataFrames
that are compatible with the DataLoader interface (same column schema as
DemoAFLDataProvider). Never falls back to demo/mock data.

Raises RuntimeError on construction if SPORTRADAR_API_KEY is not set.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd

from app.core.config import settings

logger = logging.getLogger(__name__)


class NoUpcomingFixturesError(Exception):
    """Raised when no future fixtures exist in the schedule."""


class SportradarLoader:
    """
    Live AFL data provider backed by Sportradar v2 API.

    On construction, validates API key is present. Fetches the current
    season schedule (with scores embedded for completed games) and builds
    DataFrames matching the internal schema expected by DataLoader.

    Fixture IDs are derived by extracting the numeric suffix of the
    Sportradar event ID (``sr:sport_event:12345678`` → ``12345678``),
    which is stable and deterministic.

    Player data is unavailable from the schedule endpoint alone (fetching
    per-match summaries would exhaust trial quota), so ``players_df`` and
    ``player_stats_df`` are always empty.
    """

    _UPCOMING_STATUSES = frozenset({
        "not_started", "created", "delayed", "scheduled",
    })
    _COMPLETED_STATUSES = frozenset({
        "closed", "complete", "ended", "finalised",
    })

    def __init__(self) -> None:
        if not settings.is_sportradar_configured:
            raise RuntimeError(
                "SPORTRADAR_API_KEY is not configured. "
                "Add it to your .env file: SPORTRADAR_API_KEY=your_key_here"
            )

        from app.data_ingestion.sportradar_client import SportradarClient
        from app.data_ingestion.cache_manager import CacheManager
        from app.data_ingestion.quota_manager import QuotaManager

        self._client = SportradarClient(
            api_key=settings.sportradar_api_key,
            base_url=settings.sportradar_base_url,
            rate_limit_qps=settings.api_rate_limit_qps,
            quota_manager=QuotaManager(),
            cache_manager=CacheManager(),
        )

        self._season_id: Optional[str] = None
        self._season_year: int = datetime.utcnow().year
        self._raw_events: Optional[List[Dict]] = None

        # Cached DataFrames
        self._fixtures_df: Optional[pd.DataFrame] = None
        self._teams_df: Optional[pd.DataFrame] = None
        self._team_stats_df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Season discovery
    # ------------------------------------------------------------------

    def _get_season_id(self) -> str:
        """Return the current season's Sportradar ID, auto-discovering if needed."""
        if self._season_id:
            return self._season_id

        if settings.sportradar_afl_season_id:
            self._season_id = settings.sportradar_afl_season_id
            logger.info("Using configured season ID: %s", self._season_id)
            return self._season_id

        competition_id = settings.sportradar_afl_competition_id
        endpoint = f"competitions/{competition_id}/seasons.json"
        logger.info("Auto-discovering AFL season from %s", endpoint)
        data = self._client.get(endpoint, cache_ttl_seconds=86400)

        from app.data_ingestion.sportradar_normalizer import SportradarNormalizer
        seasons = SportradarNormalizer().normalize_seasons(data)

        current_year = str(datetime.utcnow().year)
        for s in seasons:
            if str(s.get("year", "")) == current_year:
                self._season_id = s["sportradar_id"]
                self._season_year = int(current_year)
                logger.info("Discovered season: %s (%s)", self._season_id, current_year)
                return self._season_id

        # Fall back to most recent season
        if seasons:
            latest = max(seasons, key=lambda s: str(s.get("year", "0")))
            self._season_id = latest["sportradar_id"]
            try:
                self._season_year = int(latest.get("year", datetime.utcnow().year))
            except (TypeError, ValueError):
                self._season_year = datetime.utcnow().year
            logger.warning(
                "No %s season found; using %s (%s)",
                current_year, self._season_id, self._season_year,
            )
            return self._season_id

        raise RuntimeError(
            "Could not discover AFL season ID from Sportradar API. "
            "Set SPORTRADAR_AFL_SEASON_ID in your .env to override."
        )

    # ------------------------------------------------------------------
    # Raw event extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_from_schedules(data: Dict) -> List[Dict]:
        """Extract sport_event dicts from a schedules.json response.

        Handles both v3 (``schedules`` array of wrappers) and v2
        (``sport_events`` flat array) response shapes.
        """
        # v3: {"schedules": [{"sport_event": {...}, "sport_event_status": {...}}]}
        if "schedules" in data:
            events: List[Dict] = []
            for wrapper in (data.get("schedules") or []):
                if not isinstance(wrapper, dict):
                    continue
                ev = dict(wrapper.get("sport_event") or {})
                if not ev:
                    continue
                ses = wrapper.get("sport_event_status")
                if ses:
                    ev["sport_event_status"] = ses
                events.append(ev)
            return events
        # v2 / fallback: {"sport_events": [...]}
        return list(data.get("sport_events") or [])

    @staticmethod
    def _extract_from_summaries(data: Dict) -> List[Dict]:
        """Extract sport_event dicts (with embedded scores) from summaries.json.

        Sportradar Australian Rules v3/en uses ``sport_events_summaries`` as the
        top-level key.  Fall back to the generic ``summaries`` key used by some
        other Sportradar sport APIs.

        Expected wrapper shape (either key)::

            [
              {
                "sport_event": {
                  "id": "sr:sport_event:...",
                  "scheduled": "2025-03-20T09:35:00+00:00",
                  "status": "not_started" | "closed" | ...,
                  "competitors": [...],
                  "sport_event_context": {"round": {"number": 1}}
                },
                "sport_event_status": {          # present for completed games
                  "status": "closed",
                  "home_score": 95,
                  "away_score": 82,
                  "winner_id": "sr:competitor:..."
                }
              },
              ...
            ]
        """
        # Australian Rules v3/en top-level key takes priority
        wrappers = (
            data.get("sport_events_summaries")
            or data.get("summaries")
            or []
        )
        events: List[Dict] = []
        for wrapper in wrappers:
            if not isinstance(wrapper, dict):
                continue
            ev = dict(wrapper.get("sport_event") or {})
            if not ev:
                continue
            ses = wrapper.get("sport_event_status")
            if ses:
                ev["sport_event_status"] = ses
            events.append(ev)
        return events

    @staticmethod
    def _merge_events(
        schedule_events: List[Dict], summary_events: List[Dict]
    ) -> List[Dict]:
        """Merge schedule (upcoming) and summary (completed+scores) events.

        Summary events take precedence on duplicate IDs because they carry
        confirmed scores from ``sport_event_status``.
        """
        merged: Dict[str, Dict] = {}
        for ev in schedule_events:
            eid = ev.get("id") or ""
            if eid:
                merged[eid] = ev
        # Summaries win — they have the authoritative scores
        for ev in summary_events:
            eid = ev.get("id") or ""
            if eid:
                merged[eid] = ev
        return list(merged.values())

    # ------------------------------------------------------------------
    # Raw event fetch
    # ------------------------------------------------------------------

    def _fetch_events(self) -> List[Dict]:
        """Fetch all season fixtures from ``seasons/{season_id}/summaries.json``.

        This is the only supported endpoint for Australian Rules v3/en.
        ``schedules.json`` is NOT used — it is not available for this sport.

        The response top-level key is ``sport_events_summaries`` (Australian
        Rules v3/en standard).  The parser also accepts the generic
        ``summaries`` key as a fallback.

        Debug information logged at INFO level:
        * full URL being called
        * HTTP source (cache vs api_live)
        * all top-level JSON keys in the response
        * number of fixtures parsed

        If zero fixtures are parsed the full response structure is logged at
        ERROR level before raising.
        """
        if self._raw_events is not None:
            return self._raw_events

        season_id = self._get_season_id()
        ep = f"seasons/{season_id}/summaries.json"
        full_url = f"{settings.sportradar_base_url}/{ep}"
        ttl = int(settings.cache_ttl_upcoming_hours * 3600)

        logger.info("Sportradar: calling summaries — URL: %s", full_url)

        try:
            data = self._client.get(ep, cache_ttl_seconds=ttl)
        except Exception as exc:
            raise RuntimeError(
                f"Sportradar summaries.json request failed.\n"
                f"  Season  : {season_id}\n"
                f"  URL     : {full_url}\n"
                f"  Error   : {exc}"
            ) from exc

        # ── Debug: log top-level keys so we can see exactly what came back ──
        top_keys = sorted(k for k in data.keys() if not k.startswith("_"))
        logger.info(
            "Sportradar summaries response — source=%s  HTTP status=200"
            "  top-level keys: %s",
            data.get("_source", "?"), top_keys,
        )

        events = self._extract_from_summaries(data)
        logger.info(
            "Sportradar: parsed %d fixtures from summaries.json (season=%s)",
            len(events), season_id,
        )

        if not events:
            # Log the raw structure in detail before failing — makes diagnosis easy
            structure = {
                k: (
                    f"list[{len(v)}]" if isinstance(v, list)
                    else f"dict[{list(v.keys())[:5]}]" if isinstance(v, dict)
                    else type(v).__name__
                )
                for k, v in data.items()
                if not k.startswith("_")
            }
            logger.error(
                "Sportradar summaries.json returned ZERO fixtures.\n"
                "  Season    : %s\n"
                "  Full URL  : %s\n"
                "  Top-level keys   : %s\n"
                "  Response structure: %s\n"
                "Possible causes:\n"
                "  • Wrong top-level key — expected 'sport_events_summaries'\n"
                "  • Season has no matches yet (pre-season)\n"
                "  • Season ID is incorrect — set SPORTRADAR_AFL_SEASON_ID in .env to override",
                season_id, full_url, top_keys, structure,
            )
            raise RuntimeError(
                f"Sportradar summaries.json returned zero fixtures for season {season_id}.\n"
                f"  URL: {full_url}\n"
                f"  Response top-level keys: {top_keys}\n"
                "See logs for full response structure."
            )

        self._raw_events = events
        return events

    # ------------------------------------------------------------------
    # ID helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sr_to_int(sr_id: str) -> int:
        """Extract numeric suffix: ``sr:sport_event:12345`` → 12345, 0 on failure."""
        if not sr_id:
            return 0
        try:
            return int(sr_id.split(":")[-1])
        except (ValueError, IndexError):
            return 0

    @classmethod
    def _map_status(cls, sr_status: str) -> str:
        """Map Sportradar status string to internal ``"completed"`` / ``"upcoming"``."""
        s = (sr_status or "").lower()
        if s in cls._COMPLETED_STATUSES:
            return "completed"
        return "upcoming"

    # ------------------------------------------------------------------
    # DataFrame builders
    # ------------------------------------------------------------------

    def _build_fixtures_df(self) -> pd.DataFrame:
        """Build fixtures_df from raw schedule events."""
        events = self._fetch_events()
        season_year = self._season_year

        records = []
        for event in events:
            if not isinstance(event, dict):
                continue

            sr_match_id: str = event.get("id") or ""
            fixture_id = self._sr_to_int(sr_match_id)
            if fixture_id == 0:
                continue

            # Australian Rules v3/en uses "start_time"; fall back to "scheduled"
            raw_dt_str: str = (
                event.get("start_time")
                or event.get("scheduled")
                or ""
            )
            date_str = ""
            time_str = ""
            scheduled_utc = ""
            if raw_dt_str:
                try:
                    dt = datetime.fromisoformat(raw_dt_str.replace("Z", "+00:00"))
                    # Always convert to UTC before stripping timezone so comparisons
                    # against datetime.utcnow() are correct regardless of the offset
                    # the API sends (e.g. +10:00 / +11:00 AEST/AEDT).
                    if dt.tzinfo is not None:
                        dt_utc = dt.astimezone(timezone.utc).replace(tzinfo=None)
                    else:
                        dt_utc = dt
                    date_str = dt_utc.strftime("%Y-%m-%d")
                    time_str = dt_utc.strftime("%H:%M")
                    scheduled_utc = dt_utc.isoformat()
                except Exception:
                    scheduled_utc = raw_dt_str  # preserve raw; filter handles it

            sr_status = (event.get("status") or "not_started").lower()
            status = self._map_status(sr_status)

            # Competitors
            home_team_id = 0
            away_team_id = 0
            home_name = "Unknown"
            away_name = "Unknown"
            home_sr_id_str = ""
            for comp in (event.get("competitors") or []):
                if not isinstance(comp, dict):
                    continue
                q = (comp.get("qualifier") or "").lower()
                tid = self._sr_to_int(comp.get("id") or "")
                name = comp.get("name") or "Unknown"
                if q == "home":
                    home_team_id = tid
                    home_name = name
                    home_sr_id_str = comp.get("id") or ""
                elif q == "away":
                    away_team_id = tid
                    away_name = name

            # Round number
            context = event.get("sport_event_context") or {}
            round_obj = context.get("round") or {}
            round_no = 0
            try:
                round_no = int(round_obj.get("number") or 0)
            except (TypeError, ValueError):
                pass

            # Scores from embedded sport_event_status (present for completed games)
            ses = event.get("sport_event_status") or {}
            home_score = None
            away_score = None
            home_win = None
            margin = None
            total_score = None

            if status == "completed":
                raw_home = ses.get("home_score")
                raw_away = ses.get("away_score")
                if raw_home is not None and raw_away is not None:
                    try:
                        home_score = int(raw_home)
                        away_score = int(raw_away)
                        margin = home_score - away_score
                        total_score = home_score + away_score
                        winner_id = ses.get("winner_id") or ""
                        if winner_id:
                            home_win = 1 if winner_id == home_sr_id_str else 0
                        elif home_score != away_score:
                            home_win = 1 if home_score > away_score else 0
                    except (TypeError, ValueError):
                        pass

            records.append({
                "fixture_id": fixture_id,
                "sportradar_id": sr_match_id,
                "season": season_year,
                "round": round_no,
                "home_team_id": home_team_id,
                "away_team_id": away_team_id,
                "home_team_name": home_name,
                "away_team_name": away_name,
                "venue_id": 0,
                "date": date_str,
                "time": time_str,
                "scheduled_utc": scheduled_utc,
                "status": status,
                "home_score": home_score,
                "away_score": away_score,
                "home_win": home_win,
                "margin": margin,
                "total_score": total_score,
            })

        if not records:
            logger.warning("SportradarLoader: no fixture records built from schedule")
            return pd.DataFrame()

        df = pd.DataFrame(records)
        logger.info(
            "SportradarLoader: built %d fixtures (%d completed, %d upcoming)",
            len(df),
            (df["status"] == "completed").sum(),
            (df["status"] == "upcoming").sum(),
        )
        return df

    def _build_teams_df(self) -> pd.DataFrame:
        """Extract unique teams from schedule competitors."""
        events = self._fetch_events()
        seen: Dict[int, str] = {}
        for event in events:
            for comp in (event.get("competitors") or []):
                if not isinstance(comp, dict):
                    continue
                tid = self._sr_to_int(comp.get("id") or "")
                name = comp.get("name") or "Unknown"
                if tid and tid not in seen:
                    seen[tid] = name

        if not seen:
            return pd.DataFrame()
        records = [{"team_id": tid, "name": name} for tid, name in sorted(seen.items())]
        return pd.DataFrame(records)

    def _build_team_stats_df(self) -> pd.DataFrame:
        """Build minimal team_stats_df (score only) from completed fixtures."""
        fx = self.fixtures_df
        completed = fx[fx["status"] == "completed"].copy()
        if completed.empty:
            return pd.DataFrame()

        records = []
        stat_id = 1
        for _, row in completed.iterrows():
            fid = int(row["fixture_id"])
            hs = row.get("home_score")
            as_ = row.get("away_score")
            if hs is not None:
                records.append({
                    "stat_id": stat_id,
                    "fixture_id": fid,
                    "team_id": int(row["home_team_id"]),
                    "is_home": 1,
                    "score": int(hs),
                })
                stat_id += 1
            if as_ is not None:
                records.append({
                    "stat_id": stat_id,
                    "fixture_id": fid,
                    "team_id": int(row["away_team_id"]),
                    "is_home": 0,
                    "score": int(as_),
                })
                stat_id += 1

        return pd.DataFrame(records) if records else pd.DataFrame()

    # ------------------------------------------------------------------
    # Provider interface (matches DemoAFLDataProvider)
    # ------------------------------------------------------------------

    @property
    def fixtures_df(self) -> pd.DataFrame:
        if self._fixtures_df is None:
            self._fixtures_df = self._build_fixtures_df()
        return self._fixtures_df

    @property
    def teams_df(self) -> pd.DataFrame:
        if self._teams_df is None:
            self._teams_df = self._build_teams_df()
        return self._teams_df

    @property
    def team_stats_df(self) -> pd.DataFrame:
        if self._team_stats_df is None:
            self._team_stats_df = self._build_team_stats_df()
        return self._team_stats_df

    @property
    def players_df(self) -> pd.DataFrame:
        """Always empty — player stats unavailable from trial API schedule."""
        return pd.DataFrame()

    @property
    def player_stats_df(self) -> pd.DataFrame:
        """Always empty — player stats unavailable from trial API schedule."""
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Future-only fixture filtering (strict datetime comparison)
    # ------------------------------------------------------------------

    def load_upcoming_fixtures_df(self) -> pd.DataFrame:
        """Return only fixtures whose start time is strictly in the future (UTC).

        Decision order per row
        ----------------------
        1. Parse ``scheduled_utc`` (already stored as naive UTC by _build_fixtures_df).
        2. If that is empty/unparseable, fall back to ``status == "upcoming"``
           so that fixtures with a missing datetime are not silently dropped.

        Debug logs
        ----------
        * total events in fixtures_df
        * sample scheduled_utc from first row
        * current UTC comparison time
        * count before and after the filter

        Raises
        ------
        NoUpcomingFixturesError
            If no future fixtures are found.
        """
        fx = self.fixtures_df
        if fx.empty or "scheduled_utc" not in fx.columns:
            raise NoUpcomingFixturesError(
                "No fixture data available from Sportradar schedule."
            )

        now_utc = datetime.utcnow()

        # ── Debug: show what we're working with ──────────────────────────────
        logger.info(
            "load_upcoming_fixtures_df: total=%d  now_utc=%s",
            len(fx), now_utc.isoformat(),
        )
        if not fx.empty:
            sample_raw = fx.iloc[0].get("scheduled_utc", "")
            sample_status = fx.iloc[0].get("status", "")
            logger.info(
                "load_upcoming_fixtures_df: sample row — scheduled_utc=%r  status=%r",
                sample_raw, sample_status,
            )

        future_mask = []
        for _, row in fx.iterrows():
            raw = str(row.get("scheduled_utc", "") or "").strip()

            if raw:
                try:
                    dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                    # _build_fixtures_df already converts to UTC, but handle the
                    # edge case where raw still carries a tzinfo offset.
                    if dt.tzinfo is not None:
                        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
                    future_mask.append(dt > now_utc)
                    continue
                except Exception:
                    pass  # fall through to status-based fallback

            # Fallback: no parseable datetime — trust the status field
            future_mask.append(str(row.get("status", "")).strip() == "upcoming")

        future_df = fx[future_mask].copy()

        logger.info(
            "load_upcoming_fixtures_df: %d upcoming after filter (discarded %d)",
            len(future_df), len(fx) - len(future_df),
        )

        if future_df.empty:
            raise NoUpcomingFixturesError(
                f"No upcoming AFL fixtures found (checked {len(fx)} events). "
                "The season may be over, or the schedule may not yet be published."
            )

        logger.info(
            "SportradarLoader: %d upcoming fixtures (after %s UTC)",
            len(future_df),
            now_utc.strftime("%Y-%m-%d %H:%M"),
        )
        return future_df
