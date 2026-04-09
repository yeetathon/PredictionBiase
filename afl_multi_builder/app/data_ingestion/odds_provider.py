"""
OddsAPIProvider — wraps The Odds API and exposes a DataFrame compatible
with DataLoader.load_odds_df().

Requires ODDS_API_KEY in environment. If not configured, returns empty
DataFrame and logs a warning (odds are optional — pipeline still runs
but edge calculation will be estimate-only).
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

from app.core.config import settings

logger = logging.getLogger(__name__)

_CACHE_PATH = Path("data/cache/odds_api_cache.json")
_CACHE_TTL_SECONDS = 900  # 15 minutes — odds go stale fast


def _load_cache() -> Dict:
    try:
        if _CACHE_PATH.exists():
            data = json.loads(_CACHE_PATH.read_text())
            return data
    except Exception:
        pass
    return {}


def _save_cache(data: Dict) -> None:
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(json.dumps(data))
    except Exception as e:
        logger.debug("Could not save odds cache: %s", e)


def _normalise_team(name: str) -> str:
    """Normalise a team name for fuzzy matching."""
    replacements = {
        "Greater Western Sydney": "GWS",
        "GWS Giants": "GWS",
        "Gold Coast": "Gold Coast Suns",
        "Brisbane": "Brisbane Lions",
        "Adelaide": "Adelaide Crows",
        "St Kilda Saints": "St Kilda",
        "Fremantle Dockers": "Fremantle",
        "Hawthorn Hawks": "Hawthorn",
        "Port Adelaide Power": "Port Adelaide",
        "Sydney Swans": "Sydney",
        "Western Bulldogs": "Bulldogs",
        "West Coast": "West Coast Eagles",
        "Geelong": "Geelong Cats",
        "North Melbourne": "Kangaroos",
    }
    n = name.strip()
    for k, v in replacements.items():
        if k.lower() in n.lower():
            n = v
    return n.lower()


AFL_TEAMS_CANONICAL = [
    "Adelaide Crows", "Brisbane Lions", "Carlton", "Collingwood", "Essendon",
    "Fremantle", "Geelong Cats", "Gold Coast Suns",
    "Greater Western Sydney Giants", "Hawthorn", "Melbourne", "North Melbourne",
    "Port Adelaide", "Richmond", "St Kilda", "Sydney Swans",
    "West Coast Eagles", "Western Bulldogs",
]


def _match_team(name: str, candidates: List[str]) -> Optional[str]:
    """Fuzzy-match a team name against canonical list."""
    norm = _normalise_team(name)
    for c in candidates:
        if norm in _normalise_team(c) or _normalise_team(c) in norm:
            return c
    # Partial token match
    tokens = norm.split()
    for c in candidates:
        c_norm = _normalise_team(c)
        if any(t in c_norm for t in tokens if len(t) > 3):
            return c
    return None


class OddsAPIError(Exception):
    pass


class OddsAPIProvider:
    """
    Provides real bookmaker odds from The Odds API as a DataFrame.

    The DataFrame schema matches DataLoader.load_odds_df():
        fixture_id (int), market_type (str), selection (str),
        bookmaker (str), decimal_odds (float), timestamp (str), status (str)

    Fixture IDs are matched from the passed fixtures_df using team name matching.
    If a game can't be matched, it's skipped (logged at DEBUG).
    """

    def __init__(self, fixtures_df: Optional[pd.DataFrame] = None):
        self._fixtures_df = fixtures_df  # used for fixture_id mapping
        self._cache: Dict = _load_cache()
        self._odds_df: Optional[pd.DataFrame] = None
        self._last_fetch: float = 0.0
        self._requests_remaining: Optional[int] = None
        self._requests_used: Optional[int] = None

    @property
    def is_configured(self) -> bool:
        return settings.is_odds_api_configured

    def _fetch_from_api(self) -> List[Dict]:
        """Fetch odds from The Odds API. Returns list of raw event dicts."""
        if not self.is_configured:
            logger.warning("ODDS_API_KEY not configured — skipping odds fetch")
            return []

        url = f"{settings.odds_api_base_url}/sports/{settings.odds_api_sport}/odds/"
        params = {
            "apiKey": settings.odds_api_key,
            "regions": "au",
            "markets": "h2h,spreads",
            "oddsFormat": "decimal",
            "dateFormat": "iso",
        }

        cache_key = "odds_live"
        cached = self._cache.get(cache_key)
        if cached and (time.time() - cached.get("ts", 0)) < _CACHE_TTL_SECONDS:
            logger.info("Odds API: using cached odds (age=%.0fs)", time.time() - cached["ts"])
            return cached.get("data", [])

        logger.info("Odds API: fetching live odds for %s…", settings.odds_api_sport)
        try:
            resp = requests.get(url, params=params, timeout=15)
            # Track quota
            self._requests_remaining = int(resp.headers.get("x-requests-remaining", -1))
            self._requests_used = int(resp.headers.get("x-requests-used", -1))
            logger.info(
                "Odds API quota: used=%s remaining=%s",
                self._requests_used, self._requests_remaining,
            )

            if resp.status_code == 401:
                raise OddsAPIError("Odds API: invalid API key (401)")
            if resp.status_code == 422:
                raise OddsAPIError(f"Odds API: sport not found or invalid params (422): {resp.text[:200]}")
            if resp.status_code == 429:
                raise OddsAPIError("Odds API: quota exhausted (429)")
            if resp.status_code != 200:
                raise OddsAPIError(f"Odds API: HTTP {resp.status_code}: {resp.text[:200]}")

            data = resp.json()
            self._cache[cache_key] = {"ts": time.time(), "data": data}
            _save_cache(self._cache)
            logger.info("Odds API: received %d events", len(data))
            return data

        except OddsAPIError:
            raise
        except requests.RequestException as e:
            logger.warning("Odds API: request failed: %s", e)
            # Return cached if available, even if stale
            if cached:
                logger.warning("Odds API: using stale cache as fallback")
                return cached.get("data", [])
            return []

    def _build_odds_df(self, events: List[Dict]) -> pd.DataFrame:
        """Convert raw Odds API events to internal odds DataFrame."""
        if not events:
            return pd.DataFrame()

        fixtures_df = self._fixtures_df
        records = []

        for event in events:
            home_team_raw = event.get("home_team", "")
            away_team_raw = event.get("away_team", "")
            commence = event.get("commence_time", "")

            # Attempt to match fixture_id
            fixture_id = None
            if fixtures_df is not None and not fixtures_df.empty:
                fixture_id = self._match_fixture(
                    home_team_raw, away_team_raw, commence, fixtures_df
                )

            if fixture_id is None:
                # Use a synthetic ID based on team names (negative to distinguish)
                synthetic = abs(hash(f"{home_team_raw}_{away_team_raw}")) % 1_000_000
                fixture_id = -synthetic
                logger.debug(
                    "Odds API: could not match '%s vs %s' to a fixture, using synthetic ID %d",
                    home_team_raw, away_team_raw, fixture_id,
                )

            home_name = _match_team(home_team_raw, AFL_TEAMS_CANONICAL) or home_team_raw
            away_name = _match_team(away_team_raw, AFL_TEAMS_CANONICAL) or away_team_raw

            for bookmaker in (event.get("bookmakers") or []):
                bm_key = bookmaker.get("key", "unknown")
                bm_title = bookmaker.get("title", bm_key)
                last_update = bookmaker.get("last_update", commence)

                for market in (bookmaker.get("markets") or []):
                    mtype_raw = market.get("key", "")
                    if mtype_raw == "h2h":
                        market_type = "head_to_head"
                    elif mtype_raw in ("spreads", "points_spreads"):
                        market_type = "line"
                    elif mtype_raw in ("totals", "points_totals"):
                        market_type = "totals"
                    else:
                        continue

                    for outcome in (market.get("outcomes") or []):
                        out_name = outcome.get("name", "")
                        price = float(outcome.get("price", 0))
                        point = outcome.get("point")

                        if price <= 1.0:
                            continue

                        # Map to internal selection format
                        if market_type == "head_to_head":
                            if out_name == home_team_raw or _normalise_team(out_name) in _normalise_team(home_team_raw) or _normalise_team(home_team_raw) in _normalise_team(out_name):
                                selection = "home_win"
                            else:
                                selection = "away_win"
                        elif market_type == "line":
                            is_home = (out_name == home_team_raw or _normalise_team(out_name) in _normalise_team(home_team_raw))
                            pt_str = f"{point:+.1f}" if point is not None else "0.0"
                            selection = f"home_line_{pt_str}" if is_home else f"away_line_{pt_str}"
                        elif market_type == "totals":
                            direction = "over" if "over" in out_name.lower() else "under"
                            pt_str = str(point) if point is not None else "0"
                            selection = f"{direction}_{pt_str}"
                        else:
                            continue

                        records.append({
                            "fixture_id": fixture_id,
                            "home_team_name": home_name,
                            "away_team_name": away_name,
                            "commence_time": commence,
                            "market_type": market_type,
                            "selection": selection,
                            "bookmaker": bm_title,
                            "decimal_odds": round(price, 3),
                            "timestamp": last_update,
                            "status": "active",
                        })

        if not records:
            return pd.DataFrame()
        return pd.DataFrame(records)

    def _match_fixture(
        self,
        home_raw: str,
        away_raw: str,
        commence: str,
        fixtures_df: pd.DataFrame,
    ) -> Optional[int]:
        """Match Odds API event to a fixture_id using team name fuzzy matching."""
        home_norm = _normalise_team(home_raw)
        away_norm = _normalise_team(away_raw)

        if "home_team_name" not in fixtures_df.columns:
            return None

        for _, row in fixtures_df.iterrows():
            fx_home = _normalise_team(str(row.get("home_team_name", "")))
            fx_away = _normalise_team(str(row.get("away_team_name", "")))

            home_match = (home_norm in fx_home) or (fx_home in home_norm)
            away_match = (away_norm in fx_away) or (fx_away in away_norm)

            if home_match and away_match:
                return int(row["fixture_id"])

        return None

    @property
    def odds_df(self) -> pd.DataFrame:
        """Return odds DataFrame, using cache when fresh."""
        now = time.time()
        if self._odds_df is not None and (now - self._last_fetch) < _CACHE_TTL_SECONDS:
            return self._odds_df

        try:
            events = self._fetch_from_api()
            self._odds_df = self._build_odds_df(events)
            self._last_fetch = now
            logger.info(
                "OddsAPIProvider: built odds_df with %d rows for %d events",
                len(self._odds_df), len(events),
            )
        except OddsAPIError as e:
            logger.error("OddsAPIProvider: %s — returning empty odds", e)
            if self._odds_df is None:
                self._odds_df = pd.DataFrame()

        return self._odds_df

    @property
    def quota_status(self) -> Dict:
        return {
            "requests_remaining": self._requests_remaining,
            "requests_used": self._requests_used,
            "cache_age_seconds": int(time.time() - self._last_fetch) if self._last_fetch else None,
        }
