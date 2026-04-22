"""Sportradar AFL API response normalizer.

Converts raw Sportradar v2 JSON responses into clean internal schema dicts.
All methods handle missing keys gracefully using ``.get()`` and sensible
defaults (``None``, ``0``, ``"unknown"``).

Every normalized record includes:
    _source_type : "sportradar_api"
    _normalized_at : ISO-8601 UTC timestamp string
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _provenance() -> Dict[str, str]:
    """Return the standard provenance fields for every normalized record."""
    return {
        "_source_type": "sportradar_api",
        "_normalized_at": _now_iso(),
    }


class SportradarNormalizer:
    """Normalizes Sportradar AFL API JSON responses into internal schema dicts.

    All methods are stateless and can be called without instantiation if
    preferred, but are defined as regular methods for easy mocking/subclassing.
    """

    # ------------------------------------------------------------------
    # Seasons
    # ------------------------------------------------------------------

    def normalize_seasons(self, raw: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Normalize a seasons-list response.

        Expected raw shape::

            {
                "seasons": [
                    {
                        "id": "sr:season:118700",
                        "name": "AFL 2025",
                        "start_date": "2025-03-13",
                        "end_date": "2025-09-27",
                        "year": "2025",
                        "competition": {"id": "sr:competition:...", "name": "AFL"}
                    },
                    ...
                ]
            }

        Returns
        -------
        list of dict
            Each dict has keys:
            ``sportradar_id``, ``name``, ``year``, ``start_date``,
            ``end_date``, ``competition_name``,
            ``_source_type``, ``_normalized_at``.
        """
        results: List[Dict[str, Any]] = []
        seasons_raw = raw.get("seasons") or []
        if not isinstance(seasons_raw, list):
            logger.warning(
                "normalize_seasons: expected list under 'seasons', got %s",
                type(seasons_raw).__name__,
            )
            return results

        for item in seasons_raw:
            if not isinstance(item, dict):
                continue
            competition = item.get("competition") or {}
            record: Dict[str, Any] = {
                "sportradar_id": item.get("id") or None,
                "name": item.get("name") or "unknown",
                "year": item.get("year") or None,
                "start_date": item.get("start_date") or None,
                "end_date": item.get("end_date") or None,
                "competition_name": competition.get("name") or "unknown",
            }
            record.update(_provenance())
            results.append(record)

        logger.debug("normalize_seasons: produced %d records", len(results))
        return results

    # ------------------------------------------------------------------
    # Schedule
    # ------------------------------------------------------------------

    def normalize_schedule(self, raw: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Normalize a season-schedule response.

        Expected raw shape::

            {
                "sport_events": [
                    {
                        "id": "sr:match:12345",
                        "scheduled": "2025-03-20T09:35:00+00:00",
                        "status": "not_started",
                        "competitors": [
                            {"id": "sr:competitor:1", "name": "Adelaide Crows",
                             "qualifier": "home"},
                            {"id": "sr:competitor:2", "name": "Brisbane Lions",
                             "qualifier": "away"}
                        ],
                        "venue": {
                            "id": "sr:venue:1", "name": "Adelaide Oval",
                            "city_name": "Adelaide", "country_name": "Australia"
                        },
                        "sport_event_context": {
                            "round": {"number": 1, "name": "Round 1"}
                        }
                    },
                    ...
                ]
            }

        Returns
        -------
        list of dict
            Each dict has keys:
            ``sportradar_match_id``, ``scheduled_utc``, ``status``,
            ``home_team_sr_id``, ``home_team_name``,
            ``away_team_sr_id``, ``away_team_name``,
            ``venue_name``, ``venue_city``, ``round_info``,
            ``_source_type``, ``_normalized_at``.
        """
        results: List[Dict[str, Any]] = []
        events_raw = raw.get("sport_events") or []
        if not isinstance(events_raw, list):
            logger.warning(
                "normalize_schedule: expected list under 'sport_events', got %s",
                type(events_raw).__name__,
            )
            return results

        for event in events_raw:
            if not isinstance(event, dict):
                continue

            # Extract competitors
            competitors = event.get("competitors") or []
            home_team_sr_id: Optional[str] = None
            home_team_name: Optional[str] = None
            away_team_sr_id: Optional[str] = None
            away_team_name: Optional[str] = None

            for comp in competitors:
                if not isinstance(comp, dict):
                    continue
                qualifier = (comp.get("qualifier") or "").lower()
                if qualifier == "home":
                    home_team_sr_id = comp.get("id") or None
                    home_team_name = comp.get("name") or "unknown"
                elif qualifier == "away":
                    away_team_sr_id = comp.get("id") or None
                    away_team_name = comp.get("name") or "unknown"

            # Extract venue
            venue = event.get("venue") or {}
            venue_name = venue.get("name") or None
            venue_city = venue.get("city_name") or None

            # Extract round info from sport_event_context
            context = event.get("sport_event_context") or {}
            round_obj = context.get("round") or {}
            round_info: Optional[str] = None
            if round_obj:
                round_name = round_obj.get("name") or ""
                round_number = round_obj.get("number")
                if round_name:
                    round_info = round_name
                elif round_number is not None:
                    round_info = f"Round {round_number}"

            record: Dict[str, Any] = {
                "sportradar_match_id": event.get("id") or None,
                "scheduled_utc": event.get("scheduled") or None,
                "status": event.get("status") or "unknown",
                "home_team_sr_id": home_team_sr_id,
                "home_team_name": home_team_name,
                "away_team_sr_id": away_team_sr_id,
                "away_team_name": away_team_name,
                "venue_name": venue_name,
                "venue_city": venue_city,
                "round_info": round_info,
            }
            record.update(_provenance())
            results.append(record)

        logger.debug("normalize_schedule: produced %d records", len(results))
        return results

    # ------------------------------------------------------------------
    # Match summary
    # ------------------------------------------------------------------

    def normalize_match_summary(
        self,
        raw: Dict[str, Any],
        match_id: str,
    ) -> Dict[str, Any]:
        """Normalize a single match summary response.

        Expected raw shape::

            {
                "sport_event_status": {
                    "status": "closed",
                    "home_score": 120,
                    "away_score": 85,
                    "winner_id": "sr:competitor:1"
                },
                "sport_event": {
                    "id": "sr:match:12345",
                    "competitors": [
                        {"id": "sr:competitor:1", "qualifier": "home"},
                        {"id": "sr:competitor:2", "qualifier": "away"}
                    ]
                },
                "statistics": {
                    "totals": {
                        "competitors": [
                            {
                                "id": "sr:competitor:1",
                                "statistics": {"disposals": 350, "kicks": 195, ...}
                            },
                            ...
                        ]
                    }
                }
            }

        Parameters
        ----------
        raw :
            The full API response dict.
        match_id :
            Sportradar match ID (``sr:match:...``).  Used as fallback when the
            ``sport_event.id`` field is absent.

        Returns
        -------
        dict
            Keys: ``sportradar_match_id``, ``status``, ``home_score``,
            ``away_score``, ``home_win``, ``margin``, ``total_score``,
            ``home_team_sr_id``, ``away_team_sr_id``,
            ``home_stats`` (dict), ``away_stats`` (dict),
            ``_source_type``, ``_normalized_at``.
        """
        event_status = raw.get("sport_event_status") or {}
        sport_event = raw.get("sport_event") or {}
        statistics = raw.get("statistics") or {}

        status = event_status.get("status") or "unknown"
        home_score: int = int(event_status.get("home_score") or 0)
        away_score: int = int(event_status.get("away_score") or 0)
        winner_id: Optional[str] = event_status.get("winner_id") or None

        # Resolve team IDs from the sport_event competitors list
        home_team_sr_id: Optional[str] = None
        away_team_sr_id: Optional[str] = None
        competitors = sport_event.get("competitors") or []
        for comp in competitors:
            if not isinstance(comp, dict):
                continue
            qualifier = (comp.get("qualifier") or "").lower()
            if qualifier == "home":
                home_team_sr_id = comp.get("id") or None
            elif qualifier == "away":
                away_team_sr_id = comp.get("id") or None

        # Derived fields
        margin = home_score - away_score
        total_score = home_score + away_score
        home_win: Optional[bool] = None
        if winner_id is not None:
            home_win = winner_id == home_team_sr_id
        elif home_score != away_score:
            home_win = home_score > away_score

        # Statistics per competitor
        home_stats: Dict[str, Any] = {}
        away_stats: Dict[str, Any] = {}

        totals = statistics.get("totals") or {}
        stat_competitors = totals.get("competitors") or []
        for comp_stat in stat_competitors:
            if not isinstance(comp_stat, dict):
                continue
            comp_id = comp_stat.get("id") or ""
            raw_stats = comp_stat.get("statistics") or {}
            if comp_id == home_team_sr_id:
                home_stats = dict(raw_stats)
            elif comp_id == away_team_sr_id:
                away_stats = dict(raw_stats)

        sportradar_match_id = sport_event.get("id") or match_id

        record: Dict[str, Any] = {
            "sportradar_match_id": sportradar_match_id,
            "status": status,
            "home_score": home_score,
            "away_score": away_score,
            "home_win": home_win,
            "margin": margin,
            "total_score": total_score,
            "home_team_sr_id": home_team_sr_id,
            "away_team_sr_id": away_team_sr_id,
            "home_stats": home_stats,
            "away_stats": away_stats,
        }
        record.update(_provenance())
        logger.debug(
            "normalize_match_summary: match_id=%s status=%s home=%d away=%d",
            sportradar_match_id,
            status,
            home_score,
            away_score,
        )
        return record

    # ------------------------------------------------------------------
    # Teams
    # ------------------------------------------------------------------

    def normalize_teams(self, raw: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Normalize a teams-list response.

        Expected raw shape::

            {
                "teams": [
                    {
                        "id": "sr:competitor:1",
                        "name": "Adelaide Crows",
                        "abbreviation": "ADE",
                        "venue": {
                            "id": "sr:venue:1",
                            "name": "Adelaide Oval",
                            "city_name": "Adelaide"
                        }
                    },
                    ...
                ]
            }

        Returns
        -------
        list of dict
            Each dict has keys:
            ``sportradar_id``, ``name``, ``abbreviation``,
            ``venue_name``, ``venue_city``,
            ``_source_type``, ``_normalized_at``.
        """
        results: List[Dict[str, Any]] = []
        teams_raw = raw.get("teams") or []
        if not isinstance(teams_raw, list):
            logger.warning(
                "normalize_teams: expected list under 'teams', got %s",
                type(teams_raw).__name__,
            )
            return results

        for team in teams_raw:
            if not isinstance(team, dict):
                continue
            venue = team.get("venue") or {}
            record: Dict[str, Any] = {
                "sportradar_id": team.get("id") or None,
                "name": team.get("name") or "unknown",
                "abbreviation": team.get("abbreviation") or None,
                "venue_name": venue.get("name") or None,
                "venue_city": venue.get("city_name") or None,
            }
            record.update(_provenance())
            results.append(record)

        logger.debug("normalize_teams: produced %d records", len(results))
        return results
