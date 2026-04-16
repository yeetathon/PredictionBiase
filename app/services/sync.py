"""Sync service: pull data from Sportradar and persist to local DB.

Provides three sync operations:
  - ``sync_upcoming``  — refresh fixtures for the next N days
  - ``sync_settle_recent`` — settle predictions against finished games
  - ``sync_backfill``  — bulk-load historical seasons (use sparingly — costs quota)
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from loguru import logger

from app.core.config import settings
from app.db.database import SessionLocal
from app.db.models import Fixture, Team


class SyncService:
    """Wraps data ingestion and settlement in one place.

    Sync operations are idempotent: re-running ``sync_upcoming`` will update
    existing rows rather than duplicating them.
    """

    def __init__(self) -> None:
        from app.db.database import init_db
        init_db()  # no-op if tables already exist
        from app.data_ingestion.data_source_manager import get_data_source_manager
        self._dsm = get_data_source_manager()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sync_upcoming(
        self,
        season_id: Optional[str] = None,
        lookahead_days: int = 14,
    ) -> Dict[str, Any]:
        """Pull upcoming fixtures from Sportradar and upsert to DB.

        Parameters
        ----------
        season_id:
            Sportradar season ID.  Defaults to ``settings.sportradar_afl_season_id``.
        lookahead_days:
            How many days ahead to include.

        Returns
        -------
        dict with ``n_synced``, ``n_new``, ``n_updated``, ``status``
        """
        run_id = str(uuid.uuid4())[:8]
        logger.info("SyncService.sync_upcoming run_id={} lookahead={}d", run_id, lookahead_days)

        df = self._dsm.get_upcoming_fixtures(season_id=season_id)
        if df.empty:
            return {"run_id": run_id, "status": "no_data", "n_synced": 0}

        cutoff = datetime.utcnow() + timedelta(days=lookahead_days)
        if "date" in df.columns:
            # filter rows within lookahead window
            def _parse_date(d):
                try:
                    return datetime.fromisoformat(str(d).replace("Z", "+00:00").split("+")[0])
                except Exception:
                    return datetime.utcnow()

            df = df[df["date"].apply(_parse_date) <= cutoff]

        n_new = 0
        n_updated = 0

        with SessionLocal() as db:
            for _, row in df.iterrows():
                result = self._upsert_fixture(db, row)
                if result == "new":
                    n_new += 1
                elif result == "updated":
                    n_updated += 1
            db.commit()

        logger.info("sync_upcoming complete: new={} updated={}", n_new, n_updated)
        return {
            "run_id": run_id,
            "status": "ok",
            "n_synced": n_new + n_updated,
            "n_new": n_new,
            "n_updated": n_updated,
            "source": df["_source_type"].iloc[0] if "_source_type" in df.columns else "unknown",
        }

    def sync_settle_recent(self, lookback_days: Optional[int] = None) -> Dict[str, Any]:
        """Fetch completed fixtures and settle open predictions.

        Parameters
        ----------
        lookback_days:
            How far back to look for completions. Defaults to
            ``settings.recent_settlement_lookback_days``.
        """
        from app.services.settlement import SettlementService

        lookback = lookback_days or settings.recent_settlement_lookback_days
        run_id = str(uuid.uuid4())[:8]
        logger.info("SyncService.sync_settle_recent run_id={} lookback={}d", run_id, lookback)

        # First update DB with completed results from API
        df = self._dsm.get_completed_fixtures(lookback_days=lookback)
        n_results_updated = 0
        if not df.empty:
            with SessionLocal() as db:
                for _, row in df.iterrows():
                    if self._update_fixture_result(db, row):
                        n_results_updated += 1
                db.commit()

        # Now run settlement logic
        svc = SettlementService()
        leg_summary = svc.settle_recent(lookback_days=lookback)
        multi_summary = svc.settle_multis()

        result = {
            "run_id": run_id,
            "status": "ok",
            "n_results_updated": n_results_updated,
            "leg_settlement": leg_summary,
            "multi_settlement": multi_summary,
        }
        logger.info("sync_settle_recent complete: {}", result)
        return result

    def sync_backfill(
        self,
        season_ids: Optional[List[str]] = None,
        max_fixtures: int = 50,
    ) -> Dict[str, Any]:
        """Back-fill historical fixture data (costs API quota — use sparingly).

        Parameters
        ----------
        season_ids:
            List of Sportradar season IDs to backfill.
        max_fixtures:
            Hard cap on fixtures to fetch summaries for (to manage quota).
        """
        if settings.effective_data_mode == "demo":
            return {"status": "skipped", "reason": "demo_mode"}

        run_id = str(uuid.uuid4())[:8]
        logger.info("SyncService.sync_backfill run_id={} seasons={}", run_id, season_ids)

        if not season_ids:
            return {"run_id": run_id, "status": "no_seasons_specified"}

        n_fetched = 0
        for sid in season_ids:
            df = self._dsm.get_completed_fixtures(season_id=sid)
            if df.empty:
                continue
            with SessionLocal() as db:
                for _, row in df.iterrows():
                    if n_fetched >= max_fixtures:
                        break
                    if self._upsert_fixture(db, row) == "new":
                        n_fetched += 1
                db.commit()
            if n_fetched >= max_fixtures:
                break

        return {
            "run_id": run_id,
            "status": "ok",
            "n_fetched": n_fetched,
            "seasons_processed": len(season_ids),
        }

    def get_sync_status(self) -> Dict[str, Any]:
        """Return current sync / data-source status."""
        ds_status = self._dsm.get_status()
        with SessionLocal() as db:
            total_fixtures = db.query(Fixture).count()
            upcoming = db.query(Fixture).filter(Fixture.status == "upcoming").count()
            completed = db.query(Fixture).filter(Fixture.status == "completed").count()

        return {
            "data_mode": ds_status.mode,
            "effective_mode": ds_status.effective_mode,
            "sportradar_configured": ds_status.sportradar_configured,
            "demo_available": ds_status.demo_available,
            "quota": ds_status.quota_status,
            "db_fixtures": {
                "total": total_fixtures,
                "upcoming": upcoming,
                "completed": completed,
            },
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _upsert_fixture(self, db, row) -> str:
        """Insert or update a fixture row. Returns 'new' or 'updated'."""
        sport_event_id = str(row.get("sport_event_id", ""))
        season = int(row.get("season", 0)) if row.get("season") else 0
        round_no = int(row.get("round", 0)) if row.get("round") else 0

        # Try lookup by sport_event_id if available
        existing = None
        if sport_event_id:
            existing = db.query(Fixture).filter(
                Fixture.sport_event_id == sport_event_id
            ).first() if hasattr(Fixture, "sport_event_id") else None

        if existing is None and season and round_no:
            # Fallback: match by season + round (works for demo data)
            existing = db.query(Fixture).filter(
                Fixture.season == season,
                Fixture.round == round_no,
            ).first()

        if existing is None:
            fixture = Fixture(
                season=season,
                round=round_no,
                status=str(row.get("status", "upcoming")),
                date=str(row.get("date", "")) if row.get("date") else None,
            )
            db.add(fixture)
            return "new"
        else:
            existing.status = str(row.get("status", existing.status))
            if row.get("date"):
                existing.date = str(row["date"])
            db.add(existing)
            return "updated"

    def _update_fixture_result(self, db, row) -> bool:
        """Update a fixture with final score/result data."""
        season = int(row.get("season", 0)) if row.get("season") else 0
        round_no = int(row.get("round", 0)) if row.get("round") else 0

        existing = db.query(Fixture).filter(
            Fixture.season == season,
            Fixture.round == round_no,
        ).first()
        if existing is None:
            return False

        if row.get("home_score") is not None:
            existing.result_home_score = int(row["home_score"])
        if row.get("away_score") is not None:
            existing.result_away_score = int(row["away_score"])

        home_score = existing.result_home_score or 0
        away_score = existing.result_away_score or 0
        if home_score or away_score:
            existing.home_win = 1 if home_score > away_score else 0
            existing.margin = home_score - away_score
            existing.total_score = home_score + away_score
            existing.status = "completed"

        db.add(existing)
        return True
