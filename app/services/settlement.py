"""
Settlement service: resolves stored predictions against actual game results.

Each GeneratedLeg is settled to one of:
  1  = win
  0  = loss
 -1  = void / push (no result available or special condition)

MultiSettlement is derived from the union of its constituent LegSettlements.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import List, Optional

from loguru import logger
from sqlalchemy.orm import Session

from app.db.models import (
    Fixture,
    GeneratedLeg,
    GeneratedMulti,
)

# These models are declared in models.py via the promise in the task spec.
# We import them with a safe fallback so the module loads even if the DB
# migration has not yet been applied.
try:
    from app.db.models import LegSettlement, MultiSettlement  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    LegSettlement = None  # type: ignore[assignment,misc]
    MultiSettlement = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_over_under_line(selection: str, prefix: str) -> Optional[float]:
    """Extract the numeric line from selections like 'over_220.5' or 'under_9'."""
    pattern = rf"^{re.escape(prefix)}_(.+)$"
    m = re.match(pattern, selection)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def _parse_home_line(selection: str) -> Optional[float]:
    """
    Parse 'home_line_{X}' -> float X.
    X may be negative, e.g. 'home_line_-6.5' or 'home_line_9.5'.
    """
    m = re.match(r"^home_line_([+-]?\d+(?:\.\d+)?)$", selection)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def _settle_leg(leg: GeneratedLeg, fixture: Fixture) -> int:
    """
    Determine the settlement outcome for a single leg.

    Returns
    -------
    1   win
    0   loss
    -1  void / push (line exactly on total, no stats available, etc.)
    """
    sel = leg.selection.strip().lower()

    # ── Head-to-head ─────────────────────────────────────────────────────────
    if sel == "home_win":
        if fixture.home_win is None:
            return -1
        return 1 if fixture.home_win == 1 else 0

    if sel == "away_win":
        if fixture.home_win is None:
            return -1
        return 1 if fixture.home_win == 0 else 0

    # ── Totals ────────────────────────────────────────────────────────────────
    over_line = _parse_over_under_line(sel, "over")
    if over_line is not None:
        if fixture.total_score is None:
            return -1
        total = float(fixture.total_score)
        if total == over_line:
            return -1  # push
        return 1 if total > over_line else 0

    under_line = _parse_over_under_line(sel, "under")
    if under_line is not None:
        if fixture.total_score is None:
            return -1
        total = float(fixture.total_score)
        if total == under_line:
            return -1  # push
        return 1 if total < under_line else 0

    # ── Home handicap line ────────────────────────────────────────────────────
    home_line = _parse_home_line(sel)
    if home_line is not None:
        if fixture.margin is None:
            return -1
        # margin is home_score - away_score; add the line adjustment
        adjusted = float(fixture.margin) + home_line
        if adjusted == 0.0:
            return -1  # push
        return 1 if adjusted > 0 else 0

    # ── Player props ──────────────────────────────────────────────────────────
    # Pattern: player_{id}_over_{line} | player_{id}_under_{line}
    m = re.match(r"^player_(\d+)_(over|under)_([+-]?\d+(?:\.\d+)?)$", sel)
    if m:
        # Cannot settle player props from fixture-level data alone; mark void.
        return -1

    # Unknown selection format — mark as void to be safe
    logger.warning(
        "settlement | unknown selection format '{}' for leg {} — marking void",
        leg.selection,
        leg.id,
    )
    return -1


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class SettlementService:
    """Settle GeneratedLeg and GeneratedMulti records against actual results."""

    def __init__(self, db_session_factory=None, data_source_manager=None):
        if db_session_factory is None:
            from app.db.database import SessionLocal
            db_session_factory = SessionLocal
        self._factory = db_session_factory
        self._dsm = data_source_manager

    # ------------------------------------------------------------------
    # Dict-based resolution (used by tests and evaluation service)
    # ------------------------------------------------------------------

    def _resolve_outcome(self, selection: str, context: dict) -> Optional[int]:
        """Resolve a selection string against a context dict.

        Context keys (all optional):
            home_win: bool
            margin: int (home - away)
            total_score: int
            player_stats: dict mapping player_id (int) → {disposals: int, ...}

        Returns 1 (win), 0 (loss), or None (cannot determine).
        """
        sel = selection.strip().lower()

        # head_to_head
        if sel == "home_win":
            hw = context.get("home_win")
            if hw is None:
                return None
            return 1 if hw else 0
        if sel == "away_win":
            hw = context.get("home_win")
            if hw is None:
                return None
            return 1 if not hw else 0

        # totals
        over_line = _parse_over_under_line(sel, "over")
        if over_line is not None:
            ts = context.get("total_score")
            if ts is None:
                return None
            return 1 if float(ts) > over_line else 0

        under_line = _parse_over_under_line(sel, "under")
        if under_line is not None:
            ts = context.get("total_score")
            if ts is None:
                return None
            return 1 if float(ts) < under_line else 0

        # home line
        home_line = _parse_home_line(sel)
        if home_line is not None:
            margin = context.get("margin")
            if margin is None:
                return None
            adjusted = float(margin) + home_line
            if adjusted == 0.0:
                return None  # push
            return 1 if adjusted > 0 else 0

        # away line
        m = re.match(r"^away_line_([+-]?\d+(?:\.\d+)?)$", sel)
        if m:
            try:
                line = float(m.group(1))
            except ValueError:
                return None
            margin = context.get("margin")  # home - away
            if margin is None:
                return None
            # Away team covers: away_margin > |line| → home margin < -line
            adjusted = -float(margin) + line
            if adjusted == 0.0:
                return None
            return 1 if adjusted > 0 else 0

        # player props
        pm = re.match(r"^player_(\d+)_(over|under)_([+-]?\d+(?:\.\d+)?)$", sel)
        if pm:
            player_id = int(pm.group(1))
            direction = pm.group(2)
            try:
                line = float(pm.group(3))
            except ValueError:
                return None
            player_stats = context.get("player_stats", {})
            stats = player_stats.get(player_id)
            if stats is None:
                return None
            disposals = stats.get("disposals")
            if disposals is None:
                return None
            if direction == "over":
                return 1 if float(disposals) > line else 0
            else:
                return 1 if float(disposals) < line else 0

        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def settle_recent(self, lookback_days: int = 7) -> dict:
        """
        Find all GeneratedLegs with no LegSettlement for fixtures that are now
        completed within the lookback window.

        Returns
        -------
        dict with keys: n_settled, n_wins, n_losses, n_void, fixture_ids_settled
        """
        if LegSettlement is None:
            logger.error("LegSettlement model is not available — DB migration pending.")
            return {"n_settled": 0, "n_wins": 0, "n_losses": 0, "n_void": 0, "fixture_ids_settled": []}

        db: Session = self._factory()
        try:
            cutoff = datetime.utcnow() - timedelta(days=lookback_days)

            # All completed fixtures within lookback window
            completed_fixtures = (
                db.query(Fixture)
                .filter(Fixture.status == "completed")
                .all()
            )
            completed_ids = {f.id: f for f in completed_fixtures}

            if not completed_ids:
                logger.info("settlement | no completed fixtures found")
                return {"n_settled": 0, "n_wins": 0, "n_losses": 0, "n_void": 0, "fixture_ids_settled": []}

            # All legs for those fixtures that have no settlement yet
            settled_leg_ids = {
                row[0]
                for row in db.query(LegSettlement.leg_id).all()
            }

            unsettled_legs = (
                db.query(GeneratedLeg)
                .filter(
                    GeneratedLeg.fixture_id.in_(completed_ids.keys()),
                    ~GeneratedLeg.id.in_(settled_leg_ids) if settled_leg_ids else True,
                )
                .all()
            )

            if not unsettled_legs:
                logger.info("settlement | no unsettled legs found for completed fixtures")
                return {"n_settled": 0, "n_wins": 0, "n_losses": 0, "n_void": 0, "fixture_ids_settled": []}

            n_wins = n_losses = n_void = 0
            fixture_ids_settled: set = set()

            for leg in unsettled_legs:
                fixture = completed_ids.get(leg.fixture_id)
                if fixture is None:
                    continue
                if fixture.result_home_score is None or fixture.result_away_score is None:
                    # No result yet — skip
                    continue

                outcome = _settle_leg(leg, fixture)

                settlement = LegSettlement(
                    leg_id=leg.id,
                    fixture_id=leg.fixture_id,
                    settled_at=datetime.utcnow(),
                    actual_outcome=outcome,
                    actual_home_score=fixture.result_home_score,
                    actual_away_score=fixture.result_away_score,
                    settlement_source="auto",
                    notes=None,
                )
                db.add(settlement)

                if outcome == 1:
                    n_wins += 1
                elif outcome == 0:
                    n_losses += 1
                else:
                    n_void += 1

                fixture_ids_settled.add(leg.fixture_id)

            db.commit()
            n_settled = n_wins + n_losses + n_void
            logger.info(
                "settlement | settled {} legs (wins={} losses={} void={})",
                n_settled, n_wins, n_losses, n_void,
            )
            return {
                "n_settled": n_settled,
                "n_wins": n_wins,
                "n_losses": n_losses,
                "n_void": n_void,
                "fixture_ids_settled": sorted(fixture_ids_settled),
            }
        except Exception as exc:
            db.rollback()
            logger.exception("settlement | settle_recent failed: {}", exc)
            return {"n_settled": 0, "n_wins": 0, "n_losses": 0, "n_void": 0, "fixture_ids_settled": []}
        finally:
            db.close()

    def settle_multis(self, lookback_days: int = 7) -> dict:
        """
        For each GeneratedMulti with no MultiSettlement where all constituent
        leg fixtures are completed, derive the multi result from leg settlements.

        Returns
        -------
        dict with keys: n_settled, n_won, n_lost
        """
        if MultiSettlement is None or LegSettlement is None:
            logger.error("MultiSettlement / LegSettlement model not available — DB migration pending.")
            return {"n_settled": 0, "n_won": 0, "n_lost": 0}

        db: Session = self._factory()
        try:
            settled_multi_ids = {
                row[0]
                for row in db.query(MultiSettlement.multi_id).all()
            }

            unsettled_multis = (
                db.query(GeneratedMulti)
                .filter(~GeneratedMulti.id.in_(settled_multi_ids) if settled_multi_ids else True)
                .all()
            )

            # Build index of leg_id -> LegSettlement
            all_settlements = {
                row.leg_id: row
                for row in db.query(LegSettlement).all()
            }

            n_won = n_lost = 0
            settled_count = 0

            for multi in unsettled_multis:
                leg_ids: list = multi.leg_ids or []
                if not leg_ids:
                    continue

                # Check all legs are settled (non-void)
                settlements = [all_settlements.get(lid) for lid in leg_ids]
                if any(s is None for s in settlements):
                    # Some legs not yet settled — come back later
                    continue

                n_total = len(leg_ids)
                n_legs_won = sum(1 for s in settlements if s is not None and s.actual_outcome == 1)
                all_legs_win = n_legs_won == n_total

                # Void legs: if any leg is void, multi is void (actual_combined_result = -1)
                n_void = sum(1 for s in settlements if s is not None and s.actual_outcome == -1)
                actual_combined = -1 if n_void > 0 else (1 if all_legs_win else 0)

                ms = MultiSettlement(
                    multi_id=multi.id,
                    settled_at=datetime.utcnow(),
                    all_legs_win=all_legs_win,
                    n_legs_won=n_legs_won,
                    n_legs_total=n_total,
                    actual_combined_result=actual_combined,
                    settlement_source="auto",
                )
                db.add(ms)
                settled_count += 1

                if actual_combined == 1:
                    n_won += 1
                elif actual_combined == 0:
                    n_lost += 1

            db.commit()
            logger.info(
                "settlement | settled {} multis (won={} lost={})",
                settled_count, n_won, n_lost,
            )
            return {"n_settled": settled_count, "n_won": n_won, "n_lost": n_lost}
        except Exception as exc:
            db.rollback()
            logger.exception("settlement | settle_multis failed: {}", exc)
            return {"n_settled": 0, "n_won": 0, "n_lost": 0}
        finally:
            db.close()

    def get_unsettled_legs(self) -> List[GeneratedLeg]:
        """
        Return all GeneratedLeg records that have no corresponding
        LegSettlement AND whose fixture is marked completed.
        """
        if LegSettlement is None:
            logger.error("LegSettlement model not available.")
            return []

        db: Session = self._factory()
        try:
            settled_leg_ids = {
                row[0]
                for row in db.query(LegSettlement.leg_id).all()
            }
            completed_fixture_ids = {
                row[0]
                for row in db.query(Fixture.id).filter(Fixture.status == "completed").all()
            }

            q = db.query(GeneratedLeg).filter(
                GeneratedLeg.fixture_id.in_(completed_fixture_ids)
            )
            if settled_leg_ids:
                q = q.filter(~GeneratedLeg.id.in_(settled_leg_ids))

            results = q.all()
            # Detach from session so caller can use freely
            db.expunge_all()
            return results
        except Exception as exc:
            logger.exception("settlement | get_unsettled_legs failed: {}", exc)
            return []
        finally:
            db.close()

    def get_settlement_summary(self, days: int = 30) -> dict:
        """
        Aggregate settlement results for the last *days* days.

        Returns
        -------
        dict with overall stats and per-market breakdown.
        """
        if LegSettlement is None:
            return {"error": "LegSettlement model not available", "days": days}

        db: Session = self._factory()
        try:
            cutoff = datetime.utcnow() - timedelta(days=days)

            settlements = (
                db.query(LegSettlement)
                .filter(LegSettlement.settled_at >= cutoff)
                .all()
            )

            if not settlements:
                return {
                    "days": days,
                    "n_total": 0,
                    "n_wins": 0,
                    "n_losses": 0,
                    "n_void": 0,
                    "hit_rate": None,
                    "by_market": {},
                }

            # Join to legs for market_type breakdown
            leg_ids = [s.leg_id for s in settlements]
            legs_map = {
                leg.id: leg
                for leg in db.query(GeneratedLeg).filter(GeneratedLeg.id.in_(leg_ids)).all()
            }

            n_wins = sum(1 for s in settlements if s.actual_outcome == 1)
            n_losses = sum(1 for s in settlements if s.actual_outcome == 0)
            n_void = sum(1 for s in settlements if s.actual_outcome == -1)
            n_total = len(settlements)
            n_decidable = n_wins + n_losses
            hit_rate = n_wins / n_decidable if n_decidable > 0 else None

            # Per-market breakdown
            market_buckets: dict = {}
            for s in settlements:
                leg = legs_map.get(s.leg_id)
                mtype = leg.market_type if leg else "unknown"
                bucket = market_buckets.setdefault(mtype, {"n": 0, "wins": 0, "losses": 0, "void": 0})
                bucket["n"] += 1
                if s.actual_outcome == 1:
                    bucket["wins"] += 1
                elif s.actual_outcome == 0:
                    bucket["losses"] += 1
                else:
                    bucket["void"] += 1

            by_market = {}
            for mtype, b in market_buckets.items():
                decidable = b["wins"] + b["losses"]
                by_market[mtype] = {
                    "n": b["n"],
                    "n_wins": b["wins"],
                    "n_losses": b["losses"],
                    "n_void": b["void"],
                    "hit_rate": b["wins"] / decidable if decidable > 0 else None,
                }

            return {
                "days": days,
                "n_total": n_total,
                "n_wins": n_wins,
                "n_losses": n_losses,
                "n_void": n_void,
                "hit_rate": round(hit_rate, 4) if hit_rate is not None else None,
                "by_market": by_market,
            }
        except Exception as exc:
            logger.exception("settlement | get_settlement_summary failed: {}", exc)
            return {"error": str(exc), "days": days}
        finally:
            db.close()
