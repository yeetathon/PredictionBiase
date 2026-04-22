"""Bootstrap learning cycle service.

Runs the full daily research loop:
  1. Sync upcoming fixtures (refresh schedule)
  2. Generate predictions for upcoming games
  3. Settle finished games (match predictions → outcomes)
  4. Run self-evaluation (Brier, CLV, ROI, calibration)
  5. Retrain models if criteria are met
  6. Promote challenger if Brier improves by > threshold
  7. Produce a structured summary of what changed

Usage (programmatic)::

    svc = BootstrapCycleService()
    report = svc.run()
    print(report["summary"])

Usage (CLI)::

    python scripts/run_bootstrap_cycle.py
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from loguru import logger

from app.core.config import settings
from app.db.database import SessionLocal
from app.db.models import ResearchCycleLog


class BootstrapCycleService:
    """Orchestrates the daily bootstrap research cycle."""

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, force_retrain: bool = False) -> Dict[str, Any]:
        """Execute the full bootstrap cycle. Returns a report dict."""
        cycle_id = str(uuid.uuid4())[:8]
        started_at = datetime.utcnow()
        logger.info("BootstrapCycleService.run cycle_id={}", cycle_id)

        report: Dict[str, Any] = {
            "cycle_id": cycle_id,
            "started_at": started_at.isoformat(),
            "steps": {},
        }

        # ── Step 1: Sync upcoming ──────────────────────────────────────
        report["steps"]["sync_upcoming"] = self._step_sync_upcoming()

        # ── Step 2: Settle recent games ────────────────────────────────
        report["steps"]["settle_recent"] = self._step_settle_recent()

        # ── Step 3: Self-evaluation ────────────────────────────────────
        report["steps"]["evaluation"] = self._step_evaluate()

        # ── Step 3.5: Retrain meta-blender on settlement data ─────────
        report["steps"]["meta_blend_train"] = self._step_train_meta_blender()

        # ── Step 4: Retrain if criteria met ────────────────────────────
        retrain_result = self._step_maybe_retrain(
            eval_result=report["steps"]["evaluation"],
            force=force_retrain,
        )
        report["steps"]["retrain"] = retrain_result

        # ── Step 5: Model promotion ────────────────────────────────────
        if retrain_result.get("retrained"):
            report["steps"]["promotion"] = self._step_promote()
        else:
            report["steps"]["promotion"] = {"status": "skipped", "reason": "no_retrain"}

        # ── Step 6: Generate new predictions ──────────────────────────
        report["steps"]["predictions"] = self._step_generate_predictions()

        # ── Summary ───────────────────────────────────────────────────
        finished_at = datetime.utcnow()
        elapsed = (finished_at - started_at).total_seconds()
        report["finished_at"] = finished_at.isoformat()
        report["elapsed_seconds"] = round(elapsed, 2)
        report["summary"] = self._build_summary(report)

        # Persist to DB
        self._persist_cycle_log(cycle_id, started_at, finished_at, report)

        logger.info("Bootstrap cycle complete in {:.1f}s: {}", elapsed, report["summary"])
        return report

    # ------------------------------------------------------------------
    # Step implementations
    # ------------------------------------------------------------------

    def _step_sync_upcoming(self) -> Dict[str, Any]:
        try:
            from app.services.sync import SyncService
            svc = SyncService()
            result = svc.sync_upcoming(lookahead_days=settings.upcoming_lookahead_days)
            return {"status": "ok", **result}
        except Exception as exc:
            logger.exception("sync_upcoming failed: {}", exc)
            return {"status": "error", "error": str(exc)}

    def _step_settle_recent(self) -> Dict[str, Any]:
        try:
            from app.services.sync import SyncService
            svc = SyncService()
            result = svc.sync_settle_recent(lookback_days=settings.recent_settlement_lookback_days)
            return {"status": "ok", **result}
        except Exception as exc:
            logger.exception("sync_settle_recent failed: {}", exc)
            return {"status": "error", "error": str(exc)}

    def _step_train_meta_blender(self) -> Dict[str, Any]:
        """Re-fit MetaBlender on all available settlement data (best-effort)."""
        try:
            from app.core.meta_blender import MetaBlender
            from app.db.database import SessionLocal
            from app.db.models import LegSettlement

            db = SessionLocal()
            try:
                rows = (
                    db.query(LegSettlement)
                    .filter(LegSettlement.actual_outcome.isnot(None))
                    .all()
                )
            finally:
                db.close()

            if not rows:
                return {"status": "skipped", "reason": "no_settled_legs"}

            legs = [
                {
                    "ml_probability": r.ml_probability,
                    "signal_consensus_probability": r.signal_consensus_probability,
                    "signal_agreement": r.signal_agreement,
                    "prediction_variance": r.prediction_variance,
                    "data_completeness": r.data_completeness,
                    "trust_score": r.trust_score,
                    "outcome": int(r.actual_outcome),
                    "market_type": r.market_type or "head_to_head",
                }
                for r in rows
                if r.actual_outcome in (0, 1)
            ]

            blender = MetaBlender()
            results: Dict[str, Any] = {}
            for mt in ("head_to_head", "player_disposals"):
                market_legs = [l for l in legs if l["market_type"] == mt]
                updated = blender.train_from_settlements(market_legs, market_type=mt)
                results[mt] = {"updated": updated, "n_samples": len(market_legs)}

            return {"status": "ok", "markets": results}
        except Exception as exc:
            logger.warning("meta_blend_train failed (non-fatal): {}", exc)
            return {"status": "error", "error": str(exc)}

    def _step_evaluate(self) -> Dict[str, Any]:
        try:
            from app.services.evaluation import EvaluationService
            svc = EvaluationService()
            return svc.evaluate(lookback_days=settings.bootstrap_report_days)
        except Exception as exc:
            logger.exception("evaluation failed: {}", exc)
            return {"status": "error", "error": str(exc)}

    def _step_maybe_retrain(
        self, eval_result: Dict, force: bool = False
    ) -> Dict[str, Any]:
        """Retrain if:
        - ``force=True``, OR
        - Enough new settled games since last train, OR
        - Brier has degraded beyond the degradation threshold
        """
        try:
            should_retrain, reason = self._check_retrain_criteria(eval_result, force)
            if not should_retrain:
                return {"retrained": False, "reason": reason}

            logger.info("Retraining triggered: {}", reason)
            from app.services.training import TrainingService
            svc = TrainingService()
            result = svc.run_all()

            return {
                "retrained": True,
                "reason": reason,
                "training_result": result,
            }
        except Exception as exc:
            logger.exception("retrain step failed: {}", exc)
            return {"retrained": False, "status": "error", "error": str(exc)}

    def _step_promote(self) -> Dict[str, Any]:
        try:
            from app.services.model_manager import ModelManager
            mm = ModelManager()
            results = {}
            for model_name in ["match_model", "player_disposals_model"]:
                results[model_name] = mm.maybe_promote(model_name)
            return {"status": "ok", "promotions": results}
        except Exception as exc:
            logger.exception("promotion step failed: {}", exc)
            return {"status": "error", "error": str(exc)}

    def _step_generate_predictions(self) -> Dict[str, Any]:
        try:
            from app.services.pipeline import PredictionPipeline
            pipeline = PredictionPipeline()
            result = pipeline.run()
            n_legs = len(result.get("value_legs", [])) + len(result.get("safe_legs", []))
            n_multis = (
                len(result.get("value_multis", []))
                + len(result.get("safe_multis", []))
                + len(result.get("same_game_multis", []))
            )
            return {
                "status": "ok",
                "n_legs": n_legs,
                "n_multis": n_multis,
                "run_id": result.get("run_id", ""),
            }
        except Exception as exc:
            logger.exception("prediction generation failed: {}", exc)
            return {"status": "error", "error": str(exc)}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_retrain_criteria(
        self, eval_result: Dict, force: bool
    ) -> tuple[bool, str]:
        if force:
            return True, "forced"

        # Count newly settled games since last training
        n_settled = self._count_new_settled_games()
        if n_settled >= settings.retrain_min_new_games:
            return True, f"new_settled_games_{n_settled}"

        # Brier degradation
        brier = eval_result.get("overall", {}).get("brier_score")
        if brier and not _is_nan(brier):
            last_brier = self._get_last_champion_brier("match_model")
            if last_brier and brier - last_brier >= settings.retrain_brier_degradation_threshold:
                return True, f"brier_degradation_{brier:.4f}_vs_{last_brier:.4f}"

        return False, "criteria_not_met"

    def _count_new_settled_games(self) -> int:
        """Count games settled since the last training run."""
        try:
            from app.db.models import LegSettlement, ModelVersion
            with SessionLocal() as db:
                last_train = (
                    db.query(ModelVersion)
                    .filter(ModelVersion.model_name == "match_model")
                    .order_by(ModelVersion.created_at.desc())
                    .first()
                )
                if last_train is None:
                    return settings.retrain_min_new_games  # no training yet → trigger

                count = (
                    db.query(LegSettlement)
                    .filter(LegSettlement.settled_at >= last_train.created_at)
                    .filter(LegSettlement.outcome.isnot(None))
                    .count()
                )
                return count
        except Exception:
            return 0

    def _get_last_champion_brier(self, model_name: str) -> Optional[float]:
        try:
            from app.services.model_manager import ModelManager
            champ = ModelManager().get_champion(model_name)
            if champ:
                return champ.brier_score_val
        except Exception:
            pass
        return None

    def _build_summary(self, report: Dict) -> str:
        parts = []
        steps = report.get("steps", {})

        sync = steps.get("sync_upcoming", {})
        if sync.get("n_synced", 0):
            parts.append(f"synced {sync['n_synced']} fixtures")

        settle = steps.get("settle_recent", {})
        leg_s = settle.get("leg_settlement", {})
        if leg_s.get("n_settled", 0):
            parts.append(f"settled {leg_s['n_settled']} legs")

        ev = steps.get("evaluation", {})
        overall = ev.get("overall", {})
        if overall.get("brier_score") and not _is_nan(overall["brier_score"]):
            parts.append(f"brier={overall['brier_score']:.4f}")

        retrain = steps.get("retrain", {})
        if retrain.get("retrained"):
            parts.append(f"retrained ({retrain.get('reason', '')})")

        promo = steps.get("promotion", {})
        promotions = promo.get("promotions", {})
        promoted = [k for k, v in promotions.items() if v.get("promoted")]
        if promoted:
            parts.append(f"promoted {', '.join(promoted)}")

        preds = steps.get("predictions", {})
        if preds.get("n_legs", 0):
            parts.append(f"generated {preds['n_legs']} legs / {preds.get('n_multis', 0)} multis")

        return "; ".join(parts) if parts else "cycle complete (no changes)"

    def _persist_cycle_log(
        self,
        cycle_id: str,
        started_at: datetime,
        finished_at: datetime,
        report: Dict,
    ) -> None:
        try:
            with SessionLocal() as db:
                log = ResearchCycleLog(
                    cycle_id=cycle_id,
                    started_at=started_at,
                    finished_at=finished_at,
                    elapsed_seconds=(finished_at - started_at).total_seconds(),
                    n_fixtures_synced=report["steps"].get("sync_upcoming", {}).get("n_synced", 0),
                    n_legs_settled=report["steps"].get("settle_recent", {}).get("leg_settlement", {}).get("n_settled", 0),
                    brier_score=report["steps"].get("evaluation", {}).get("overall", {}).get("brier_score"),
                    roi=report["steps"].get("evaluation", {}).get("overall", {}).get("roi"),
                    retrained=report["steps"].get("retrain", {}).get("retrained", False),
                    promoted=bool(
                        [v for v in report["steps"].get("promotion", {}).get("promotions", {}).values() if v.get("promoted")]
                    ),
                    summary=report.get("summary", ""),
                )
                db.add(log)
                db.commit()
        except Exception as exc:
            logger.warning("Could not persist cycle log: {}", exc)


def _is_nan(v) -> bool:
    try:
        import math
        return math.isnan(v)
    except Exception:
        return False
