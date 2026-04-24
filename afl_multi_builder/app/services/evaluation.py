"""Self-evaluation service: Brier, log-loss, CLV, ROI, calibration bias.

Called after each settlement batch to quantify model quality and detect
systematic biases (market type, team, player, odds range, etc.).
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from sqlalchemy.orm import Session

from app.core.adaptive_config import get_adaptive_config
from app.db.database import SessionLocal
from app.db.models import LegSettlement, MultiSettlement, ResearchCycleLog


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean((y_prob - y_true) ** 2))


def _log_loss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-7) -> float:
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return float("nan")
    p = np.clip(y_prob, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def _roi(y_true: np.ndarray, decimal_odds: np.ndarray) -> float:
    """Simple flat-stake ROI assuming 1 unit staked on each bet."""
    if len(y_true) == 0:
        return float("nan")
    profit = np.where(y_true == 1, decimal_odds - 1, -1.0)
    return float(np.sum(profit) / len(y_true))


def _clv(model_prob: np.ndarray, closing_odds: np.ndarray) -> float:
    """Closing Line Value: average (model_prob - 1/closing_odds)."""
    if len(model_prob) == 0:
        return float("nan")
    market_implied = np.where(closing_odds > 0, 1.0 / closing_odds, np.nan)
    valid = ~np.isnan(market_implied)
    if not valid.any():
        return float("nan")
    return float(np.mean(model_prob[valid] - market_implied[valid]))


def _calibration_bins(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> List[Dict[str, float]]:
    """Return calibration data: each bin's mean predicted prob vs actual hit rate."""
    bins = []
    edges = np.linspace(0, 1, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        bins.append({
            "bin_lo": round(float(lo), 3),
            "bin_hi": round(float(hi), 3),
            "n": int(mask.sum()),
            "mean_predicted": round(float(y_prob[mask].mean()), 4),
            "actual_hit_rate": round(float(y_true[mask].mean()), 4),
        })
    return bins


# ---------------------------------------------------------------------------
# EvaluationService
# ---------------------------------------------------------------------------

class EvaluationService:
    """Compute and store self-evaluation metrics after settlement.

    Usage::

        svc = EvaluationService()
        report = svc.evaluate(lookback_days=30)
        print(report["overall"]["brier_score"])
    """

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, lookback_days: int = 30) -> Dict[str, Any]:
        """Run full evaluation over recently settled legs.

        Returns a structured report dict with overall metrics, per-market-type
        breakdown, calibration bins, bias analysis, and CLV.
        """
        report: Dict[str, Any] = {
            "run_id": str(uuid.uuid4())[:8],
            "timestamp": datetime.utcnow().isoformat(),
            "lookback_days": lookback_days,
        }

        with SessionLocal() as db:
            legs = self._load_settled_legs(db, lookback_days)

        if not legs:
            logger.info("EvaluationService: no settled legs in last {} days", lookback_days)
            report["status"] = "no_data"
            return report

        report["n_legs"] = len(legs)

        # Overall metrics
        y_true = np.array([l["outcome"] for l in legs], dtype=float)
        y_prob = np.array([l["model_probability"] for l in legs], dtype=float)
        odds = np.array([l["decimal_odds"] for l in legs], dtype=float)

        report["overall"] = {
            "brier_score": _brier_score(y_true, y_prob),
            "log_loss": _log_loss(y_true, y_prob),
            "roi": _roi(y_true, odds),
            "clv": _clv(y_prob, odds),
            "hit_rate": float(y_true.mean()),
            "avg_edge": float((y_prob - 1.0 / np.where(odds > 0, odds, np.nan)).mean()),
        }

        # Per market-type breakdown
        market_types = list({l["market_type"] for l in legs})
        report["by_market_type"] = {}
        for mt in market_types:
            sub = [l for l in legs if l["market_type"] == mt]
            yt = np.array([l["outcome"] for l in sub], dtype=float)
            yp = np.array([l["model_probability"] for l in sub], dtype=float)
            od = np.array([l["decimal_odds"] for l in sub], dtype=float)
            raw_imp = np.where(od > 0, 1.0 / od, np.nan)
            vig_adj = np.array([l.get("vig_adj_prob", 1.0 / l["decimal_odds"]) for l in sub], dtype=float)
            predicted_edge = yp - vig_adj
            realized_edge = np.where(yt == 1, od - 1 - vig_adj, -vig_adj)
            report["by_market_type"][mt] = {
                "n": len(sub),
                "brier_score": _brier_score(yt, yp),
                "roi": _roi(yt, od),
                "hit_rate": float(yt.mean()) if len(yt) > 0 else float("nan"),
                "log_loss": _log_loss(yt, yp),
                "clv": _clv(yp, od),
                "avg_predicted_edge": round(float(np.nanmean(predicted_edge)), 4),
                "avg_realized_edge": round(float(np.nanmean(realized_edge)), 4),
                "edge_realization_ratio": round(
                    float(np.nanmean(realized_edge) / max(abs(np.nanmean(predicted_edge)), 0.001)), 3
                ),
            }

        # Calibration
        report["calibration"] = _calibration_bins(y_true, y_prob)

        # Bias analysis by odds bucket
        report["bias_by_odds_range"] = self._bias_by_odds_range(legs)

        # Edge realization by decile
        report["edge_realization_by_decile"] = self._edge_realization_by_decile(legs)

        # Player prop deep analysis
        player_legs = [l for l in legs if l["market_type"] == "player_disposals"]
        if player_legs:
            report["player_prop_analysis"] = self._analyze_player_props(player_legs)

        # Correlation effectiveness: multis
        with SessionLocal() as db:
            multi_metrics = self._multi_effectiveness(db, lookback_days)
        report["multi_effectiveness"] = multi_metrics

        # Failure pattern analysis
        report["failure_patterns"] = self._detect_failure_patterns(legs)

        report["status"] = "ok"
        logger.info(
            "Evaluation complete: brier={:.4f} roi={:.3f} n={}",
            report["overall"]["brier_score"],
            report["overall"]["roi"],
            len(legs),
        )

        # Update adaptive thresholds based on this evaluation
        try:
            adaptive_cfg = get_adaptive_config()
            adaptive_cfg.update_from_evaluation(report)
            report["adaptive_thresholds_updated"] = True
        except Exception as e:
            logger.warning("Could not update adaptive thresholds: {}", e)
            report["adaptive_thresholds_updated"] = False

        # Attach market health summary
        report["market_health"] = self.generate_market_health_report(report)

        return report

    def generate_market_health_report(self, eval_report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Structured market health report: healthy / watch / degraded / insufficient_data.
        Fed into adaptive threshold updates and the nightly report email.
        """
        by_market = eval_report.get("by_market_type", {})
        health: Dict[str, Any] = {}
        for market, metrics in by_market.items():
            n = metrics.get("n", 0)
            brier = metrics.get("brier_score", 0.25)
            roi = metrics.get("roi", 0.0)
            hit_rate = metrics.get("hit_rate", 0.5)

            if n < 5:
                status = "insufficient_data"
            elif brier > 0.25:
                status = "degraded_worse_than_baseline"
            elif roi < -0.15:
                status = "degraded_negative_roi"
            elif brier < 0.20 and roi >= 0.0:
                status = "healthy"
            else:
                status = "watch"

            health[market] = {
                "status": status,
                "n_bets": n,
                "brier_score": round(brier, 4),
                "roi": round(roi, 4),
                "hit_rate": round(hit_rate, 4),
                "recommendation": (
                    "Continue" if status == "healthy" else
                    "Tighten thresholds" if status == "watch" else
                    "Review urgently" if "degraded" in status else
                    "Collect more data"
                ),
            }
        return health

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_settled_legs(
        self,
        db: Session,
        lookback_days: int,
    ) -> List[Dict[str, Any]]:
        """Load settled legs from the database."""
        try:
            from datetime import timedelta
            cutoff = datetime.utcnow() - timedelta(days=lookback_days)
            rows = (
                db.query(LegSettlement)
                .filter(LegSettlement.settled_at >= cutoff)
                .filter(LegSettlement.actual_outcome.isnot(None))
                .all()
            )
            return [
                {
                    "leg_id": r.leg_id,
                    "fixture_id": r.fixture_id,
                    "market_type": r.market_type or "unknown",
                    "selection": r.selection or "",
                    "decimal_odds": r.decimal_odds or 2.0,
                    "model_probability": r.model_probability or 0.5,
                    "outcome": int(r.actual_outcome),
                    # Rich metadata (None if not yet populated)
                    "vig_adj_prob": r.vig_adjusted_probability,
                    "edge_at_prediction": r.edge_at_prediction,
                    "ml_probability": r.ml_probability,
                    "signal_consensus_probability": r.signal_consensus_probability,
                    "signal_agreement": r.signal_agreement,
                    "prediction_variance": r.prediction_variance,
                    "data_completeness": r.data_completeness,
                    "trust_score": r.trust_score,
                    "n_games": r.n_games,
                }
                for r in rows
            ]
        except Exception as exc:
            logger.warning("Could not load settled legs: {}", exc)
            return []

    def _bias_by_odds_range(self, legs: List[Dict]) -> List[Dict[str, Any]]:
        """Detect if model is systematically off in certain odds brackets."""
        buckets = [
            ("favourite", 1.0, 1.8),
            ("slight_favourite", 1.8, 2.3),
            ("even", 2.3, 2.7),
            ("outsider", 2.7, 4.0),
            ("big_outsider", 4.0, 100.0),
        ]
        result = []
        for label, lo, hi in buckets:
            sub = [l for l in legs if lo <= l["decimal_odds"] < hi]
            if not sub:
                continue
            yt = np.array([l["outcome"] for l in sub], dtype=float)
            yp = np.array([l["model_probability"] for l in sub], dtype=float)
            result.append({
                "odds_range": label,
                "odds_lo": lo,
                "odds_hi": hi,
                "n": len(sub),
                "mean_model_prob": round(float(yp.mean()), 4),
                "actual_hit_rate": round(float(yt.mean()), 4),
                "bias": round(float(yp.mean() - yt.mean()), 4),  # positive = overconfident
            })
        return result

    def _multi_effectiveness(
        self, db: Session, lookback_days: int
    ) -> Dict[str, Any]:
        """Analyse how well multi selections are performing."""
        try:
            from datetime import timedelta
            cutoff = datetime.utcnow() - timedelta(days=lookback_days)
            rows = (
                db.query(MultiSettlement)
                .filter(MultiSettlement.settled_at >= cutoff)
                .filter(MultiSettlement.outcome.isnot(None))
                .all()
            )
            if not rows:
                return {"n": 0}

            won = [r for r in rows if r.outcome == 1]
            hit_rate = len(won) / len(rows)
            avg_odds = np.mean([r.combined_odds or 1.0 for r in rows])
            roi = sum((r.combined_odds or 1.0) - 1.0 if r.outcome == 1 else -1.0 for r in rows) / len(rows)

            # Correlation effectiveness: compare adjusted vs naive probability
            adj_probs = np.array([r.adjusted_probability or 0.0 for r in rows])
            naive_probs = np.array([r.raw_probability or 0.0 for r in rows])
            outcomes = np.array([r.outcome for r in rows], dtype=float)

            return {
                "n": len(rows),
                "hit_rate": round(hit_rate, 4),
                "roi": round(roi, 4),
                "avg_odds": round(float(avg_odds), 2),
                "brier_adjusted": _brier_score(outcomes, adj_probs),
                "brier_naive": _brier_score(outcomes, naive_probs),
                "correlation_benefit": round(
                    _brier_score(outcomes, naive_probs) - _brier_score(outcomes, adj_probs), 4
                ),
            }
        except Exception as exc:
            logger.warning("Multi effectiveness analysis failed: {}", exc)
            return {"error": str(exc)}

    def _detect_failure_patterns(self, legs: List[Dict]) -> Dict[str, Any]:
        """Surface systematic failure patterns."""
        patterns: Dict[str, Any] = {}

        # Overconfidence: model prob > 0.7 but outcome = 0
        overconf = [l for l in legs if l["model_probability"] > 0.70 and l["outcome"] == 0]
        patterns["overconfident_losses"] = {
            "count": len(overconf),
            "pct_of_total": round(len(overconf) / max(len(legs), 1), 3),
        }

        # Underdog wins where model < 0.35
        underdog_wins = [l for l in legs if l["model_probability"] < 0.35 and l["outcome"] == 1]
        patterns["underdog_wins_missed"] = {
            "count": len(underdog_wins),
            "pct_of_total": round(len(underdog_wins) / max(len(legs), 1), 3),
        }

        # Market-specific issues
        for mt in {"player_disposals", "head_to_head", "line", "total"}:
            sub = [l for l in legs if l["market_type"] == mt]
            if len(sub) < 5:
                continue
            hit = np.mean([l["outcome"] for l in sub])
            exp_hit = np.mean([l["model_probability"] for l in sub])
            patterns[f"{mt}_bias"] = round(float(exp_hit - hit), 4)

        return patterns

    def _edge_realization_by_decile(self, legs: List[Dict]) -> List[Dict[str, Any]]:
        """
        Group legs by predicted-edge decile and report realized vs predicted edge.
        High realization ratio → model edge is real.
        """
        if not legs:
            return []
        predicted_edges = []
        for l in legs:
            # Prefer stored edge_at_prediction, then vig_adj_prob, then raw implied fallback
            if l.get("edge_at_prediction") is not None:
                predicted_edges.append(float(l["edge_at_prediction"]))
            else:
                vig_adj = l.get("vig_adj_prob") or (1.0 / max(l["decimal_odds"], 1.01))
                predicted_edges.append(float(l["model_probability"]) - float(vig_adj))

        edges = np.array(predicted_edges)
        outcomes = np.array([l["outcome"] for l in legs], dtype=float)
        odds = np.array([l["decimal_odds"] for l in legs], dtype=float)
        realized = np.where(outcomes == 1, odds - 1, -1.0)

        n_bins = min(10, len(legs) // 5)
        if n_bins < 2:
            return []

        percentiles = np.linspace(0, 100, n_bins + 1)
        result = []
        for lo_pct, hi_pct in zip(percentiles[:-1], percentiles[1:]):
            lo = float(np.percentile(edges, lo_pct))
            hi = float(np.percentile(edges, hi_pct))
            mask = (edges >= lo) & (edges < hi) if hi_pct < 100 else (edges >= lo) & (edges <= hi)
            if mask.sum() == 0:
                continue
            mean_pred = float(np.mean(edges[mask]))
            mean_real = float(np.mean(realized[mask]))
            n = int(mask.sum())
            result.append({
                "decile": f"{lo_pct:.0f}-{hi_pct:.0f}%",
                "n": n,
                "avg_predicted_edge": round(mean_pred, 4),
                "avg_realized_edge": round(mean_real, 4),
                "realization_ratio": round(
                    mean_real / max(abs(mean_pred), 0.001), 3
                ),
                "hit_rate": round(float(np.mean(outcomes[mask])), 4),
            })
        return result

    def _analyze_player_props(self, player_legs: List[Dict]) -> Dict[str, Any]:
        """
        Deep analysis of player prop performance stratified by:
          - n_games bucket (sample size)
          - consistency bucket (high/medium/low CV)
          - line range
          - trust bucket (if available)
        """
        result: Dict[str, Any] = {}
        total = len(player_legs)
        if total == 0:
            return result

        # Helper: compute metrics for a subset
        def _metrics(subset: List[Dict]) -> Dict:
            if not subset:
                return {"n": 0}
            yt = np.array([l["outcome"] for l in subset], dtype=float)
            yp = np.array([l["model_probability"] for l in subset], dtype=float)
            od = np.array([l["decimal_odds"] for l in subset], dtype=float)
            vig_adj = np.array([
                l.get("vig_adj_prob") if l.get("vig_adj_prob") is not None
                else 1.0 / max(l["decimal_odds"], 1.01)
                for l in subset
            ], dtype=float)
            pred_edge = yp - vig_adj
            real_edge = np.where(yt == 1, od - 1 - vig_adj, -vig_adj)
            return {
                "n": len(subset),
                "brier_score": round(_brier_score(yt, yp), 4),
                "roi": round(_roi(yt, od), 4),
                "hit_rate": round(float(yt.mean()), 4),
                "avg_predicted_edge": round(float(np.nanmean(pred_edge)), 4),
                "avg_realized_edge": round(float(np.nanmean(real_edge)), 4),
            }

        # By n_games bucket
        result["by_sample_size"] = {}
        for label, lo, hi in [("sparse", 0, 5), ("moderate", 5, 12), ("adequate", 12, 100)]:
            subset = [
                l for l in player_legs
                if lo <= int(l.get("n_games", l.get("metadata_n_games", 0))) < hi
            ]
            result["by_sample_size"][label] = _metrics(subset)

        # By odds range (line difficulty)
        result["by_odds_range"] = {}
        for label, lo, hi in [("favourite", 1.0, 1.8), ("evens", 1.8, 2.3), ("outsider", 2.3, 4.0)]:
            subset = [l for l in player_legs if lo <= l["decimal_odds"] < hi]
            result["by_odds_range"][label] = _metrics(subset)

        # By trust bucket (if trust stored)
        trust_vals = [l.get("trust_score") for l in player_legs if l.get("trust_score") is not None]
        if trust_vals:
            trust_arr = np.array(trust_vals)
            thresholds = [(0, 35, "low_trust"), (35, 55, "medium_trust"), (55, 100, "high_trust")]
            result["by_trust"] = {}
            for lo, hi, label in thresholds:
                subset = [l for l in player_legs
                          if l.get("trust_score") is not None and lo <= l["trust_score"] < hi]
                result["by_trust"][label] = _metrics(subset)

        # Overall false-positive EV rate (predicted +EV but lost)
        ev_pos = [l for l in player_legs if l.get("model_probability", 0.5) > (
            1.0 / max(l["decimal_odds"], 1.01)
        )]
        false_pos = [l for l in ev_pos if l["outcome"] == 0]
        result["false_positive_ev_rate"] = round(len(false_pos) / max(len(ev_pos), 1), 4)
        result["n_total"] = total

        # Recommend disable if chronic underperformance
        all_yt = np.array([l["outcome"] for l in player_legs], dtype=float)
        all_od = np.array([l["decimal_odds"] for l in player_legs], dtype=float)
        overall_roi = _roi(all_yt, all_od)
        if total >= 20 and overall_roi < -0.20:
            result["recommendation"] = "disable_market"
        elif total >= 10 and overall_roi < -0.10:
            result["recommendation"] = "tighten_thresholds"
        else:
            result["recommendation"] = "continue"

        return result
