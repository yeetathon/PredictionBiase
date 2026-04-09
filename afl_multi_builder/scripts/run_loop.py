#!/usr/bin/env python3
"""
AFL Prediction continuous loop runner.

Modes
-----
--once      Run pipeline once and exit (default)
--loop      Run pipeline continuously on a configurable interval

Steps each cycle
----------------
1. Sync upcoming fixtures (if Sportradar configured)
2. Settle recently completed games
3. Run predictions → save to data/latest_predictions.json
4. Log performance summary
5. (Optional) retrain models when criteria met

Usage
-----
    # Run once
    python scripts/run_loop.py --once

    # Loop every 30 minutes
    python scripts/run_loop.py --loop --interval 30

    # Loop with forced model retrain every cycle
    python scripts/run_loop.py --loop --interval 60 --retrain

    # Loop but skip sync (demo/offline mode)
    python scripts/run_loop.py --loop --interval 15 --no-sync
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Make the app package importable when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUTPUT_PATH = Path("data/latest_predictions.json")


def log(msg: str, level: str = "INFO"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


def run_cycle(args) -> dict:
    """Execute one full prediction + sync cycle. Returns a summary dict."""
    cycle_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log(f"=== Cycle {cycle_id} starting ===")
    summary = {"cycle_id": cycle_id, "started_at": datetime.now().isoformat()}

    # ── Step 1: Sync fixtures ──────────────────────────────────────
    if not args.no_sync:
        try:
            from app.core.config import settings
            if settings.is_sportradar_configured:
                log("Syncing upcoming fixtures from Sportradar…")
                from app.services.sync import SyncService
                svc = SyncService()
                sync_result = svc.sync_upcoming(lookahead_days=14)
                log(f"Sync complete: {sync_result.get('n_synced', 0)} fixtures synced "
                    f"({sync_result.get('source', 'unknown')})")
                summary["sync"] = sync_result
            else:
                log("Sportradar not configured — skipping sync (demo mode)")
                summary["sync"] = {"status": "skipped", "reason": "no_api_key"}
        except Exception as exc:
            log(f"Sync failed: {exc}", "WARN")
            summary["sync"] = {"status": "error", "error": str(exc)}

    # ── Step 2: Settle completed games ────────────────────────────
    try:
        from app.core.config import settings
        if settings.is_sportradar_configured:
            log("Settling completed games…")
            from app.services.sync import SyncService
            svc = SyncService()
            settle_result = svc.sync_settle_recent(lookback_days=7)
            n_settled = settle_result.get("leg_settlement", {}).get("n_settled", 0)
            log(f"Settlement complete: {n_settled} legs settled")
            summary["settlement"] = settle_result
        else:
            summary["settlement"] = {"status": "skipped"}
    except Exception as exc:
        log(f"Settlement failed: {exc}", "WARN")
        summary["settlement"] = {"status": "error", "error": str(exc)}

    # ── Step 3: Run predictions ───────────────────────────────────
    log("Running prediction pipeline…")
    try:
        from app.services.pipeline import PredictionPipeline
        pipeline = PredictionPipeline()
        result = pipeline.run()

        n_legs = result.get("n_candidate_legs", 0)
        n_value = len(result.get("value_legs", []))
        n_multis = (
            len(result.get("value_multis", []))
            + len(result.get("safe_multis", []))
            + len(result.get("same_game_multis", []))
        )
        log(f"Pipeline complete: {n_legs} legs, {n_value} value picks, {n_multis} multis "
            f"(source: {result.get('data_source', '?')}, {result.get('elapsed_seconds', 0):.1f}s)")
        summary["predictions"] = {
            "n_candidate_legs": n_legs,
            "n_value_legs": n_value,
            "n_multis": n_multis,
            "data_source": result.get("data_source"),
            "run_id": result.get("run_id"),
        }

        # Save to disk
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_PATH, "w") as f:
            json.dump(result, f, indent=2, default=str)
        log(f"Predictions saved → {OUTPUT_PATH}")

        # Print top 5 value legs
        _print_top_legs(result.get("value_legs", [])[:5])

    except Exception as exc:
        log(f"Pipeline failed: {exc}", "ERROR")
        summary["predictions"] = {"status": "error", "error": str(exc)}
        import traceback
        traceback.print_exc()

    # ── Step 4: Self-evaluation ────────────────────────────────────
    try:
        from app.services.evaluation import EvaluationService
        eval_svc = EvaluationService()
        eval_report = eval_svc.evaluate(lookback_days=30)
        if eval_report.get("status") == "ok":
            overall = eval_report.get("overall", {})
            brier = overall.get("brier_score")
            roi   = overall.get("roi")
            log(f"Self-evaluation: brier={brier:.4f if brier else 'n/a'} roi={roi:.3f if roi else 'n/a'} "
                f"n={eval_report.get('n_legs', 0)}")
            summary["evaluation"] = {
                "brier_score": brier,
                "roi": roi,
                "n_settled_legs": eval_report.get("n_legs", 0),
            }
        else:
            summary["evaluation"] = {"status": eval_report.get("status", "no_data")}
    except Exception as exc:
        log(f"Evaluation failed: {exc}", "WARN")
        summary["evaluation"] = {"status": "error", "error": str(exc)}

    # ── Step 5: Retrain if requested or criteria met ───────────────
    if args.retrain:
        log("Forced model retrain requested…")
        try:
            from app.services.training import TrainingService
            svc = TrainingService()
            train_result = svc.run_all()
            log("Retrain complete")
            summary["retrain"] = {"status": "ok"}

            # Promotion check
            from app.services.model_manager import ModelManager
            mm = ModelManager()
            promotions = {}
            for mname in ["match_model", "player_disposals_model"]:
                p = mm.maybe_promote(mname)
                promotions[mname] = p
                if p.get("promoted"):
                    log(f"Model promoted: {mname} (brier {p.get('brier_val', '?'):.4f})")
            summary["promotions"] = promotions
        except Exception as exc:
            log(f"Retrain failed: {exc}", "WARN")
            summary["retrain"] = {"status": "error", "error": str(exc)}

    summary["finished_at"] = datetime.now().isoformat()
    log(f"=== Cycle {cycle_id} complete ===\n")
    return summary


def _print_top_legs(legs: list):
    if not legs:
        return
    print("\n  Top Value Picks:")
    print(f"  {'Match':<35} {'Pick':<35} {'Odds':>6}  {'Model':>7}  {'Edge':>7}  {'Conf':<8}")
    print("  " + "-" * 100)
    for leg in legs:
        match  = leg.get("match_label", "?")[:34]
        pick   = leg.get("selection_label", leg.get("selection", "?"))[:34]
        odds   = leg.get("decimal_odds", 0)
        model  = leg.get("model_probability", 0)
        edge   = leg.get("edge", 0)
        conf   = leg.get("confidence_label", "?")
        print(f"  {match:<35} {pick:<35} ${odds:>5.2f}  {model:>6.1%}  {edge:>+6.1%}  {conf:<8}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="AFL Prediction loop runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--once", action="store_true", default=True,
                             help="Run once and exit (default)")
    mode_group.add_argument("--loop", action="store_true",
                             help="Run continuously on interval")
    parser.add_argument("--interval", type=int, default=30,
                        help="Loop interval in minutes (default: 30)")
    parser.add_argument("--retrain", action="store_true",
                        help="Force model retrain each cycle")
    parser.add_argument("--no-sync", action="store_true",
                        help="Skip Sportradar sync (use cached/demo data only)")

    args = parser.parse_args()

    # If --loop is given, once is effectively False
    if args.loop:
        args.once = False

    if args.loop:
        log(f"Starting continuous loop: interval={args.interval}min, retrain={args.retrain}")
        cycle_num = 0
        while True:
            cycle_num += 1
            log(f"Starting cycle #{cycle_num}")
            try:
                run_cycle(args)
            except KeyboardInterrupt:
                log("Interrupted by user. Exiting.")
                sys.exit(0)
            except Exception as exc:
                log(f"Cycle #{cycle_num} failed: {exc}", "ERROR")
                import traceback
                traceback.print_exc()

            log(f"Sleeping {args.interval} minutes until next cycle…")
            try:
                time.sleep(args.interval * 60)
            except KeyboardInterrupt:
                log("Interrupted by user. Exiting.")
                sys.exit(0)
    else:
        run_cycle(args)


if __name__ == "__main__":
    main()
