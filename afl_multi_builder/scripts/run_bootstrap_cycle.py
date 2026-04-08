#!/usr/bin/env python3
"""Run one full bootstrap research cycle.

The cycle executes these steps in order:
  1. Sync upcoming fixtures
  2. Settle recently completed games
  3. Self-evaluate (Brier, CLV, ROI, calibration)
  4. Retrain models if criteria met (or --force-retrain)
  5. Promote challenger if Brier improves > threshold
  6. Generate fresh predictions for upcoming fixtures

Usage
-----
    python scripts/run_bootstrap_cycle.py
    python scripts/run_bootstrap_cycle.py --force-retrain
    python scripts/run_bootstrap_cycle.py --json          # machine-readable output
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Run AFL bootstrap research cycle")
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force model retraining regardless of criteria",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Output full report as JSON (default: human-readable summary)",
    )
    args = parser.parse_args()

    from app.services.bootstrap import BootstrapCycleService

    print("Starting bootstrap cycle...", flush=True)
    svc = BootstrapCycleService()
    report = svc.run(force_retrain=args.force_retrain)

    if args.as_json:
        print(json.dumps(report, indent=2, default=str))
    else:
        _print_human(report)


def _print_human(report: dict):
    print(f"\n{'='*60}")
    print(f"Bootstrap Cycle  {report.get('cycle_id', '')}  ({report.get('started_at', '')})")
    print(f"{'='*60}")

    steps = report.get("steps", {})

    sync = steps.get("sync_upcoming", {})
    print(f"  Sync upcoming : {sync.get('status', '?')} — {sync.get('n_synced', 0)} fixtures")

    settle = steps.get("settle_recent", {})
    leg_s = settle.get("leg_settlement", {})
    print(f"  Settle recent : {settle.get('status', '?')} — {leg_s.get('n_settled', 0)} legs settled")

    ev = steps.get("evaluation", {})
    overall = ev.get("overall", {})
    brier = overall.get("brier_score")
    roi = overall.get("roi")
    print(f"  Evaluation    : n={report.get('n_legs', 0)} | "
          f"brier={brier:.4f if brier else 'n/a'} | roi={roi:.3f if roi else 'n/a'}")

    retrain = steps.get("retrain", {})
    print(f"  Retrain       : {'YES — ' + retrain.get('reason','') if retrain.get('retrained') else 'No — ' + retrain.get('reason','')}")

    promo = steps.get("promotion", {})
    for model_name, r in promo.get("promotions", {}).items():
        print(f"  Promotion [{model_name}]: {'PROMOTED' if r.get('promoted') else 'not promoted'}")

    preds = steps.get("predictions", {})
    print(f"  Predictions   : {preds.get('n_legs', 0)} legs / {preds.get('n_multis', 0)} multis")

    print(f"\n  Summary: {report.get('summary', '')}")
    print(f"  Elapsed: {report.get('elapsed_seconds', 0):.1f}s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
