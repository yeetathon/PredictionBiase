#!/usr/bin/env python3
"""Run AFL prediction research cycle with detailed reporting.

This script orchestrates: evaluation → report generation → optional retraining.
It is separate from the bootstrap cycle — use this for ad-hoc research without
triggering sync or settlement.

Usage
-----
    python scripts/run_research_cycle.py
    python scripts/run_research_cycle.py --lookback-days 60
    python scripts/run_research_cycle.py --retrain
    python scripts/run_research_cycle.py --backtest
    python scripts/run_research_cycle.py --json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Run AFL research cycle")
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=30,
        help="Days of settled data to evaluate (default: 30)",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Retrain models after evaluation",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run walk-forward backtest",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Output as JSON",
    )
    args = parser.parse_args()

    results = {}

    # ── Self-evaluation ────────────────────────────────────────────────
    print("Running self-evaluation...", flush=True)
    from app.services.evaluation import EvaluationService
    eval_svc = EvaluationService()
    eval_report = eval_svc.evaluate(lookback_days=args.lookback_days)
    results["evaluation"] = eval_report

    # ── Backtest ───────────────────────────────────────────────────────
    if args.backtest:
        print("Running walk-forward backtest...", flush=True)
        from app.services.backtest import WalkForwardBacktester
        backtester = WalkForwardBacktester()
        backtest_report = backtester.run()
        results["backtest"] = backtest_report

    # ── Retrain ────────────────────────────────────────────────────────
    if args.retrain:
        print("Retraining models...", flush=True)
        from app.services.training import TrainingService
        train_svc = TrainingService()
        train_result = train_svc.run_all()
        results["training"] = train_result

        # Promotion check
        from app.services.model_manager import ModelManager
        mm = ModelManager()
        promotions = {}
        for model_name in ["match_model", "player_disposals_model"]:
            promotions[model_name] = mm.maybe_promote(model_name)
        results["promotions"] = promotions

    # ── Report generation ──────────────────────────────────────────────
    print("Generating reports...", flush=True)
    from app.services.reports import ReportsService
    report_svc = ReportsService()
    results["summary"] = report_svc.get_summary()

    if args.as_json:
        print(json.dumps(results, indent=2, default=str))
    else:
        _print_human(results)


def _print_human(results: dict):
    print(f"\n{'='*60}")
    print("Research Cycle Report")
    print(f"{'='*60}")

    ev = results.get("evaluation", {})
    overall = ev.get("overall", {})
    print(f"\nEvaluation (n={ev.get('n_legs', 0)} settled legs):")
    if overall:
        bs = overall.get("brier_score")
        roi = overall.get("roi")
        clv = overall.get("clv")
        hr = overall.get("hit_rate")
        print(f"  Brier score : {bs:.4f}" if bs else "  Brier score : n/a")
        print(f"  ROI         : {roi:.3f}" if roi else "  ROI         : n/a")
        print(f"  CLV         : {clv:.4f}" if clv else "  CLV         : n/a")
        print(f"  Hit rate    : {hr:.3f}" if hr else "  Hit rate    : n/a")
    else:
        print("  No settled data available")

    by_market = ev.get("by_market_type", {})
    if by_market:
        print("\nBy market type:")
        for mt, m in by_market.items():
            bs = m.get("brier_score")
            roi = m.get("roi")
            print(f"  {mt:25s} n={m.get('n',0):4d} | brier={bs:.4f if bs else 'n/a':>8} | roi={roi:.3f if roi else 'n/a':>8}")

    bias = ev.get("bias_by_odds_range", [])
    if bias:
        print("\nBias by odds range:")
        for b in bias:
            print(f"  {b['odds_range']:18s} n={b['n']:4d} | bias={b['bias']:+.4f}")

    failures = ev.get("failure_patterns", {})
    if failures:
        print("\nFailure patterns:")
        oc = failures.get("overconfident_losses", {})
        if oc:
            print(f"  Overconfident losses: {oc.get('count', 0)} ({oc.get('pct_of_total', 0):.1%})")

    bt = results.get("backtest")
    if bt:
        metrics = bt.get("overall_metrics", {})
        print(f"\nBacktest: brier={metrics.get('brier_score', 'n/a')} roi={metrics.get('roi', 'n/a')}")

    promos = results.get("promotions", {})
    if promos:
        print("\nModel promotions:")
        for name, r in promos.items():
            status = "PROMOTED" if r.get("promoted") else f"not promoted ({r.get('reason', '')})"
            print(f"  {name}: {status}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
