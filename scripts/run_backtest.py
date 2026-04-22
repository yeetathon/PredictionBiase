#!/usr/bin/env python3
"""Run walk-forward backtesting."""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.logging import setup_logging
from app.services.backtest import WalkForwardBacktester
from app.services.reports import ReportsService


def main():
    import os
    os.makedirs("data/models", exist_ok=True)
    os.makedirs("data/logs", exist_ok=True)

    print("=" * 60)
    print("AFL Multi Builder - Walk-Forward Backtest")
    print("=" * 60)

    backtester = WalkForwardBacktester()
    results = backtester.run()

    report_svc = ReportsService()
    print(report_svc.format_backtest_report(results))

    # Save results
    out_path = Path("data/backtest_results.json")
    with open(out_path, "w") as f:
        safe = json.loads(json.dumps(results, default=str))
        json.dump(safe, f, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
