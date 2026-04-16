#!/usr/bin/env python3
"""Train all prediction models."""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.logging import setup_logging
from app.services.training import TrainingService
from app.services.reports import ReportsService


def main():
    import os
    os.makedirs("data/models", exist_ok=True)
    os.makedirs("data/logs", exist_ok=True)

    print("=" * 60)
    print("AFL Multi Builder - Model Training")
    print("=" * 60)

    svc = TrainingService()
    results = svc.run_all()

    report_svc = ReportsService()
    print(report_svc.format_training_report(results))

    # Save results to file
    out_path = Path("data/training_results.json")
    with open(out_path, "w") as f:
        # Filter non-serializable items
        safe_results = json.loads(json.dumps(results, default=str))
        json.dump(safe_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
