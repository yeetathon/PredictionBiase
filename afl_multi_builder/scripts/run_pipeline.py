#!/usr/bin/env python3
"""Run the full prediction pipeline and print results."""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.logging import setup_logging
from app.services.pipeline import PredictionPipeline


def main():
    import os
    os.makedirs("data/models", exist_ok=True)
    os.makedirs("data/logs", exist_ok=True)

    print("=" * 60)
    print("AFL Multi Builder - Prediction Pipeline")
    print("=" * 60)

    pipeline = PredictionPipeline()
    results = pipeline.run()

    print(f"\nRun ID: {results['run_id']}")
    print(f"Fixtures processed: {results['n_fixtures']}")
    print(f"Candidate legs: {results['n_candidate_legs']}")
    print(f"Elapsed: {results['elapsed_seconds']:.2f}s")

    print("\n── Top Value Legs ──────────────────────────────────")
    for leg in results["value_legs"][:5]:
        print(
            f"  {leg['selection']:<30} "
            f"odds={leg['decimal_odds']:.2f}  "
            f"prob={leg['model_probability']:.1%}  "
            f"EV={leg['ev']:+.1%}  "
            f"conf={leg['confidence_score']:.0f}"
        )

    print("\n── Top Value Multis ────────────────────────────────")
    for m in results["value_multis"][:5]:
        print(
            f"  {m['n_legs']}-leg {m['multi_type']:<12} "
            f"odds=${m['combined_odds']:.2f}  "
            f"adj_prob={m['adjusted_probability']:.1%}  "
            f"EV={m['ev']:+.1%}  "
            f"corr={m['correlation_label']}"
        )

    print("\n── Same-Game Multis ────────────────────────────────")
    for m in results["same_game_multis"][:3]:
        print(
            f"  {m['n_legs']}-leg SGM  "
            f"odds=${m['combined_odds']:.2f}  "
            f"EV={m['ev']:+.1%}"
        )

    # Save
    out_path = Path("data/pipeline_results.json")
    with open(out_path, "w") as f:
        safe = json.loads(json.dumps(results, default=str))
        json.dump(safe, f, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
