#!/usr/bin/env python3
"""
REMOVED — seed_demo_data.py has been deprecated.

This script previously seeded the database with local demo CSV files.
The system now requires live data from real APIs only.

There is no demo seeding, no CSV seeding, and no offline fallback.

Setup instructions:
  1. Configure your API keys in .env:
       AFL_DATA_AUTHKEY=<your_key>
       ODDS_API_KEY=<your_key>  (optional)

  2. Verify your setup:
       python -c "from app.services.preflight import PreflightService; PreflightService().run()"

  3. Sync live data:
       python scripts/sync_afl_data.py --mode upcoming

  4. Train models on live data:
       python scripts/run_training.py

  5. Run the pipeline:
       python scripts/run_pipeline.py

See README.md for full setup instructions.
"""

import sys

print(
    "\n[ERROR] seed_demo_data.py is no longer available.\n"
    "\n"
    "This system uses live data from real APIs only. There is no demo mode.\n"
    "\n"
    "Setup steps:\n"
    "  1. Add AFL_DATA_AUTHKEY to your .env file\n"
    "  2. Optionally add ODDS_API_KEY for bookmaker edge calculations\n"
    "  3. Run: python scripts/sync_afl_data.py --mode upcoming\n"
    "  4. Run: python scripts/run_training.py\n"
    "\n"
    "See README.md for full setup instructions.\n",
    file=sys.stderr
)
sys.exit(1)
