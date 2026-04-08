#!/usr/bin/env python3
"""Sportradar sync script.

Usage
-----
    python scripts/sync_sportradar.py --mode upcoming
    python scripts/sync_sportradar.py --mode settle_recent
    python scripts/sync_sportradar.py --mode backfill --season-ids sr:season:12345
    python scripts/sync_sportradar.py --mode status

Modes
-----
upcoming
    Pull fixtures for the next N days and upsert to DB.
settle_recent
    Fetch completed game results and settle open predictions.
backfill
    Bulk-load historical data for one or more season IDs (costs quota).
status
    Show data-source and quota status without making changes.
"""
import argparse
import json
import sys
from pathlib import Path

# Ensure the app package is importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def cmd_upcoming(args):
    from app.services.sync import SyncService
    svc = SyncService()
    result = svc.sync_upcoming(lookahead_days=args.lookahead_days)
    _print(result)


def cmd_settle_recent(args):
    from app.services.sync import SyncService
    svc = SyncService()
    result = svc.sync_settle_recent(lookback_days=args.lookback_days)
    _print(result)


def cmd_backfill(args):
    season_ids = args.season_ids or []
    if not season_ids:
        print("ERROR: --season-ids is required for backfill mode", file=sys.stderr)
        sys.exit(1)
    from app.services.sync import SyncService
    svc = SyncService()
    result = svc.sync_backfill(season_ids=season_ids, max_fixtures=args.max_fixtures)
    _print(result)


def cmd_status(args):
    from app.services.sync import SyncService
    svc = SyncService()
    result = svc.get_sync_status()
    _print(result)


def _print(data):
    print(json.dumps(data, indent=2, default=str))


def main():
    parser = argparse.ArgumentParser(
        description="Sync AFL data from Sportradar",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["upcoming", "settle_recent", "backfill", "status"],
        required=True,
        help="Sync operation to run",
    )
    parser.add_argument(
        "--lookahead-days",
        type=int,
        default=14,
        help="Days ahead for 'upcoming' mode (default: 14)",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=7,
        help="Days back for 'settle_recent' mode (default: 7)",
    )
    parser.add_argument(
        "--season-ids",
        nargs="+",
        metavar="SEASON_ID",
        help="Sportradar season IDs for 'backfill' mode",
    )
    parser.add_argument(
        "--max-fixtures",
        type=int,
        default=50,
        help="Max fixtures to backfill (default: 50)",
    )

    args = parser.parse_args()

    dispatch = {
        "upcoming": cmd_upcoming,
        "settle_recent": cmd_settle_recent,
        "backfill": cmd_backfill,
        "status": cmd_status,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()
