"""
Database seeding — live-data system only.

Demo CSV seeding has been removed. The database is populated exclusively
from live Sportradar API data via the SyncService.

To populate the database:
  1. Configure SPORTRADAR_API_KEY and SPORTRADAR_AFL_SEASON_ID in .env
  2. Run: python scripts/sync_sportradar.py
     or POST /api/v1/sync/upcoming to sync upcoming fixtures

Any import of seed_from_csv will raise RuntimeError to prevent accidental
use of the removed demo seeding path.
"""


def seed_from_csv(*args, **kwargs):
    raise RuntimeError(
        "seed_from_csv has been removed. The database is populated from live "
        "Sportradar API data via SyncService.\n"
        "Run 'python scripts/sync_sportradar.py' or POST /api/v1/sync/upcoming."
    )
