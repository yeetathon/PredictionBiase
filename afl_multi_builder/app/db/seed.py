"""
Database seeding — live-data system only.

Demo CSV seeding has been removed. The database is populated exclusively
from the AFL Data Sports Group API via the SyncService.

To populate the database:
  1. Configure AFL_DATA_AUTHKEY in .env
  2. POST /api/v1/sync/upcoming to sync upcoming fixtures

Any import of seed_from_csv will raise RuntimeError to prevent accidental
use of the removed demo seeding path.
"""


def seed_from_csv(*args, **kwargs):
    raise RuntimeError(
        "seed_from_csv has been removed. The database is populated from live "
        "AFL Data Sports Group API via SyncService.\n"
        "POST /api/v1/sync/upcoming to sync upcoming fixtures."
    )
