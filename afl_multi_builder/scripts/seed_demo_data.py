#!/usr/bin/env python3
"""Seed the database from demo CSV files."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.logging import setup_logging
from app.db.database import init_db, SessionLocal
from app.db.seed import seed_from_csv
from app.core.config import settings


def main():
    import os
    os.makedirs("data/models", exist_ok=True)
    os.makedirs("data/logs", exist_ok=True)

    print("Initialising database...")
    init_db()

    print(f"Seeding from {settings.demo_data_dir}...")
    db = SessionLocal()
    try:
        seed_from_csv(db, settings.demo_data_dir)
        print("Done. Database seeded successfully.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
