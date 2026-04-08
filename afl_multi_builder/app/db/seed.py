"""Database seeding from demo CSV files."""
import pandas as pd
from pathlib import Path
from sqlalchemy.orm import Session
from loguru import logger
from app.db.database import SessionLocal, init_db
from app.db.models import Team, Venue, Player, Fixture, PlayerStat, TeamStat, Odds
from app.core.config import settings


def seed_from_csv(db: Session, demo_dir: Path) -> None:
    """Seed all tables from demo CSV files."""
    seed_venues(db, demo_dir / "venues.csv")
    seed_teams(db, demo_dir / "teams.csv")
    seed_players(db, demo_dir / "players.csv")
    seed_fixtures(db, demo_dir / "fixtures.csv")
    seed_team_stats(db, demo_dir / "team_stats.csv")
    seed_player_stats(db, demo_dir / "player_stats.csv")
    seed_odds(db, demo_dir / "odds.csv")
    logger.info("All demo data seeded successfully.")


def seed_venues(db: Session, path: Path) -> None:
    if not path.exists():
        logger.warning(f"Venues file not found: {path}")
        return
    if db.query(Venue).count() > 0:
        logger.info("Venues already seeded, skipping.")
        return
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        venue = Venue(
            id=int(row["venue_id"]),
            name=str(row["name"]),
            city=str(row["city"]),
            state=str(row["state"]),
            capacity=int(row["capacity"]) if pd.notna(row.get("capacity")) else None,
            surface=str(row.get("surface", "")),
            roof=str(row.get("roof", "")),
            timezone=str(row.get("timezone", "")),
        )
        db.merge(venue)
    db.commit()
    logger.info(f"Seeded {len(df)} venues.")


def seed_teams(db: Session, path: Path) -> None:
    if not path.exists():
        logger.warning(f"Teams file not found: {path}")
        return
    if db.query(Team).count() > 0:
        logger.info("Teams already seeded, skipping.")
        return
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        team = Team(
            id=int(row["team_id"]),
            name=str(row["name"]),
            abbreviation=str(row["abbreviation"]),
            home_ground=str(row.get("home_ground", "")),
            state=str(row.get("state", "")),
            founded=int(row["founded"]) if pd.notna(row.get("founded")) else None,
            primary_colour=str(row.get("primary_colour", "")),
            secondary_colour=str(row.get("secondary_colour", "")),
            elo_rating=1500.0,
        )
        db.merge(team)
    db.commit()
    logger.info(f"Seeded {len(df)} teams.")


def seed_players(db: Session, path: Path) -> None:
    if not path.exists():
        logger.warning(f"Players file not found: {path}")
        return
    if db.query(Player).count() > 0:
        logger.info("Players already seeded, skipping.")
        return
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        player = Player(
            id=int(row["player_id"]),
            team_id=int(row["team_id"]),
            name=str(row["name"]),
            position=str(row.get("position", "")),
            dob=str(row.get("dob", "")),
            height_cm=int(row["height_cm"]) if pd.notna(row.get("height_cm")) else None,
            weight_kg=int(row["weight_kg"]) if pd.notna(row.get("weight_kg")) else None,
            debut_year=int(row["debut_year"]) if pd.notna(row.get("debut_year")) else None,
            status=str(row.get("status", "Active")),
        )
        db.merge(player)
    db.commit()
    logger.info(f"Seeded {len(df)} players.")


def seed_fixtures(db: Session, path: Path) -> None:
    if not path.exists():
        logger.warning(f"Fixtures file not found: {path}")
        return
    if db.query(Fixture).count() > 0:
        logger.info("Fixtures already seeded, skipping.")
        return
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        def safe_int(val):
            try:
                if pd.isna(val):
                    return None
                return int(val)
            except (ValueError, TypeError):
                return None

        fixture = Fixture(
            id=int(row["fixture_id"]),
            season=int(row["season"]),
            round=int(row["round"]),
            home_team_id=int(row["home_team_id"]),
            away_team_id=int(row["away_team_id"]),
            venue_id=int(row["venue_id"]),
            date=str(row["date"]),
            time=str(row.get("time", "")),
            result_home_score=safe_int(row.get("result_home_score")),
            result_away_score=safe_int(row.get("result_away_score")),
            home_win=safe_int(row.get("home_win")),
            margin=safe_int(row.get("margin")),
            total_score=safe_int(row.get("total_score")),
            status=str(row.get("status", "upcoming")).strip(),
        )
        db.merge(fixture)
    db.commit()
    logger.info(f"Seeded {len(df)} fixtures.")


def seed_team_stats(db: Session, path: Path) -> None:
    if not path.exists():
        logger.warning(f"Team stats file not found: {path}")
        return
    if db.query(TeamStat).count() > 0:
        logger.info("Team stats already seeded, skipping.")
        return
    df = pd.read_csv(path)

    def si(val):
        try:
            if pd.isna(val):
                return None
            return int(val)
        except (ValueError, TypeError):
            return None

    for _, row in df.iterrows():
        stat = TeamStat(
            id=si(row["stat_id"]),
            fixture_id=si(row["fixture_id"]),
            team_id=si(row["team_id"]),
            is_home=bool(int(row["is_home"])),
            score=si(row.get("score")),
            goals=si(row.get("goals")),
            behinds=si(row.get("behinds")),
            shots_at_goal=si(row.get("shots_at_goal")),
            disposals=si(row.get("disposals")),
            kicks=si(row.get("kicks")),
            handballs=si(row.get("handballs")),
            marks=si(row.get("marks")),
            tackles=si(row.get("tackles")),
            hitouts=si(row.get("hitouts")),
            clearances=si(row.get("clearances")),
            inside_50s=si(row.get("inside_50s")),
            rebound_50s=si(row.get("rebound_50s")),
            contested_possessions=si(row.get("contested_possessions")),
            uncontested_possessions=si(row.get("uncontested_possessions")),
            metres_gained=si(row.get("metres_gained")),
            turnovers=si(row.get("turnovers")),
            free_kicks_for=si(row.get("free_kicks_for")),
            free_kicks_against=si(row.get("free_kicks_against")),
        )
        db.merge(stat)
    db.commit()
    logger.info(f"Seeded {len(df)} team stats.")


def seed_player_stats(db: Session, path: Path) -> None:
    if not path.exists():
        logger.warning(f"Player stats file not found: {path}")
        return
    if db.query(PlayerStat).count() > 0:
        logger.info("Player stats already seeded, skipping.")
        return
    df = pd.read_csv(path)

    def si(val):
        try:
            if pd.isna(val):
                return None
            return int(float(val))
        except (ValueError, TypeError):
            return None

    def sf(val):
        try:
            if pd.isna(val):
                return None
            return float(val)
        except (ValueError, TypeError):
            return None

    for _, row in df.iterrows():
        stat = PlayerStat(
            id=si(row["stat_id"]),
            fixture_id=si(row["fixture_id"]),
            player_id=si(row["player_id"]),
            team_id=si(row["team_id"]),
            disposals=si(row.get("disposals")),
            kicks=si(row.get("kicks")),
            handballs=si(row.get("handballs")),
            marks=si(row.get("marks")),
            tackles=si(row.get("tackles")),
            goals=si(row.get("goals")),
            behinds=si(row.get("behinds")),
            hitouts=si(row.get("hitouts")),
            clearances=si(row.get("clearances")),
            contested_possessions=si(row.get("contested_possessions")),
            uncontested_possessions=si(row.get("uncontested_possessions")),
            metres_gained=si(row.get("metres_gained")),
            time_on_ground_pct=sf(row.get("time_on_ground_pct")),
            fantasy_score=sf(row.get("fantasy_score")),
        )
        db.merge(stat)
    db.commit()
    logger.info(f"Seeded {len(df)} player stats.")


def seed_odds(db: Session, path: Path) -> None:
    if not path.exists():
        logger.warning(f"Odds file not found: {path}")
        return
    if db.query(Odds).count() > 0:
        logger.info("Odds already seeded, skipping.")
        return
    df = pd.read_csv(path)

    def si(val):
        try:
            if pd.isna(val):
                return None
            return int(float(val))
        except (ValueError, TypeError):
            return None

    for _, row in df.iterrows():
        odds = Odds(
            id=si(row["odds_id"]),
            fixture_id=si(row["fixture_id"]),
            market_type=str(row["market_type"]),
            selection=str(row["selection"]),
            bookmaker=str(row["bookmaker"]),
            decimal_odds=float(row["decimal_odds"]),
            american_odds=si(row.get("american_odds")),
            timestamp=str(row.get("timestamp", "")),
            status=str(row.get("status", "active")),
        )
        db.merge(odds)
    db.commit()
    logger.info(f"Seeded {len(df)} odds records.")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    init_db()
    db = SessionLocal()
    seed_from_csv(db, settings.demo_data_dir)
    db.close()
