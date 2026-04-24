"""Database connection and session management."""
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy.pool import StaticPool
from app.core.config import settings


class Base(DeclarativeBase):
    pass


def create_db_engine():
    """Create database engine with appropriate settings."""
    db_url = settings.database_url

    if "sqlite" in db_url:
        engine = create_engine(
            db_url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            echo=settings.app_debug,
        )
        # Enable WAL mode and foreign keys for SQLite
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()
    else:
        engine = create_engine(db_url, echo=settings.app_debug)

    return engine


engine = create_db_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """FastAPI dependency: yield database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables."""
    from app.db import models  # noqa: F401 - ensure models are registered
    Base.metadata.create_all(bind=engine)
    run_migrations()


def run_migrations():
    """
    Apply additive ALTER TABLE migrations for columns added after initial schema.
    Uses try/except per column so partial failures don't block startup.
    """
    _new_generated_leg_cols = [
        ("ml_probability", "REAL"),
        ("signal_consensus_probability", "REAL"),
        ("signal_agreement", "REAL"),
        ("prediction_variance", "REAL"),
        ("data_completeness", "REAL"),
        ("trust_score", "REAL"),
        ("n_games", "INTEGER"),
    ]
    _new_leg_settlement_cols = [
        ("vig_adjusted_probability", "REAL"),
        ("edge_at_prediction", "REAL"),
        ("ml_probability", "REAL"),
        ("signal_consensus_probability", "REAL"),
        ("signal_agreement", "REAL"),
        ("prediction_variance", "REAL"),
        ("data_completeness", "REAL"),
        ("trust_score", "REAL"),
        ("n_games", "INTEGER"),
    ]

    with engine.connect() as conn:
        for col, typ in _new_generated_leg_cols:
            try:
                conn.execute(
                    __import__("sqlalchemy").text(
                        f"ALTER TABLE generated_legs ADD COLUMN {col} {typ}"
                    )
                )
                conn.commit()
            except Exception:
                pass  # column already exists

        for col, typ in _new_leg_settlement_cols:
            try:
                conn.execute(
                    __import__("sqlalchemy").text(
                        f"ALTER TABLE leg_settlements ADD COLUMN {col} {typ}"
                    )
                )
                conn.commit()
            except Exception:
                pass  # column already exists

        _new_player_stat_cols = [
            ("effective_kicks", "INTEGER"),
            ("effective_handballs", "INTEGER"),
            ("effective_disposals", "INTEGER"),
            ("clangers", "INTEGER"),
            ("contested_marks", "INTEGER"),
            ("marks_inside_50", "INTEGER"),
            ("inside_50s", "INTEGER"),
            ("rebound_50s", "INTEGER"),
            ("centre_clearances", "INTEGER"),
            ("stoppage_clearances", "INTEGER"),
            ("hitouts_to_advantage", "INTEGER"),
            ("score_involvements", "INTEGER"),
            ("goal_assists", "INTEGER"),
            ("one_percenters", "INTEGER"),
            ("bounces", "INTEGER"),
            ("frees_for", "INTEGER"),
            ("frees_against", "INTEGER"),
            ("supercoach_score", "REAL"),
            ("rating_points", "REAL"),
        ]
        _new_team_stat_cols = [
            ("effective_kicks", "INTEGER"),
            ("effective_handballs", "INTEGER"),
            ("effective_disposals", "INTEGER"),
            ("clangers", "INTEGER"),
            ("contested_marks", "INTEGER"),
            ("marks_inside_50", "INTEGER"),
            ("centre_clearances", "INTEGER"),
            ("stoppage_clearances", "INTEGER"),
            ("hitouts_to_advantage", "INTEGER"),
            ("score_involvements", "INTEGER"),
            ("goal_assists", "INTEGER"),
            ("one_percenters", "INTEGER"),
            ("bounces", "INTEGER"),
        ]

        for col, typ in _new_player_stat_cols:
            try:
                conn.execute(
                    __import__("sqlalchemy").text(
                        f"ALTER TABLE player_stats ADD COLUMN {col} {typ}"
                    )
                )
                conn.commit()
            except Exception:
                pass

        for col, typ in _new_team_stat_cols:
            try:
                conn.execute(
                    __import__("sqlalchemy").text(
                        f"ALTER TABLE team_stats ADD COLUMN {col} {typ}"
                    )
                )
                conn.commit()
            except Exception:
                pass
