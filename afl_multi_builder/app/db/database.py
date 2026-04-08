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
