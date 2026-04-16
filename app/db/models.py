"""SQLAlchemy ORM models for all database tables."""
from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey,
    Integer, JSON, String, Text, UniqueConstraint, Index
)
from sqlalchemy.orm import relationship
from app.db.database import Base


class Team(Base):
    __tablename__ = "teams"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, unique=True)
    abbreviation = Column(String(10), nullable=False)
    home_ground = Column(String(100))
    state = Column(String(10))
    founded = Column(Integer)
    primary_colour = Column(String(10))
    secondary_colour = Column(String(10))
    elo_rating = Column(Float, default=1500.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    home_fixtures = relationship("Fixture", foreign_keys="Fixture.home_team_id", back_populates="home_team")
    away_fixtures = relationship("Fixture", foreign_keys="Fixture.away_team_id", back_populates="away_team")
    players = relationship("Player", back_populates="team")


class Venue(Base):
    __tablename__ = "venues"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    city = Column(String(100))
    state = Column(String(10))
    capacity = Column(Integer)
    surface = Column(String(20))
    roof = Column(String(20))
    timezone = Column(String(50))


class Player(Base):
    __tablename__ = "players"

    id = Column(Integer, primary_key=True, index=True)
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    name = Column(String(100), nullable=False)
    position = Column(String(30))
    dob = Column(String(20))
    height_cm = Column(Integer)
    weight_kg = Column(Integer)
    debut_year = Column(Integer)
    status = Column(String(20), default="Active")
    created_at = Column(DateTime, default=datetime.utcnow)

    team = relationship("Team", back_populates="players")
    stats = relationship("PlayerStat", back_populates="player")

    __table_args__ = (Index("ix_players_team_id", "team_id"),)


class Fixture(Base):
    __tablename__ = "fixtures"

    id = Column(Integer, primary_key=True, index=True)
    season = Column(Integer, nullable=False)
    round = Column(Integer, nullable=False)
    home_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    away_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    venue_id = Column(Integer, ForeignKey("venues.id"))
    date = Column(String(20))
    time = Column(String(10))
    result_home_score = Column(Integer)
    result_away_score = Column(Integer)
    home_win = Column(Integer)  # 1/0/null
    margin = Column(Integer)
    total_score = Column(Integer)
    status = Column(String(20), default="upcoming")  # completed / upcoming
    created_at = Column(DateTime, default=datetime.utcnow)

    home_team = relationship("Team", foreign_keys=[home_team_id], back_populates="home_fixtures")
    away_team = relationship("Team", foreign_keys=[away_team_id], back_populates="away_fixtures")
    venue = relationship("Venue")
    player_stats = relationship("PlayerStat", back_populates="fixture")
    team_stats = relationship("TeamStat", back_populates="fixture")
    odds = relationship("Odds", back_populates="fixture")

    __table_args__ = (
        UniqueConstraint("season", "round", "home_team_id", "away_team_id", name="uq_fixture"),
        Index("ix_fixtures_season_round", "season", "round"),
    )


class PlayerStat(Base):
    __tablename__ = "player_stats"

    id = Column(Integer, primary_key=True, index=True)
    fixture_id = Column(Integer, ForeignKey("fixtures.id"), nullable=False)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    disposals = Column(Integer)
    kicks = Column(Integer)
    handballs = Column(Integer)
    marks = Column(Integer)
    tackles = Column(Integer)
    goals = Column(Integer)
    behinds = Column(Integer)
    hitouts = Column(Integer)
    clearances = Column(Integer)
    contested_possessions = Column(Integer)
    uncontested_possessions = Column(Integer)
    metres_gained = Column(Integer)
    time_on_ground_pct = Column(Float)
    fantasy_score = Column(Float)

    fixture = relationship("Fixture", back_populates="player_stats")
    player = relationship("Player", back_populates="stats")

    __table_args__ = (
        UniqueConstraint("fixture_id", "player_id", name="uq_player_stat"),
        Index("ix_player_stats_player_id", "player_id"),
    )


class TeamStat(Base):
    __tablename__ = "team_stats"

    id = Column(Integer, primary_key=True, index=True)
    fixture_id = Column(Integer, ForeignKey("fixtures.id"), nullable=False)
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    is_home = Column(Boolean, default=False)
    score = Column(Integer)
    goals = Column(Integer)
    behinds = Column(Integer)
    shots_at_goal = Column(Integer)
    disposals = Column(Integer)
    kicks = Column(Integer)
    handballs = Column(Integer)
    marks = Column(Integer)
    tackles = Column(Integer)
    hitouts = Column(Integer)
    clearances = Column(Integer)
    inside_50s = Column(Integer)
    rebound_50s = Column(Integer)
    contested_possessions = Column(Integer)
    uncontested_possessions = Column(Integer)
    metres_gained = Column(Integer)
    turnovers = Column(Integer)
    free_kicks_for = Column(Integer)
    free_kicks_against = Column(Integer)

    fixture = relationship("Fixture", back_populates="team_stats")

    __table_args__ = (
        UniqueConstraint("fixture_id", "team_id", name="uq_team_stat"),
        Index("ix_team_stats_team_id", "team_id"),
    )


class Odds(Base):
    __tablename__ = "odds"

    id = Column(Integer, primary_key=True, index=True)
    fixture_id = Column(Integer, ForeignKey("fixtures.id"), nullable=False)
    market_type = Column(String(50), nullable=False)  # head_to_head, line, total, player_disposals
    selection = Column(String(100), nullable=False)
    bookmaker = Column(String(50), nullable=False)
    decimal_odds = Column(Float, nullable=False)
    american_odds = Column(Integer)
    timestamp = Column(String(30))
    status = Column(String(20), default="active")  # active / closed

    fixture = relationship("Fixture", back_populates="odds")

    __table_args__ = (Index("ix_odds_fixture_market", "fixture_id", "market_type"),)


class GeneratedLeg(Base):
    __tablename__ = "generated_legs"

    id = Column(Integer, primary_key=True, index=True)
    fixture_id = Column(Integer, ForeignKey("fixtures.id"))
    player_id = Column(Integer, ForeignKey("players.id"), nullable=True)
    market_type = Column(String(50), nullable=False)
    selection = Column(String(100), nullable=False)
    bookmaker = Column(String(50))
    decimal_odds = Column(Float)
    model_probability = Column(Float)
    calibrated_probability = Column(Float)
    market_implied_probability = Column(Float)
    vig_adjusted_probability = Column(Float)
    edge = Column(Float)
    ev = Column(Float)
    confidence_score = Column(Float)
    explanation = Column(Text)
    run_id = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)


class GeneratedMulti(Base):
    __tablename__ = "generated_multis"

    id = Column(Integer, primary_key=True, index=True)
    leg_ids = Column(JSON)  # list of GeneratedLeg ids
    n_legs = Column(Integer)
    multi_type = Column(String(30))  # same_game / cross_game
    combined_odds = Column(Float)
    raw_probability = Column(Float)
    adjusted_probability = Column(Float)
    ev = Column(Float)
    correlation_score = Column(Float)
    correlation_label = Column(String(30))
    risk_score = Column(Float)
    explanation = Column(Text)
    run_id = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)


class BacktestResult(Base):
    __tablename__ = "backtest_results"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String(50), nullable=False)
    model_name = Column(String(100))
    market_type = Column(String(50))
    season = Column(Integer)
    n_predictions = Column(Integer)
    n_bets = Column(Integer)
    hit_rate = Column(Float)
    roi = Column(Float)
    brier_score = Column(Float)
    log_loss = Column(Float)
    avg_edge = Column(Float)
    avg_ev = Column(Float)
    calibration_data = Column(JSON)
    feature_importance = Column(JSON)
    parameters = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)


class TrainingMetadata(Base):
    __tablename__ = "training_metadata"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(String(50), nullable=False)
    model_name = Column(String(100))
    model_type = Column(String(50))
    market_type = Column(String(50))
    train_seasons = Column(JSON)
    n_train_samples = Column(Integer)
    n_val_samples = Column(Integer)
    brier_score_train = Column(Float)
    brier_score_val = Column(Float)
    log_loss_train = Column(Float)
    log_loss_val = Column(Float)
    feature_names = Column(JSON)
    hyperparameters = Column(JSON)
    artifact_path = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)


# ---------------------------------------------------------------------------
# Phase-2 tables: settlement, model versioning, research cycle
# ---------------------------------------------------------------------------

class LegSettlement(Base):
    """Tracks the outcome of every generated leg after game completion."""
    __tablename__ = "leg_settlements"

    id = Column(Integer, primary_key=True, index=True)
    # leg_id references GeneratedLeg.id (kept as string for flexibility)
    leg_id = Column(String(50), nullable=False, unique=True)
    fixture_id = Column(Integer, ForeignKey("fixtures.id"), nullable=True)
    # Denormalised columns (populated at settlement time for fast queries)
    market_type = Column(String(50))
    selection = Column(String(100))
    decimal_odds = Column(Float)
    model_probability = Column(Float)
    # Settlement result: 1=win, 0=loss, -1=void/push
    actual_outcome = Column(Integer, nullable=True)
    actual_home_score = Column(Integer, nullable=True)
    actual_away_score = Column(Integer, nullable=True)
    settlement_source = Column(String(30), default="auto")  # auto / manual
    notes = Column(Text, nullable=True)
    settled_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Convenience alias used by evaluation service
    @property
    def outcome(self):
        return self.actual_outcome

    __table_args__ = (Index("ix_leg_settlements_fixture", "fixture_id"),)


class MultiSettlement(Base):
    """Tracks the outcome of every generated multi after all legs settle."""
    __tablename__ = "multi_settlements"

    id = Column(Integer, primary_key=True, index=True)
    multi_id = Column(String(50), nullable=False, unique=True)
    leg_ids = Column(JSON)             # list of GeneratedLeg ids
    n_legs = Column(Integer)
    n_legs_won = Column(Integer, default=0)
    n_legs_total = Column(Integer, default=0)
    combined_odds = Column(Float)
    raw_probability = Column(Float)
    adjusted_probability = Column(Float)
    correlation_score = Column(Float)
    all_legs_win = Column(Boolean, default=False)
    # Settlement result: 1=won, 0=lost, -1=void (a leg was voided)
    actual_combined_result = Column(Integer, nullable=True)
    settlement_source = Column(String(30), default="auto")
    settled_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Convenience alias
    @property
    def outcome(self):
        return self.actual_combined_result

    __table_args__ = (Index("ix_multi_settlements_settled_at", "settled_at"),)


class ModelVersion(Base):
    """Tracks every trained model version with its validation metrics."""
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True)
    version_id = Column(String(20), nullable=False, unique=True)
    model_name = Column(String(100), nullable=False)   # e.g. "match_model"
    model_type = Column(String(50))                     # e.g. "ensemble"
    artifact_path = Column(String(255))
    brier_score_train = Column(Float)
    brier_score_val = Column(Float)
    log_loss_val = Column(Float)
    n_train_samples = Column(Integer)
    n_val_samples = Column(Integer)
    hyperparameters = Column(JSON)
    feature_names = Column(JSON)
    extra_metadata = Column(JSON)
    status = Column(String(20), default="challenger")  # challenger / champion / retired
    promoted_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_model_versions_name_status", "model_name", "status"),
    )


class ResearchCycleLog(Base):
    """Audit trail for each bootstrap research cycle execution."""
    __tablename__ = "research_cycle_log"

    id = Column(Integer, primary_key=True, index=True)
    cycle_id = Column(String(20), nullable=False, unique=True)
    started_at = Column(DateTime, nullable=False)
    finished_at = Column(DateTime)
    elapsed_seconds = Column(Float)
    n_fixtures_synced = Column(Integer, default=0)
    n_legs_settled = Column(Integer, default=0)
    brier_score = Column(Float, nullable=True)
    roi = Column(Float, nullable=True)
    retrained = Column(Boolean, default=False)
    promoted = Column(Boolean, default=False)
    summary = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index("ix_research_cycle_log_started", "started_at"),)
