"""Application configuration using pydantic-settings."""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    # App
    app_env: str = "development"
    app_debug: bool = True
    log_level: str = "INFO"

    # Database
    database_url: str = "sqlite:///./data/afl_multi_builder.db"

    # Data paths
    demo_data_dir: Path = Path("./data/demo")
    artifacts_dir: Path = Path("./data/models")

    # Model settings
    min_edge_threshold: float = 0.03
    max_correlation_score: float = 0.7
    min_ev_threshold: float = 0.02
    max_legs_per_game: int = 3
    max_legs_per_player: int = 2
    max_multi_legs: int = 4
    min_multi_legs: int = 2

    # Backtesting
    backtest_start_season: int = 2021
    backtest_end_season: int = 2023
    walk_forward_window_games: int = 50

    # Elo settings
    elo_k_factor: float = 32.0
    elo_initial_rating: float = 1500.0
    elo_home_advantage: float = 70.0

    @property
    def model_artifacts_dir(self) -> Path:
        return self.artifacts_dir


settings = Settings()
