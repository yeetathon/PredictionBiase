"""
Extended application configuration.
All secrets read from environment variables / .env file only.
"""
from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    # ── Application ────────────────────────────────────────────────────────
    app_env: str = "development"
    app_debug: bool = True
    log_level: str = "INFO"

    # ── Database ───────────────────────────────────────────────────────────
    database_url: str = "sqlite:///./data/afl_multi_builder.db"

    # ── Data paths ─────────────────────────────────────────────────────────
    demo_data_dir: Path = Path("./data/demo")
    artifacts_dir: Path = Path("./data/models")
    raw_cache_dir: Path = Path("./data/cache")

    # ── Sportradar API ─────────────────────────────────────────────────────
    # Read from environment - NEVER hardcode
    sportradar_api_key: str = ""
    sportradar_base_url: str = "https://api.sportradar.com/afl/trial/v2/en"
    # AFL 2025 season ID (verify against your plan's competition/season IDs)
    sportradar_afl_competition_id: str = "sr:competition:3394"
    sportradar_afl_season_id: str = ""  # populated at runtime from API or set manually

    # ── Data Mode ─────────────────────────────────────────────────────────
    # live: prefer API (uses cache, fetches only when stale)
    # cache: only use cached API responses, never make live calls
    # demo: NOT supported in production — raises RuntimeError on use
    data_mode: Literal["live", "cache", "demo"] = "live"
    enable_demo_fallback: bool = False  # demo fallback disabled

    # ── Rate Limiting & Quota ─────────────────────────────────────────────
    api_rate_limit_qps: float = 1.0       # max requests per second
    api_quota_total: int = 1000            # lifetime quota
    api_quota_warn_pct: float = 0.80       # warn when this fraction used
    api_quota_refuse_pct: float = 0.95     # refuse requests when this fraction used

    # ── Cache TTL ─────────────────────────────────────────────────────────
    cache_ttl_hours: int = 6               # default TTL for API responses
    cache_ttl_upcoming_hours: int = 2      # shorter TTL for upcoming fixtures
    cache_ttl_results_hours: int = 24      # longer TTL for completed results

    # ── Sync Windows ──────────────────────────────────────────────────────
    upcoming_lookahead_days: int = 14
    recent_settlement_lookback_days: int = 7

    # ── Model Settings ────────────────────────────────────────────────────
    min_edge_threshold: float = 0.03
    max_correlation_score: float = 0.7
    min_ev_threshold: float = 0.02
    max_legs_per_game: int = 3
    max_legs_per_player: int = 2
    max_multi_legs: int = 4
    min_multi_legs: int = 2

    # ── Training / Retraining ─────────────────────────────────────────────
    retrain_min_new_games: int = 10        # min new settled games to trigger retraining
    retrain_brier_degradation_threshold: float = 0.02  # retrain if Brier worsens by this
    model_promotion_brier_improvement: float = 0.002   # min improvement to promote new model

    # ── Bootstrap ─────────────────────────────────────────────────────────
    enable_bootstrap_mode: bool = True
    bootstrap_report_days: int = 30

    # ── Backtesting ───────────────────────────────────────────────────────
    backtest_start_season: int = 2021
    backtest_end_season: int = 2023
    walk_forward_window_games: int = 50

    # ── Elo ───────────────────────────────────────────────────────────────
    elo_k_factor: float = 32.0
    elo_initial_rating: float = 1500.0
    elo_home_advantage: float = 70.0

    @property
    def model_artifacts_dir(self) -> Path:
        return self.artifacts_dir

    @property
    def is_sportradar_configured(self) -> bool:
        return bool(self.sportradar_api_key and self.sportradar_api_key.strip())

    @property
    def effective_data_mode(self) -> str:
        """
        Return the effective data mode.

        Never silently returns 'demo' — if the API key is missing the caller
        will get 'live' or 'cache' back and SportradarLoader will raise a
        clear RuntimeError on construction.
        """
        return self.data_mode


settings = Settings()
