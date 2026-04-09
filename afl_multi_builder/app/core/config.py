"""
Extended application configuration.
All secrets read from environment variables / .env file only.
"""
from pathlib import Path
from typing import List, Literal
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
    sportradar_api_key: str = ""
    sportradar_base_url: str = "https://api.sportradar.com/afl/trial/v3"
    sportradar_afl_competition_id: str = "sr:competition:3394"
    sportradar_afl_season_id: str = ""

    # ── API-Sports AFL ─────────────────────────────────────────────────────
    api_sports_key: str = ""
    api_sports_base_url: str = "https://v1.afl.api-sports.io"
    api_sports_afl_league_id: int = 1

    # ── The Odds API ───────────────────────────────────────────────────────
    odds_api_key: str = ""
    odds_api_base_url: str = "https://api.the-odds-api.com/v4"
    odds_api_sport: str = "aussierules_afl"
    odds_api_bookmakers: str = "tab,sportsbet,bet365,unibet,pointsbet,betfair,williamhill,neds"

    # ── Data Mode ─────────────────────────────────────────────────────────
    data_mode: Literal["live", "cache", "demo"] = "live"
    enable_demo_fallback: bool = False

    # ── Rate Limiting & Quota ─────────────────────────────────────────────
    api_rate_limit_qps: float = 1.0
    api_quota_total: int = 1000
    api_quota_warn_pct: float = 0.80
    api_quota_refuse_pct: float = 0.95

    # ── Cache TTL ─────────────────────────────────────────────────────────
    cache_ttl_hours: int = 6
    cache_ttl_upcoming_hours: int = 2
    cache_ttl_results_hours: int = 24

    # ── Scraping / Edge Intelligence ───────────────────────────────────────
    enable_scraping: bool = True
    scrape_cache_ttl_minutes: int = 30
    scrape_max_age_hours: int = 6

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
    retrain_min_new_games: int = 10
    retrain_brier_degradation_threshold: float = 0.02
    model_promotion_brier_improvement: float = 0.002

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

    # ── Computed helpers ──────────────────────────────────────────────────

    @property
    def model_artifacts_dir(self) -> Path:
        return self.artifacts_dir

    @property
    def is_sportradar_configured(self) -> bool:
        return bool(self.sportradar_api_key and self.sportradar_api_key.strip())

    @property
    def is_api_sports_configured(self) -> bool:
        return bool(self.api_sports_key and self.api_sports_key.strip())

    @property
    def is_odds_api_configured(self) -> bool:
        return bool(self.odds_api_key and self.odds_api_key.strip())

    @property
    def odds_api_bookmakers_list(self) -> List[str]:
        return [b.strip() for b in self.odds_api_bookmakers.split(",") if b.strip()]

    @property
    def effective_data_mode(self) -> str:
        """Return the effective data mode. Never silently falls back to demo."""
        return self.data_mode

    @property
    def any_api_configured(self) -> bool:
        return self.is_sportradar_configured or self.is_api_sports_configured


settings = Settings()
