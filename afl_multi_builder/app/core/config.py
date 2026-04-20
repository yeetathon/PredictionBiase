"""
Extended application configuration.
All secrets read from environment variables / .env file only.

Live-data only — no demo mode, no fallback to CSV files.
Data source: AFL Data Sports Group API + The Odds API.
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
    artifacts_dir: Path = Path("./data/models")
    raw_cache_dir: Path = Path("./data/cache")

    # ── AFL Data Sports Group API ──────────────────────────────────────────
    afl_data_username: str = ""
    afl_data_password: str = ""
    afl_data_authkey: str = ""
    afl_data_base_url: str = "https://api.afl.com.au"
    afl_data_competition_id: int = 1  # 1 = AFL Men's

    # ── The Odds API ───────────────────────────────────────────────────────
    odds_api_key: str = ""
    odds_api_base_url: str = "https://api.the-odds-api.com/v4"
    odds_api_sport: str = "aussierules_afl"
    odds_api_bookmakers: str = "tab,sportsbet,bet365,unibet,pointsbet,betfair,williamhill,neds"

    # ── Data Mode (live | cache only — no demo) ───────────────────────────
    # live  = fetch from AFL Data Sports Group API (uses cache, respects TTL)
    # cache = only use cached responses (no new API calls)
    data_mode: Literal["live", "cache"] = "live"

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
    # Precision-first thresholds — conservative defaults.
    # The system should produce FEWER, BETTER bets rather than many weak ones.

    # Minimum real edge over bookmaker vig-adjusted probability
    # 5% = meaningful edge; below this is noise given model uncertainty
    min_edge_threshold: float = 0.05

    # Maximum allowed correlation between legs in a multi.
    # 0.45 rejects most same-game H2H+Line combos (0.85 correlation)
    max_correlation_score: float = 0.45

    # Minimum expected value per bet (5% = meaningful positive EV)
    min_ev_threshold: float = 0.04

    # Maximum legs per game in one multi (2 = enough same-game exposure)
    max_legs_per_game: int = 2

    # Maximum legs per player in one multi
    max_legs_per_player: int = 1

    # Maximum total legs in a multi (shorter = more reliable)
    max_multi_legs: int = 3

    # Minimum legs in a multi
    min_multi_legs: int = 2

    # Minimum calibrated win probability for a leg to enter the pool.
    # A leg at 51% is barely better than a coin flip — requires real conviction.
    min_calibrated_probability: float = 0.54

    # Minimum confidence score (0-100) for a leg to enter the pool.
    min_confidence_score: float = 40.0

    # Maximum decimal odds for any leg.
    # High odds = high uncertainty; reject longshots regardless of edge.
    max_leg_odds: float = 4.00

    # Minimum legs in the quality pool before any multi is built.
    # If fewer than this pass all gates, return no-bet instead of forcing output.
    min_pool_size_for_multi: int = 3

    # Require real bookmaker odds for a leg to qualify.
    # If True, legs synthesised from model probability alone are rejected.
    odds_required: bool = True

    # Minimum signal agreement (0–1) across independent prediction signals.
    # Legs where signals strongly disagree are rejected regardless of EV.
    # 0.30 rejects when signal std ≥ 0.105 (≈10.5 pp spread between signals).
    # Set to 0.0 to disable (not recommended).
    min_signal_agreement: float = 0.30

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
    def is_afl_data_configured(self) -> bool:
        return bool(self.afl_data_authkey and self.afl_data_authkey.strip())

    @property
    def is_odds_api_configured(self) -> bool:
        return bool(self.odds_api_key and self.odds_api_key.strip())

    @property
    def odds_api_bookmakers_list(self) -> List[str]:
        return [b.strip() for b in self.odds_api_bookmakers.split(",") if b.strip()]

    @property
    def effective_data_mode(self) -> str:
        """Return the effective data mode. Only 'live' or 'cache' — never demo."""
        return self.data_mode

    @property
    def any_api_configured(self) -> bool:
        return self.is_afl_data_configured


settings = Settings()
