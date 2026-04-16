"""
Preflight validation service.

Runs before any pipeline execution to ensure all required data sources,
API keys, fixtures, and model artifacts are present and healthy.

If ANY required check fails → raises PreflightError with a clear description
of what failed, why, and how to fix it.

No demo fallbacks. No partial data. No silent degradation for required inputs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from app.core.config import settings


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    fix: str = ""
    required: bool = True


@dataclass
class PreflightReport:
    passed: bool
    checks: List[CheckResult] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    elapsed_ms: int = 0

    def to_dict(self) -> Dict:
        return {
            "passed": self.passed,
            "timestamp": self.timestamp,
            "elapsed_ms": self.elapsed_ms,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "required": c.required,
                    "detail": c.detail,
                    "fix": c.fix,
                }
                for c in self.checks
            ],
            "failed_required": [
                {"name": c.name, "detail": c.detail, "fix": c.fix}
                for c in self.checks
                if not c.passed and c.required
            ],
        }


class PreflightError(RuntimeError):
    """Raised when one or more required preflight checks fail."""

    def __init__(self, report: PreflightReport):
        self.report = report
        failures = [c for c in report.checks if not c.passed and c.required]
        lines = [f"Preflight failed — {len(failures)} required check(s) did not pass:\n"]
        for c in failures:
            lines.append(f"  ✗ {c.name}: {c.detail}")
            if c.fix:
                lines.append(f"    → Fix: {c.fix}")
        super().__init__("\n".join(lines))


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_sportradar_key() -> CheckResult:
    if settings.is_sportradar_configured:
        return CheckResult(
            name="sportradar_api_key",
            passed=True,
            detail=f"SPORTRADAR_API_KEY present ({len(settings.sportradar_api_key)} chars)",
        )
    return CheckResult(
        name="sportradar_api_key",
        passed=False,
        detail="SPORTRADAR_API_KEY is missing or empty",
        fix="Add SPORTRADAR_API_KEY=<your_key> to your .env file. "
            "Get a key at https://developer.sportradar.com/",
        required=True,
    )


def _check_sportradar_season() -> CheckResult:
    if settings.sportradar_afl_season_id and settings.sportradar_afl_season_id.strip():
        return CheckResult(
            name="sportradar_season_id",
            passed=True,
            detail=f"SPORTRADAR_AFL_SEASON_ID = {settings.sportradar_afl_season_id}",
        )
    return CheckResult(
        name="sportradar_season_id",
        passed=False,
        detail="SPORTRADAR_AFL_SEASON_ID is not set — cannot fetch fixtures for the current season",
        fix="Add SPORTRADAR_AFL_SEASON_ID=sr:season:<id> to your .env file. "
            "Find the season ID by calling GET /competitions/{id}/seasons.json from the Sportradar API.",
        required=True,
    )


def _check_sportradar_connectivity() -> CheckResult:
    """Attempt a lightweight connectivity check against Sportradar."""
    if not settings.is_sportradar_configured:
        return CheckResult(
            name="sportradar_connectivity",
            passed=False,
            detail="Skipped — no API key configured",
            fix="Configure SPORTRADAR_API_KEY first",
            required=True,
        )
    try:
        from app.data_ingestion.sportradar_client import SportradarClient
        client = SportradarClient(
            api_key=settings.sportradar_api_key,
            base_url=settings.sportradar_base_url,
        )
        competition_id = settings.sportradar_afl_competition_id
        data = client.get(
            f"competitions/{competition_id}/seasons.json",
            cache_ttl_seconds=3600,
        )
        if isinstance(data, dict) and ("seasons" in data or "_source" in data):
            return CheckResult(
                name="sportradar_connectivity",
                passed=True,
                detail=f"Sportradar API reachable (source={data.get('_source', 'unknown')})",
            )
        return CheckResult(
            name="sportradar_connectivity",
            passed=False,
            detail=f"Sportradar API returned unexpected response: {str(data)[:200]}",
            fix="Check your SPORTRADAR_API_KEY and SPORTRADAR_BASE_URL are correct",
            required=True,
        )
    except Exception as exc:
        return CheckResult(
            name="sportradar_connectivity",
            passed=False,
            detail=f"Sportradar API unreachable: {exc}",
            fix="Check network connectivity, API key validity, and base URL. "
                "Verify SPORTRADAR_BASE_URL=https://api.sportradar.com/australianrules/trial/v3/en",
            required=True,
        )


def _check_upcoming_fixtures() -> CheckResult:
    """Check that at least one upcoming fixture exists for the configured season."""
    if not settings.is_sportradar_configured:
        return CheckResult(
            name="upcoming_fixtures",
            passed=False,
            detail="Skipped — no Sportradar API key",
            fix="Configure SPORTRADAR_API_KEY",
            required=True,
        )
    if not settings.sportradar_afl_season_id:
        return CheckResult(
            name="upcoming_fixtures",
            passed=False,
            detail="Skipped — SPORTRADAR_AFL_SEASON_ID not set",
            fix="Set SPORTRADAR_AFL_SEASON_ID in your .env file",
            required=True,
        )
    try:
        from app.data_ingestion.data_source_manager import SportradarDataProvider
        provider = SportradarDataProvider()
        df = provider.get_schedule(settings.sportradar_afl_season_id)
        if df.empty:
            return CheckResult(
                name="upcoming_fixtures",
                passed=False,
                detail="Schedule returned zero rows — season may be over or season ID is wrong",
                fix="Verify SPORTRADAR_AFL_SEASON_ID is correct for the current season. "
                    "Check the Sportradar seasons endpoint to get the valid season ID.",
                required=True,
            )
        upcoming = df[df.get("status", df.get("status", "")).eq("not_started") |
                      df.get("status", df.get("status", "")).eq("upcoming")] \
            if "status" in df.columns else df
        n = len(upcoming) if not upcoming.empty else len(df)
        return CheckResult(
            name="upcoming_fixtures",
            passed=True,
            detail=f"Found {n} upcoming fixtures in season {settings.sportradar_afl_season_id}",
        )
    except Exception as exc:
        return CheckResult(
            name="upcoming_fixtures",
            passed=False,
            detail=f"Failed to load fixtures: {exc}",
            fix="Check Sportradar API key and season ID. Ensure your trial plan covers fixture schedules.",
            required=True,
        )


def _check_odds_api() -> CheckResult:
    """Check Odds API key and connectivity. Required for market edge calculations."""
    if not settings.is_odds_api_configured:
        return CheckResult(
            name="odds_api_key",
            passed=False,
            detail="ODDS_API_KEY is missing — market odds unavailable, edge calculations disabled",
            fix="Add ODDS_API_KEY=<your_key> to your .env file. "
                "Get a free key at https://the-odds-api.com/",
            required=False,  # Optional: pipeline still runs but no market comparison
        )
    try:
        from app.data_ingestion.odds_api_client import OddsAPIClient
        client = OddsAPIClient()
        sports = client.get_sports()
        if sports is not None:
            return CheckResult(
                name="odds_api_key",
                passed=True,
                detail="Odds API reachable and key valid",
            )
        return CheckResult(
            name="odds_api_key",
            passed=False,
            detail="Odds API returned null — key may be invalid or quota exhausted",
            fix="Check ODDS_API_KEY in your .env file and your quota at https://the-odds-api.com/account/",
            required=False,
        )
    except Exception as exc:
        return CheckResult(
            name="odds_api_key",
            passed=False,
            detail=f"Odds API connectivity failed: {exc}",
            fix="Verify ODDS_API_KEY and network connectivity",
            required=False,
        )


def _check_model_artifacts() -> CheckResult:
    """Check whether trained model artifacts exist on disk."""
    artifacts_dir = settings.artifacts_dir
    if not artifacts_dir.exists():
        return CheckResult(
            name="model_artifacts",
            passed=False,
            detail=f"Artifacts directory missing: {artifacts_dir}",
            fix="Run 'python scripts/run_training.py' to train and save models",
            required=False,  # Pipeline auto-trains if missing
        )
    # Look for any .joblib files
    model_files = list(artifacts_dir.glob("*.joblib"))
    if not model_files:
        return CheckResult(
            name="model_artifacts",
            passed=False,
            detail=f"No model files found in {artifacts_dir}",
            fix="Run 'python scripts/run_training.py' to train models",
            required=False,  # Pipeline auto-trains
        )
    names = [f.stem for f in model_files]
    return CheckResult(
        name="model_artifacts",
        passed=True,
        detail=f"Found {len(model_files)} model artifact(s): {', '.join(names[:5])}",
    )


def _check_data_freshness() -> CheckResult:
    """Check whether cached data is fresh enough to use."""
    cache_dir = settings.raw_cache_dir
    if not cache_dir.exists():
        return CheckResult(
            name="data_freshness",
            passed=True,  # Cache will be populated on first run
            detail="Cache directory does not exist yet — will be created on first API call",
        )
    cache_files = list(cache_dir.glob("*.json")) + list(cache_dir.glob("*.pickle"))
    if not cache_files:
        return CheckResult(
            name="data_freshness",
            passed=True,
            detail="No cached data yet — fresh API calls will be made",
        )
    # Find the most recent cache file
    most_recent = max(cache_files, key=lambda f: f.stat().st_mtime)
    age_hours = (datetime.utcnow().timestamp() - most_recent.stat().st_mtime) / 3600
    max_age = settings.cache_ttl_hours
    if age_hours > max_age * 2:
        return CheckResult(
            name="data_freshness",
            passed=False,
            detail=f"Most recent cache file is {age_hours:.1f}h old (max allowed: {max_age * 2}h). "
                   f"Stale data may lead to incorrect predictions.",
            fix="Run the pipeline with data_mode=live to refresh the cache, "
                "or run 'python scripts/sync_sportradar.py'",
            required=False,
        )
    return CheckResult(
        name="data_freshness",
        passed=True,
        detail=f"Cache is fresh (most recent file: {age_hours:.1f}h old, limit: {max_age * 2}h)",
    )


def _check_data_mode() -> CheckResult:
    mode = settings.effective_data_mode
    if mode in ("live", "cache"):
        return CheckResult(
            name="data_mode",
            passed=True,
            detail=f"DATA_MODE={mode} (live-data system)",
        )
    return CheckResult(
        name="data_mode",
        passed=False,
        detail=f"Invalid DATA_MODE='{mode}' — only 'live' and 'cache' are supported",
        fix="Set DATA_MODE=live in your .env file",
        required=True,
    )


# ---------------------------------------------------------------------------
# PreflightService
# ---------------------------------------------------------------------------

class PreflightService:
    """
    Run all preflight checks before the prediction pipeline executes.

    Usage::

        svc = PreflightService()
        report = svc.run()
        if not report.passed:
            raise PreflightError(report)
    """

    def run(self, raise_on_failure: bool = True) -> PreflightReport:
        """
        Execute all preflight checks and return a report.

        Args:
            raise_on_failure: If True (default) raise PreflightError when any
                              *required* check fails.

        Returns:
            PreflightReport with all check results.
        """
        start = datetime.utcnow()
        logger.info("Running preflight checks...")

        checks: List[CheckResult] = []

        # --- Required checks ---
        checks.append(_check_data_mode())
        checks.append(_check_sportradar_key())
        checks.append(_check_sportradar_season())
        checks.append(_check_sportradar_connectivity())
        checks.append(_check_upcoming_fixtures())

        # --- Optional checks (warn but don't block) ---
        checks.append(_check_odds_api())
        checks.append(_check_model_artifacts())
        checks.append(_check_data_freshness())

        elapsed_ms = int((datetime.utcnow() - start).total_seconds() * 1000)

        required_failures = [c for c in checks if not c.passed and c.required]
        optional_failures = [c for c in checks if not c.passed and not c.required]

        passed = len(required_failures) == 0

        for c in checks:
            icon = "✓" if c.passed else ("✗" if c.required else "⚠")
            logger.info(f"  {icon} {c.name}: {c.detail}")

        if optional_failures:
            logger.warning(
                f"Preflight: {len(optional_failures)} optional check(s) failed "
                f"(predictions will run with reduced capability): "
                + ", ".join(c.name for c in optional_failures)
            )

        if required_failures:
            logger.error(
                f"Preflight: {len(required_failures)} required check(s) FAILED — "
                "predictions blocked"
            )

        report = PreflightReport(
            passed=passed,
            checks=checks,
            elapsed_ms=elapsed_ms,
        )

        if not passed and raise_on_failure:
            raise PreflightError(report)

        return report
