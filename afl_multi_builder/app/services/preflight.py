"""
Preflight validation service.

Runs before any pipeline execution to ensure all required data sources,
API keys, fixtures, and model artifacts are present and healthy.

If ANY required check fails → raises PreflightError with a clear description
of what failed, why, and how to fix it.

No demo fallbacks. No partial data. No silent degradation for required inputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List

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

def _check_afl_data_key() -> CheckResult:
    if settings.is_afl_data_configured:
        key_preview = settings.afl_data_authkey[:6] + "..." if len(settings.afl_data_authkey) > 6 else "***"
        return CheckResult(
            name="afl_data_authkey",
            passed=True,
            detail=f"AFL_DATA_AUTHKEY present ({key_preview}, {len(settings.afl_data_authkey)} chars)",
        )
    return CheckResult(
        name="afl_data_authkey",
        passed=False,
        detail="AFL_DATA_AUTHKEY is missing or empty",
        fix="Add AFL_DATA_AUTHKEY=<your_key> to your .env file. "
            "Contact AFL Data Sports Group to obtain credentials.",
        required=True,
    )


def _check_afl_data_connectivity() -> CheckResult:
    """Attempt a lightweight connectivity check against AFL Data Sports Group API."""
    if not settings.is_afl_data_configured:
        return CheckResult(
            name="afl_data_connectivity",
            passed=False,
            detail="Skipped — no AFL_DATA_AUTHKEY configured",
            fix="Configure AFL_DATA_AUTHKEY first",
            required=True,
        )
    try:
        from app.data_ingestion.afl_data_client import AFLDataClient
        client = AFLDataClient()
        year = client.current_year()
        data = client.get_team_list(year=year, ttl=3600)
        if isinstance(data, dict):
            # Any valid response (even empty team list) means API is reachable
            source = data.get("_source", "unknown")
            teams = data.get("teams") or data.get("teamList") or data.get("data", {}).get("teams", [])
            n_teams = len(teams) if isinstance(teams, list) else "?"
            return CheckResult(
                name="afl_data_connectivity",
                passed=True,
                detail=f"AFL Data API reachable (source={source}, {n_teams} teams returned)",
            )
        return CheckResult(
            name="afl_data_connectivity",
            passed=False,
            detail=f"AFL Data API returned unexpected response: {str(data)[:200]}",
            fix="Check AFL_DATA_AUTHKEY and AFL_DATA_BASE_URL are correct",
            required=True,
        )
    except Exception as exc:
        return CheckResult(
            name="afl_data_connectivity",
            passed=False,
            detail=f"AFL Data API unreachable: {exc}",
            fix="Check network connectivity, AFL_DATA_AUTHKEY validity, "
                "and AFL_DATA_BASE_URL. Default: https://api.afl.com.au",
            required=True,
        )


def _check_upcoming_fixtures() -> CheckResult:
    """Check that at least one upcoming fixture is available."""
    if not settings.is_afl_data_configured:
        return CheckResult(
            name="upcoming_fixtures",
            passed=False,
            detail="Skipped — no AFL_DATA_AUTHKEY configured",
            fix="Configure AFL_DATA_AUTHKEY",
            required=True,
        )
    try:
        from app.data_ingestion.afl_data_loader import AFLDataLoader
        loader = AFLDataLoader()
        df = loader.load_upcoming_fixtures_df()
        if df.empty:
            all_df = loader.fixtures_df
            return CheckResult(
                name="upcoming_fixtures",
                passed=False,
                detail=(
                    f"No upcoming fixtures found (total fixtures in schedule: {len(all_df)}). "
                    "The season may be between rounds or the competition ID may be wrong."
                ),
                fix=f"Verify AFL_DATA_COMPETITION_ID={settings.afl_data_competition_id} "
                    "is correct for the current AFL season.",
                required=True,
            )
        return CheckResult(
            name="upcoming_fixtures",
            passed=True,
            detail=f"Found {len(df)} upcoming fixtures",
        )
    except Exception as exc:
        return CheckResult(
            name="upcoming_fixtures",
            passed=False,
            detail=f"Failed to load upcoming fixtures: {exc}",
            fix="Check AFL_DATA_AUTHKEY and AFL_DATA_COMPETITION_ID.",
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
            required=False,
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
            fix="Check ODDS_API_KEY in your .env file and quota at https://the-odds-api.com/account/",
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
            required=False,
        )
    model_files = list(artifacts_dir.glob("*.joblib"))
    if not model_files:
        return CheckResult(
            name="model_artifacts",
            passed=False,
            detail=f"No model files found in {artifacts_dir}",
            fix="Run 'python scripts/run_training.py' to train models",
            required=False,
        )
    names = [f.stem for f in model_files]
    return CheckResult(
        name="model_artifacts",
        passed=True,
        detail=f"Found {len(model_files)} model artifact(s): {', '.join(names[:5])}",
    )


def _check_data_freshness() -> CheckResult:
    """Check whether cached data is fresh enough to use."""
    import time
    cache_dir = settings.raw_cache_dir / "afl_data"
    if not cache_dir.exists():
        return CheckResult(
            name="data_freshness",
            passed=True,
            detail="Cache directory does not exist yet — will be created on first API call",
        )
    cache_files = list(cache_dir.glob("*.json"))
    if not cache_files:
        return CheckResult(
            name="data_freshness",
            passed=True,
            detail="No cached data yet — fresh API calls will be made",
        )
    most_recent = max(cache_files, key=lambda f: f.stat().st_mtime)
    age_hours = (time.time() - most_recent.stat().st_mtime) / 3600
    max_age = settings.cache_ttl_hours
    if age_hours > max_age * 2:
        return CheckResult(
            name="data_freshness",
            passed=False,
            detail=(
                f"Most recent cache file is {age_hours:.1f}h old (limit: {max_age * 2}h). "
                "Stale data may lead to incorrect predictions."
            ),
            fix="Run the pipeline with data_mode=live to refresh the cache.",
            required=False,
        )
    return CheckResult(
        name="data_freshness",
        passed=True,
        detail=f"Cache is fresh (most recent: {age_hours:.1f}h old, limit: {max_age * 2}h)",
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
        start = datetime.utcnow()
        logger.info("Running preflight checks...")

        checks: List[CheckResult] = []

        # --- Required checks ---
        checks.append(_check_data_mode())
        checks.append(_check_afl_data_key())
        checks.append(_check_afl_data_connectivity())
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
