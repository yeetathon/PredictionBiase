"""
Selection-driven multi generation service.

Manages async generation jobs with staged progress reporting.
Each job runs in a background thread; the frontend polls /generate/{job_id}/status
for live progress and final results.

New workflow:
  1. Load only the selected fixtures
  2. Fetch bookmaker odds (real API or cache)
  3. Fetch weather conditions
  4. Gather news / injury intelligence
  5. Build feature set
  6. Score and filter candidate legs
  7. Construct multis (style-constrained)
  8. Rank and return exactly n_multis
"""
from __future__ import annotations

import hashlib
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from app.core.config import settings

# ── AFL team → home venue mapping ────────────────────────────────────────────

_TEAM_VENUE = {
    "Adelaide Crows": "Adelaide Oval",
    "Brisbane Lions": "The Gabba",
    "Carlton": "Marvel Stadium",
    "Collingwood": "MCG",
    "Essendon": "Marvel Stadium",
    "Fremantle": "Optus Stadium",
    "Geelong Cats": "GMHBA Stadium",
    "GWS Giants": "ENGIE Stadium",
    "Gold Coast Suns": "Heritage Bank Stadium",
    "Hawthorn": "MCG",
    "Melbourne": "MCG",
    "North Melbourne": "Marvel Stadium",
    "Port Adelaide": "Adelaide Oval",
    "Richmond": "MCG",
    "St Kilda": "Marvel Stadium",
    "Sydney Swans": "SCG",
    "West Coast Eagles": "Optus Stadium",
    "Western Bulldogs": "Marvel Stadium",
}

# ── Multi style profiles ──────────────────────────────────────────────────────

STYLE_PROFILES: Dict[str, Dict[str, Any]] = {
    "safest": {
        "label": "Safest",
        "icon": "🛡",
        "description": "High-confidence legs only. Conservative 2-leg multis with tight correlation controls.",
        "risk_label": "Very Low Risk",
        "min_confidence": 60,
        "min_edge": 0.02,
        "min_ev": 0.01,
        "max_multi_legs": 2,
        "min_multi_legs": 2,
        "mode": "safe",
        "max_combined_odds": 8.0,
        "max_correlation": 0.5,
        "leg_top_n": 15,
    },
    "balanced": {
        "label": "Balanced",
        "icon": "⚖",
        "description": "Solid mix of value and confidence. 2–3 leg multis with moderate risk.",
        "risk_label": "Low–Medium Risk",
        "min_confidence": 45,
        "min_edge": 0.03,
        "min_ev": 0.02,
        "max_multi_legs": 3,
        "min_multi_legs": 2,
        "mode": "safe",
        "max_combined_odds": 20.0,
        "max_correlation": 0.6,
        "leg_top_n": 20,
    },
    "value": {
        "label": "Value",
        "icon": "💎",
        "description": "Edge-focused analysis targeting genuine market inefficiencies. 3–4 leg multis.",
        "risk_label": "Medium Risk",
        "min_confidence": 35,
        "min_edge": 0.04,
        "min_ev": 0.03,
        "max_multi_legs": 4,
        "min_multi_legs": 2,
        "mode": "value",
        "max_combined_odds": 40.0,
        "max_correlation": 0.65,
        "leg_top_n": 25,
    },
    "aggressive": {
        "label": "Aggressive",
        "icon": "🔥",
        "description": "High EV focus with elevated risk tolerance. 3–4 leg multis at extended odds.",
        "risk_label": "High Risk",
        "min_confidence": 25,
        "min_edge": 0.03,
        "min_ev": 0.02,
        "max_multi_legs": 4,
        "min_multi_legs": 3,
        "mode": "value",
        "max_combined_odds": 80.0,
        "max_correlation": 0.70,
        "leg_top_n": 30,
    },
    "longshot": {
        "label": "Longshot",
        "icon": "🎯",
        "description": "Speculative high-value plays. Max legs, extended odds — high risk, high reward.",
        "risk_label": "Very High Risk",
        "min_confidence": 15,
        "min_edge": 0.02,
        "min_ev": 0.01,
        "max_multi_legs": 5,
        "min_multi_legs": 4,
        "mode": "value",
        "max_combined_odds": 500.0,
        "max_correlation": 0.75,
        "leg_top_n": 40,
    },
}

# Pipeline stage definitions
PIPELINE_STAGES = [
    ("fetching_fixtures",   "Fetching fixture data"),
    ("fetching_odds",       "Fetching bookmaker odds"),
    ("fetching_weather",    "Fetching weather conditions"),
    ("gathering_intel",     "Gathering news & injury intelligence"),
    ("building_features",   "Building feature set"),
    ("scoring_legs",        "Scoring candidate legs"),
    ("constructing_multis", "Constructing multis"),
    ("ranking_output",      "Ranking final output"),
]


# ── Job data structures ───────────────────────────────────────────────────────

@dataclass
class StageInfo:
    key: str
    label: str
    status: str = "pending"   # pending | running | done | error
    detail: str = ""


@dataclass
class GenerateJob:
    job_id: str
    fixture_ids: List[int]
    multi_style: str
    n_multis: int
    status: str = "pending"     # pending | running | complete | error
    stages: List[StageInfo] = field(default_factory=list)
    result: Optional[Dict] = None
    error: Optional[str] = None
    started_at: float = 0.0
    elapsed_seconds: float = 0.0
    n_candidate_legs: int = 0
    n_multis_generated: int = 0

    def to_dict(self) -> Dict:
        elapsed = (
            time.time() - self.started_at
            if self.status == "running" and self.started_at
            else self.elapsed_seconds
        )
        return {
            "job_id": self.job_id,
            "status": self.status,
            "multi_style": self.multi_style,
            "n_multis_requested": self.n_multis,
            "n_fixtures_selected": len(self.fixture_ids),
            "n_candidate_legs": self.n_candidate_legs,
            "n_multis_generated": self.n_multis_generated,
            "elapsed_seconds": round(elapsed, 1),
            "stages": [
                {
                    "key": s.key,
                    "label": s.label,
                    "status": s.status,
                    "detail": s.detail,
                }
                for s in self.stages
            ],
            "result": self.result,
            "error": self.error,
        }


# ── Singleton job registry ────────────────────────────────────────────────────

_jobs: Dict[str, GenerateJob] = {}
_jobs_lock = threading.Lock()

# Cache for weekend fixtures so we don't reload on every poll
_weekend_fixtures_cache: Optional[Tuple[float, List[Dict]]] = None
_weekend_cache_ttl = 300.0   # 5 minutes


# ── GenerateService ───────────────────────────────────────────────────────────

class GenerateService:
    """Manages async generation jobs."""

    @staticmethod
    def get_style_profiles() -> Dict[str, Dict]:
        """Return all style profile definitions for the frontend."""
        return {
            k: {
                "label": v["label"],
                "icon": v["icon"],
                "description": v["description"],
                "risk_label": v["risk_label"],
            }
            for k, v in STYLE_PROFILES.items()
        }

    @staticmethod
    def get_weekend_fixtures() -> List[Dict]:
        """
        Return upcoming AFL fixtures for the current or next weekend round.

        Strategy:
          1. Load all upcoming fixtures from the configured data source
          2. Find the soonest round number that has at least one unplayed game
          3. Return all fixtures in that round, enriched with team/venue info
        """
        global _weekend_fixtures_cache
        now = time.time()
        if _weekend_fixtures_cache is not None:
            ts, cached = _weekend_fixtures_cache
            if now - ts < _weekend_cache_ttl:
                return cached

        try:
            from app.data_ingestion.loader import DataLoader
            loader = DataLoader()
            upcoming = loader.load_upcoming_fixtures_df()

            if upcoming.empty:
                logger.warning("No upcoming fixtures available for weekend display")
                return []

            # Sort by date/scheduled_utc and pick the soonest round
            if "scheduled_utc" in upcoming.columns:
                upcoming = upcoming.sort_values("scheduled_utc", na_position="last")
            elif "date" in upcoming.columns:
                upcoming = upcoming.sort_values("date", na_position="last")

            target_round = int(upcoming["round"].iloc[0])
            round_fixtures = upcoming[upcoming["round"] == target_round]

            # Load venues for join (demo mode)
            venues_map: Dict[int, str] = {}
            try:
                import pandas as pd
                vpath = settings.demo_data_dir / "venues.csv"
                if vpath.exists():
                    vdf = pd.read_csv(vpath)
                    for _, row in vdf.iterrows():
                        try:
                            venues_map[int(row["venue_id"])] = str(row["name"])
                        except (ValueError, KeyError):
                            pass
            except Exception:
                pass

            result = []
            for _, row in round_fixtures.iterrows():
                fixture_id = int(row.get("fixture_id", 0))
                home_name = str(row.get("home_team_name", row.get("home_team_id", "Home")))
                away_name = str(row.get("away_team_name", row.get("away_team_id", "Away")))

                # Venue: try venues_map → home team mapping → unknown
                venue_id = int(row.get("venue_id", 0))
                venue = venues_map.get(venue_id) or _TEAM_VENUE.get(home_name, "")

                # Date/time display — prefer AEST offset from UTC
                date_str = str(row.get("date", ""))
                time_str = str(row.get("time", ""))
                scheduled_utc = str(row.get("scheduled_utc", ""))

                # Build display time: if we have a UTC timestamp, convert to AEST
                display_time = ""
                if scheduled_utc and scheduled_utc not in ("", "nan", "None"):
                    try:
                        from datetime import timezone as tz
                        import pytz
                        utc_dt = datetime.fromisoformat(scheduled_utc.replace("Z", "+00:00"))
                        aest = pytz.timezone("Australia/Melbourne")
                        local_dt = utc_dt.astimezone(aest)
                        display_time = local_dt.strftime("%a %d %b · %I:%M %p AEST")
                    except Exception:
                        display_time = f"{date_str} {time_str}".strip()
                else:
                    display_time = f"{date_str} {time_str}".strip()

                result.append({
                    "fixture_id": fixture_id,
                    "round": int(row.get("round", 0)),
                    "season": int(row.get("season", 0)),
                    "home_team": home_name,
                    "away_team": away_name,
                    "venue": venue or "TBC",
                    "display_time": display_time,
                    "date": date_str,
                    "time": time_str,
                    "status": str(row.get("status", "upcoming")),
                })

            _weekend_fixtures_cache = (now, result)
            return result

        except Exception as exc:
            logger.exception("get_weekend_fixtures failed: %s", exc)
            return []

    @staticmethod
    def start_job(fixture_ids: List[int], multi_style: str, n_multis: int) -> str:
        """Validate inputs, create a GenerateJob, launch background thread, return job_id."""
        if multi_style not in STYLE_PROFILES:
            raise ValueError(f"Unknown multi_style '{multi_style}'. Valid: {list(STYLE_PROFILES)}")
        if not fixture_ids:
            raise ValueError("At least one fixture must be selected.")
        if n_multis < 1 or n_multis > 20:
            raise ValueError("n_multis must be between 1 and 20.")

        job_id = uuid.uuid4().hex[:10]
        stages = [StageInfo(key=k, label=l) for k, l in PIPELINE_STAGES]
        job = GenerateJob(
            job_id=job_id,
            fixture_ids=list(fixture_ids),
            multi_style=multi_style,
            n_multis=n_multis,
            stages=stages,
        )
        with _jobs_lock:
            _jobs[job_id] = job

        t = threading.Thread(target=GenerateService._run_job, args=(job,), daemon=True)
        t.start()
        return job_id

    @staticmethod
    def get_job_status(job_id: str) -> Optional[Dict]:
        with _jobs_lock:
            job = _jobs.get(job_id)
        if job is None:
            return None
        return job.to_dict()

    # ── Internal job runner ───────────────────────────────────────────────────

    @staticmethod
    def _run_job(job: GenerateJob) -> None:
        """Execute the staged analysis pipeline in a background thread."""
        job.status = "running"
        job.started_at = time.time()

        def stage_start(idx: int, detail: str = "") -> None:
            job.stages[idx].status = "running"
            job.stages[idx].detail = detail

        def stage_done(idx: int, detail: str = "") -> None:
            job.stages[idx].status = "done"
            if detail:
                job.stages[idx].detail = detail

        def stage_error(idx: int, detail: str) -> None:
            job.stages[idx].status = "error"
            job.stages[idx].detail = detail

        try:
            profile = STYLE_PROFILES[job.multi_style]

            # ── Stage 0: Load fixtures ────────────────────────────────────────
            stage_start(0, "Initialising data connection...")
            from app.data_ingestion.loader import DataLoader
            loader = DataLoader()
            all_upcoming = loader.load_upcoming_fixtures_df()

            if all_upcoming.empty:
                raise RuntimeError("No upcoming fixtures available from data source.")

            # Filter to selected fixtures only
            import pandas as pd
            selected_fixtures = all_upcoming[
                all_upcoming["fixture_id"].astype(int).isin(job.fixture_ids)
            ]
            if selected_fixtures.empty:
                # Fallback: use all upcoming (selection might not have matched IDs)
                selected_fixtures = all_upcoming
                logger.warning(
                    "Selected fixture_ids %s not found in upcoming fixtures; "
                    "using all %d upcoming fixtures",
                    job.fixture_ids, len(all_upcoming)
                )

            n_selected = len(selected_fixtures)
            stage_done(0, f"{n_selected} fixture{'s' if n_selected != 1 else ''} loaded")

            # ── Stage 1: Bookmaker odds ───────────────────────────────────────
            stage_start(1, "Connecting to bookmaker feeds...")
            odds_df = loader.load_odds_df()
            odds_note = f"{len(odds_df)} odds records" if not odds_df.empty else "no odds available"
            stage_done(1, odds_note)

            # ── Stage 2: Weather ──────────────────────────────────────────────
            stage_start(2, "Querying weather services...")
            weather_df = loader.load_weather_df()
            wx_note = f"{len(weather_df)} venue forecasts" if not weather_df.empty else "using defaults"
            stage_done(2, wx_note)

            # ── Stage 3: Edge intelligence ────────────────────────────────────
            stage_start(3, "Scanning news and injury reports...")
            signals = loader.load_all_edge_signals()
            intel_note = f"{len(signals)} signals detected" if signals else "no active signals"
            stage_done(3, intel_note)

            # ── Stage 4: Feature engineering ─────────────────────────────────
            stage_start(4, "Computing team ratings and rolling statistics...")
            from app.features.pipeline import FeaturePipeline
            feature_pipeline = FeaturePipeline(loader)
            team_features = feature_pipeline.get_team_features()
            feat_note = (
                f"{len(team_features)} feature vectors"
                if not team_features.empty else "baseline features"
            )
            stage_done(4, feat_note)

            # ── Stage 5: Score candidate legs ────────────────────────────────
            stage_start(5, f"Analysing markets across {n_selected} fixture(s)...")

            from app.pricing.models import ModelRegistry
            from app.pricing.calibration import IsotonicCalibrator
            from app.core.metrics import compute_ev, compute_edge, compute_confidence_score

            registry = ModelRegistry()
            match_model = registry.load("match_win_ensemble")
            if match_model is None:
                from app.services.training import TrainingService
                svc = TrainingService(loader)
                svc.run_match_model_training()
                match_model = registry.load("match_win_ensemble")

            player_model = registry.load("player_disposals_model")
            if player_model is None:
                from app.services.training import TrainingService
                svc = TrainingService(loader)
                svc.run_player_disposals_training()
                player_model = registry.load("player_disposals_model")

            player_calibrator = registry.load("player_disposals_calibrator") or IsotonicCalibrator()

            # Reuse pipeline leg-generation logic from the main pipeline
            from app.services.pipeline import PredictionPipeline
            inner_pipeline = PredictionPipeline(loader=loader)
            inner_pipeline._build_lookups()

            # Override loaded data with what we've already fetched
            all_legs = []
            legs_meta: Dict[str, Dict] = {}

            for _, fixture_row in selected_fixtures.iterrows():
                fid = int(fixture_row["fixture_id"])
                legs, meta = inner_pipeline._generate_fixture_legs(
                    fixture_id=fid,
                    fixture=fixture_row,
                    team_features=team_features,
                    match_model=match_model,
                    player_model=player_model,
                    player_calibrator=player_calibrator,
                )
                all_legs.extend(legs)
                legs_meta.update(meta)

            job.n_candidate_legs = len(all_legs)

            # Apply style-profile confidence filter
            min_conf = profile.get("min_confidence", 0)
            filtered_legs = [l for l in all_legs if l.confidence_score >= min_conf]

            # Apply style-profile EV filter
            min_ev_p = profile.get("min_ev", 0.0)
            filtered_legs = [l for l in filtered_legs if l.ev >= min_ev_p]

            # Rank by mode
            from app.optimizer.multi_builder import LegRanker
            ranker = LegRanker(min_edge=profile.get("min_edge", 0.02), min_ev=min_ev_p)
            leg_top_n = profile.get("leg_top_n", 20)
            if profile.get("mode") == "safe":
                candidate_legs = ranker.get_safe_legs(filtered_legs, top_n=leg_top_n)
            else:
                candidate_legs = ranker.get_value_legs(filtered_legs, top_n=leg_top_n)

            stage_done(5, f"{len(candidate_legs)} qualifying legs from {job.n_candidate_legs} candidates")

            # ── Stage 6: Construct multis ─────────────────────────────────────
            stage_start(6, f"Building {job.multi_style} multi combinations...")

            from app.optimizer.multi_builder import MultiBuilder
            builder = MultiBuilder(
                min_legs=profile["min_multi_legs"],
                max_legs=profile["max_multi_legs"],
                max_correlation=profile["max_correlation"],
                min_ev=max(profile.get("min_ev", 0.0), 0.005),
            )

            # Generate enough candidates (3× requested to allow odds filtering)
            pool_size = max(job.n_multis * 4, 20)
            raw_multis = builder.build(
                candidate_legs,
                mode=profile["mode"],
                max_results=pool_size,
            )

            # Apply max combined odds filter
            max_odds = profile.get("max_combined_odds", 500.0)
            filtered_multis = [m for m in raw_multis if m.combined_odds <= max_odds]

            # If style filter was too strict, progressively relax odds cap
            if len(filtered_multis) < job.n_multis and raw_multis:
                filtered_multis = raw_multis  # use full pool

            stage_done(6, f"{len(filtered_multis)} valid combinations found")

            # ── Stage 7: Rank and trim ────────────────────────────────────────
            stage_start(7, f"Selecting top {job.n_multis} multi(s) by quality score...")

            final_multis = filtered_multis[: job.n_multis]
            job.n_multis_generated = len(final_multis)

            # Build the leg dict lookup for serialisation
            leg_dict_lookup = {
                l.leg_id: inner_pipeline._leg_to_dict(l, legs_meta)
                for l in all_legs
            }

            multis_out = [
                GenerateService._multi_to_output(m, leg_dict_lookup, job.multi_style, profile)
                for m in final_multis
            ]

            elapsed = time.time() - job.started_at
            stage_done(7, f"{job.n_multis_generated} multi(s) ready in {elapsed:.1f}s")

            job.result = {
                "job_id": job.job_id,
                "multi_style": job.multi_style,
                "style_label": profile["label"],
                "n_requested": job.n_multis,
                "n_generated": job.n_multis_generated,
                "n_candidate_legs": job.n_candidate_legs,
                "n_qualifying_legs": len(candidate_legs),
                "elapsed_seconds": round(elapsed, 1),
                "multis": multis_out,
            }
            job.elapsed_seconds = elapsed
            job.status = "complete"

        except Exception as exc:
            logger.exception("GenerateJob %s failed: %s", job.job_id, exc)
            # Mark the running stage as errored
            for s in job.stages:
                if s.status == "running":
                    stage_error(job.stages.index(s), str(exc)[:120])
            job.error = str(exc)
            job.elapsed_seconds = time.time() - job.started_at if job.started_at else 0
            job.status = "error"

    # ── Serialisation helpers ─────────────────────────────────────────────────

    @staticmethod
    def _multi_to_output(multi, leg_dict_lookup: Dict, style: str, profile: Dict) -> Dict:
        """Serialise a Multi into a rich output dict for the UI."""
        from app.services.pipeline import _confidence_label

        legs_out = []
        for lid in (multi.leg_ids or []):
            ld = leg_dict_lookup.get(lid)
            if ld:
                legs_out.append({
                    "leg_id": lid,
                    "match_label": ld.get("match_label", ""),
                    "selection_label": ld.get("selection_label", ""),
                    "market_label": ld.get("market_label", ""),
                    "decimal_odds": ld.get("decimal_odds"),
                    "model_probability": ld.get("model_probability"),
                    "market_probability": ld.get("market_probability"),
                    "edge": ld.get("edge"),
                    "ev": ld.get("ev"),
                    "confidence_score": ld.get("confidence_score"),
                    "confidence_label": ld.get("confidence_label", ""),
                    "team_name": ld.get("team_name", ""),
                    "player_name": ld.get("player_name", ""),
                    "start_time": ld.get("start_time", ""),
                    "explanation": ld.get("explanation", ""),
                })

        # Risk profile label
        risk = multi.risk_score
        if risk < 30:
            risk_label = "Low Risk"
        elif risk < 55:
            risk_label = "Moderate Risk"
        elif risk < 75:
            risk_label = "High Risk"
        else:
            risk_label = "Very High Risk"

        return {
            "multi_id": multi.multi_id,
            "n_legs": multi.n_legs,
            "multi_type": multi.multi_type,
            "style": style,
            "style_label": profile["label"],
            "combined_odds": round(multi.combined_odds, 2),
            "raw_probability": round(getattr(multi, "raw_probability", multi.adjusted_probability), 4),
            "adjusted_probability": round(multi.adjusted_probability, 4),
            "ev": round(multi.ev, 4),
            "correlation_score": round(multi.correlation_score, 3),
            "correlation_label": multi.correlation_label,
            "risk_score": round(multi.risk_score, 1),
            "risk_label": risk_label,
            "explanation": multi.explanation,
            "leg_ids": multi.leg_ids,
            "legs": legs_out,
        }
