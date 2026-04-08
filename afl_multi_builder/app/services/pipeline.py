"""
Main prediction pipeline service.
Generates candidate legs and multis for upcoming fixtures.
"""
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from loguru import logger

from app.data_ingestion.loader import DataLoader
from app.features.pipeline import FeaturePipeline
from app.pricing.models import (
    CalibratedModel, PlayerDisposalsModel, ModelRegistry, EnsembleModel
)
from app.pricing.calibration import IsotonicCalibrator
from app.pricing.odds_processor import OddsProcessor, ProcessedOdds
from app.correlation.engine import Leg
from app.optimizer.multi_builder import MultiBuilder, LegRanker, Multi
from app.core.metrics import (
    compute_ev, compute_edge, compute_confidence_score, implied_probability
)
from app.core.config import settings


class PredictionPipeline:
    """
    End-to-end prediction pipeline:
    1. Load upcoming fixtures
    2. Build features
    3. Generate model predictions
    4. Process odds
    5. Compute edge/EV
    6. Build and rank legs
    7. Build and rank multis
    """

    def __init__(self, loader: Optional[DataLoader] = None):
        self.loader = loader or DataLoader()
        self.feature_pipeline = FeaturePipeline(self.loader)
        self.odds_processor = OddsProcessor(self.loader)
        self.registry = ModelRegistry()
        self.leg_ranker = LegRanker()
        self.multi_builder = MultiBuilder()
        self.run_id = str(uuid.uuid4())[:8]

    def run(self) -> Dict:
        """Execute full pipeline. Returns structured results."""
        logger.info(f"[{self.run_id}] Starting prediction pipeline...")
        start_time = datetime.utcnow()

        # 1. Get upcoming fixtures
        fixtures = self.loader.load_fixtures_df()
        upcoming = fixtures[fixtures["status"] == "upcoming"]
        if upcoming.empty:
            # Fall back to last few completed fixtures for demo
            completed = fixtures[fixtures["status"] == "completed"].tail(5)
            upcoming = completed
            logger.info("No upcoming fixtures found, using recent completed for demo.")

        logger.info(f"Processing {len(upcoming)} fixtures...")

        # 2. Load/train models
        match_model = self._get_match_model()
        player_model = self._get_player_model()
        player_calibrator = self._get_player_calibrator()

        # 3. Build team features
        team_features = self.feature_pipeline.get_team_features()

        # 4. Generate candidate legs
        all_legs: List[Leg] = []

        for _, fixture in upcoming.iterrows():
            fixture_id = int(fixture["fixture_id"])
            legs = self._generate_fixture_legs(
                fixture_id=fixture_id,
                fixture=fixture,
                team_features=team_features,
                match_model=match_model,
                player_model=player_model,
                player_calibrator=player_calibrator,
            )
            all_legs.extend(legs)

        logger.info(f"Generated {len(all_legs)} candidate legs.")

        # 5. Rank legs
        value_legs = self.leg_ranker.get_value_legs(all_legs, top_n=20)
        safe_legs = self.leg_ranker.get_safe_legs(all_legs, top_n=20)

        # 6. Build multis
        value_multis = self.multi_builder.build(value_legs, mode="value", max_results=15)
        safe_multis = self.multi_builder.build(safe_legs, mode="safe", max_results=15)

        # Same-game multis per fixture
        same_game_multis: List[Multi] = []
        for fixture_id in upcoming["fixture_id"].unique():
            sgm = self.multi_builder.build_same_game_multis(
                all_legs, fixture_id=int(fixture_id), max_results=5
            )
            same_game_multis.extend(sgm)

        elapsed = (datetime.utcnow() - start_time).total_seconds()
        logger.info(f"[{self.run_id}] Pipeline complete in {elapsed:.2f}s")

        return {
            "run_id": self.run_id,
            "timestamp": datetime.utcnow().isoformat(),
            "n_fixtures": len(upcoming),
            "n_candidate_legs": len(all_legs),
            "elapsed_seconds": round(elapsed, 2),
            "value_legs": [self._leg_to_dict(l) for l in value_legs],
            "safe_legs": [self._leg_to_dict(l) for l in safe_legs],
            "value_multis": [self._multi_to_dict(m) for m in value_multis],
            "safe_multis": [self._multi_to_dict(m) for m in safe_multis],
            "same_game_multis": [self._multi_to_dict(m) for m in same_game_multis],
        }

    def _generate_fixture_legs(
        self,
        fixture_id: int,
        fixture: pd.Series,
        team_features: pd.DataFrame,
        match_model,
        player_model,
        player_calibrator,
    ) -> List[Leg]:
        """Generate all candidate legs for a single fixture."""
        legs: List[Leg] = []

        # ── Head-to-head legs ─────────────────────────────────────────────
        h2h_legs = self._generate_h2h_legs(
            fixture_id, fixture, team_features, match_model
        )
        legs.extend(h2h_legs)

        # ── Player disposals legs ──────────────────────────────────────────
        if player_model is not None:
            player_legs = self._generate_player_legs(
                fixture_id, fixture, player_model, player_calibrator
            )
            legs.extend(player_legs)

        return legs

    def _generate_h2h_legs(
        self, fixture_id: int, fixture: pd.Series, team_features: pd.DataFrame, match_model
    ) -> List[Leg]:
        """Generate head-to-head win/loss legs."""
        legs = []

        # Get model prediction for this fixture
        fx_features = team_features[team_features["fixture_id"] == fixture_id]

        if not fx_features.empty and match_model is not None:
            X_cols = [c for c in fx_features.columns
                      if c.startswith(("elo_", "home_roll_", "away_roll_", "diff_roll_",
                                       "temperature_c", "wind_speed_kmh", "is_rain"))]
            X_cols = [c for c in X_cols if c in fx_features.columns]
            if X_cols:
                X_pred = fx_features[X_cols].fillna(0)
                try:
                    probs = match_model.predict_proba(X_pred)
                    cal_home_prob = float(probs[0, 1])
                    raw_probs_raw = match_model.predict_proba_raw(X_pred)
                    raw_home_prob = float(raw_probs_raw[0, 1])
                except Exception:
                    cal_home_prob = float(fx_features.get("elo_win_prob_home", pd.Series([0.5])).iloc[0])
                    raw_home_prob = cal_home_prob
            else:
                cal_home_prob = float(fx_features.get("elo_win_prob_home", pd.Series([0.5])).iloc[0])
                raw_home_prob = cal_home_prob
        else:
            # Fallback to Elo
            elo_prob = float(fixture.get("elo_win_prob_home", 0.5)) if "elo_win_prob_home" in fixture else 0.5
            cal_home_prob = elo_prob
            raw_home_prob = elo_prob

        cal_away_prob = 1.0 - cal_home_prob

        # Get market odds
        h2h_odds = self._get_best_odds(fixture_id, "head_to_head")
        home_id = int(fixture["home_team_id"])
        away_id = int(fixture["away_team_id"])

        for selection, model_prob, team_id in [
            ("home_win", cal_home_prob, home_id),
            ("away_win", cal_away_prob, away_id),
        ]:
            market_data = h2h_odds.get(selection)
            if market_data is None:
                # No odds available — use default odds from model
                decimal_odds = round(1.0 / max(model_prob, 0.05), 2)
                vig_adj_prob = model_prob * 0.95  # rough market estimate
            else:
                decimal_odds = market_data["decimal_odds"]
                vig_adj_prob = market_data["vig_adj_prob"]

            edge = compute_edge(model_prob, vig_adj_prob)
            ev = compute_ev(model_prob, decimal_odds)
            confidence = compute_confidence_score(model_prob, vig_adj_prob, edge, n_historical_games=30)

            explanation = (
                f"Model: {model_prob:.1%}, Market: {vig_adj_prob:.1%}, "
                f"Edge: {edge:+.1%}, EV: {ev:+.1%}. "
                f"{'Home advantage factored in.' if selection == 'home_win' else 'Away team upside.'}"
            )

            import hashlib
            leg_id = "L_" + hashlib.md5(f"{fixture_id}_{selection}".encode()).hexdigest()[:8]

            legs.append(Leg(
                leg_id=leg_id,
                fixture_id=fixture_id,
                player_id=None,
                team_id=team_id,
                market_type="head_to_head",
                selection=selection,
                decimal_odds=decimal_odds,
                calibrated_probability=model_prob,
                ev=ev,
                confidence_score=confidence,
                explanation=explanation,
            ))

        return legs

    def _generate_player_legs(
        self, fixture_id: int, fixture: pd.Series, player_model, player_calibrator
    ) -> List[Leg]:
        """Generate player disposals over/under legs."""
        legs = []
        odds_df = self.loader.load_odds_df()
        if odds_df.empty:
            return []

        player_odds = odds_df[
            (odds_df["fixture_id"] == fixture_id) &
            (odds_df["market_type"] == "player_disposals")
        ]
        if player_odds.empty:
            return []

        # Group by player and line
        processed = {}
        for _, row in player_odds.iterrows():
            sel = str(row["selection"])
            # Parse: "player_{id}_over_{line}" or "player_{id}_under_{line}"
            parts = sel.split("_")
            if len(parts) < 4:
                continue
            try:
                pid = int(parts[1])
                direction = parts[2]  # over / under
                line = float(parts[3])
                key = (pid, line)
                if key not in processed:
                    processed[key] = {}
                processed[key][direction] = {
                    "decimal_odds": float(row["decimal_odds"]),
                    "selection": sel,
                    "bookmaker": str(row["bookmaker"]),
                }
            except (ValueError, IndexError):
                continue

        players = self.loader.load_players_df()
        home_id = int(fixture["home_team_id"])
        away_id = int(fixture["away_team_id"])

        for (pid, line), directions in processed.items():
            player_info = players[players["player_id"] == pid]
            team_id = int(player_info["team_id"].iloc[0]) if not player_info.empty else None

            # Get features for this player
            feat = self.feature_pipeline.player_engineer.get_player_prediction_features(
                player_id=pid, fixture_id=fixture_id
            )
            if feat is None:
                continue

            feat_df = pd.DataFrame([feat])
            feature_cols = [
                "roll_mean_3", "roll_mean_5", "roll_mean_10", "roll_std_5",
                "roll_max_5", "roll_min_5", "consistency_cv", "form_trend",
                "n_games", "opp_disposals_allowed_mean",
                "pos_midfielder", "pos_forward", "pos_defender", "pos_ruckman",
            ]
            feat_df = feat_df.reindex(columns=feature_cols, fill_value=0)

            try:
                raw_over_prob = float(player_model.predict_over_prob(feat_df, line)[0])
                cal_over_prob = float(player_calibrator.calibrate(np.array([raw_over_prob]))[0])
            except Exception:
                # Fallback: use rolling mean vs line
                roll_mean = feat.get("roll_mean_5", 20.0)
                from scipy.stats import norm
                raw_over_prob = float(1.0 - norm.cdf(line, loc=roll_mean, scale=5.0))
                cal_over_prob = raw_over_prob

            cal_over_prob = float(np.clip(cal_over_prob, 0.05, 0.95))
            cal_under_prob = 1.0 - cal_over_prob

            for direction, probs in [("over", cal_over_prob), ("under", cal_under_prob)]:
                if direction not in directions:
                    continue
                d = directions[direction]
                decimal_odds = d["decimal_odds"]
                selection = d["selection"]

                # Market implied prob
                opp_sel = directions.get("under" if direction == "over" else "over", {})
                if opp_sel:
                    raw_imp = 1.0 / decimal_odds
                    raw_opp_imp = 1.0 / opp_sel.get("decimal_odds", decimal_odds)
                    total_imp = raw_imp + raw_opp_imp
                    vig_adj_prob = raw_imp / total_imp if total_imp > 0 else 0.5
                else:
                    vig_adj_prob = 1.0 / decimal_odds

                edge = compute_edge(probs, vig_adj_prob)
                ev = compute_ev(probs, decimal_odds)
                confidence = compute_confidence_score(probs, vig_adj_prob, edge, n_historical_games=feat.get("n_games", 0))

                roll_mean = feat.get("roll_mean_5", 20.0)
                explanation = (
                    f"Player {pid} {direction} {line} disposals. "
                    f"5-game avg: {roll_mean:.1f}. "
                    f"Model: {probs:.1%}, Market: {vig_adj_prob:.1%}, "
                    f"Edge: {edge:+.1%}, EV: {ev:+.1%}."
                )

                import hashlib
                leg_id = "L_" + hashlib.md5(f"{fixture_id}_{selection}".encode()).hexdigest()[:8]

                legs.append(Leg(
                    leg_id=leg_id,
                    fixture_id=fixture_id,
                    player_id=pid,
                    team_id=team_id,
                    market_type="player_disposals",
                    selection=selection,
                    decimal_odds=decimal_odds,
                    calibrated_probability=probs,
                    ev=ev,
                    confidence_score=confidence,
                    explanation=explanation,
                ))

        return legs

    def _get_best_odds(self, fixture_id: int, market_type: str) -> Dict:
        """Get best available odds per selection for a market."""
        odds_df = self.loader.load_odds_df()
        if odds_df.empty:
            return {}
        fx = odds_df[(odds_df["fixture_id"] == fixture_id) & (odds_df["market_type"] == market_type)]
        if fx.empty:
            return {}

        result = {}
        for sel in fx["selection"].unique():
            sel_odds = fx[fx["selection"] == sel]
            best_idx = sel_odds["decimal_odds"].idxmax()
            best = sel_odds.loc[best_idx]
            # Compute vig-adjusted prob
            all_sels = fx["selection"].unique()
            all_imp = []
            for s in all_sels:
                best_d = fx[fx["selection"] == s]["decimal_odds"].max()
                all_imp.append(1.0 / best_d)
            total_imp = sum(all_imp)
            raw_imp = 1.0 / float(best["decimal_odds"])
            vig_adj = raw_imp / total_imp if total_imp > 0 else raw_imp

            result[str(sel)] = {
                "decimal_odds": float(best["decimal_odds"]),
                "bookmaker": str(best["bookmaker"]),
                "raw_implied_prob": round(raw_imp, 5),
                "vig_adj_prob": round(vig_adj, 5),
            }
        return result

    def _get_match_model(self):
        """Load trained match model or train fresh if not available."""
        model = self.registry.load("match_win_ensemble")
        if model is None:
            logger.info("No trained match model found, training now...")
            from app.services.training import TrainingService
            svc = TrainingService(self.loader)
            svc.run_match_model_training()
            model = self.registry.load("match_win_ensemble")
        return model

    def _get_player_model(self):
        """Load trained player model or train fresh."""
        model = self.registry.load("player_disposals_model")
        if model is None:
            logger.info("No trained player model found, training now...")
            from app.services.training import TrainingService
            svc = TrainingService(self.loader)
            svc.run_player_disposals_training()
            model = self.registry.load("player_disposals_model")
        return model

    def _get_player_calibrator(self):
        """Load player calibrator."""
        cal = self.registry.load("player_disposals_calibrator")
        if cal is None:
            cal = IsotonicCalibrator()
        return cal

    @staticmethod
    def _leg_to_dict(leg: Leg) -> Dict:
        return {
            "leg_id": leg.leg_id,
            "fixture_id": leg.fixture_id,
            "player_id": leg.player_id,
            "team_id": leg.team_id,
            "market_type": leg.market_type,
            "selection": leg.selection,
            "decimal_odds": round(leg.decimal_odds, 2),
            "model_probability": round(leg.calibrated_probability, 4),
            "ev": round(leg.ev, 4),
            "confidence_score": round(leg.confidence_score, 1),
            "explanation": leg.explanation,
        }

    @staticmethod
    def _multi_to_dict(multi: Multi) -> Dict:
        return {
            "multi_id": multi.multi_id,
            "n_legs": multi.n_legs,
            "multi_type": multi.multi_type,
            "combined_odds": round(multi.combined_odds, 2),
            "adjusted_probability": round(multi.adjusted_probability, 4),
            "ev": round(multi.ev, 4),
            "correlation_score": round(multi.correlation_score, 3),
            "correlation_label": multi.correlation_label,
            "risk_score": round(multi.risk_score, 1),
            "explanation": multi.explanation,
            "leg_ids": multi.leg_ids,
        }
