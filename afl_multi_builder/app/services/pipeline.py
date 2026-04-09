"""
Main prediction pipeline service.
Generates candidate legs and multis for upcoming fixtures.
All outputs are fully human-readable: team names, player names, plain-English labels.
"""
import hashlib
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from loguru import logger

from app.data_ingestion.loader import DataLoader
from app.data_ingestion.sportradar_loader import NoUpcomingFixturesError
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


def _confidence_label(score: float) -> str:
    if score >= 75:
        return "High"
    elif score >= 50:
        return "Medium"
    return "Low"


def _make_selection_label(
    selection: str,
    team_name: str = "",
    player_name: str = "",
    line: float = None,
) -> str:
    """Convert raw selection string into human-readable label."""
    sel = selection.strip().lower()

    if sel == "home_win":
        return f"{team_name} to Win" if team_name else "Home Team to Win"
    if sel == "away_win":
        return f"{team_name} to Win" if team_name else "Away Team to Win"

    if sel.startswith("over_"):
        v = sel[5:]
        return f"Over {v} Total Score"
    if sel.startswith("under_"):
        v = sel[6:]
        return f"Under {v} Total Score"

    if sel.startswith("home_line_"):
        v = sel[10:]
        sign = "+" if not v.startswith("-") else ""
        return f"{team_name} {sign}{v} Line" if team_name else f"Home {sign}{v} Line"
    if sel.startswith("away_line_"):
        v = sel[10:]
        sign = "+" if not v.startswith("-") else ""
        return f"{team_name} {sign}{v} Line" if team_name else f"Away {sign}{v} Line"

    # player_disposals: player_{id}_over_{line} or player_{id}_under_{line}
    import re
    m = re.match(r"player_(\d+)_(over|under)_(.+)", sel)
    if m:
        direction = m.group(2).title()
        line_val = m.group(3)
        name = player_name if player_name else f"Player {m.group(1)}"
        return f"{name} {direction} {line_val} Disposals"

    # Fallback: pretty-print
    return selection.replace("_", " ").title()


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

    All outputs include full human-readable names and labels.
    """

    def __init__(self, loader: Optional[DataLoader] = None):
        self.loader = loader or DataLoader()
        self.feature_pipeline = FeaturePipeline(self.loader)
        self.odds_processor = OddsProcessor(self.loader)
        self.registry = ModelRegistry()
        self.leg_ranker = LegRanker()
        self.multi_builder = MultiBuilder()
        self.run_id = str(uuid.uuid4())[:8]

        # Name lookups — populated in run()
        self._team_names: Dict[int, str] = {}
        self._player_names: Dict[int, str] = {}
        self._fixture_info: Dict[int, Dict] = {}

    def run(self) -> Dict:
        """Execute full pipeline. Returns structured, human-readable results."""
        logger.info(f"[{self.run_id}] Starting prediction pipeline...")
        start_time = datetime.utcnow()

        # Build name lookups first
        self._build_lookups()

        # 1. Get upcoming fixtures — strict future-only, never falls back to completed
        upcoming = self.loader.load_upcoming_fixtures_df()
        if upcoming.empty:
            raise NoUpcomingFixturesError(
                "No upcoming AFL fixtures found. "
                "The season may be over or fixtures not yet published."
            )

        logger.info(f"Processing {len(upcoming)} fixtures...")

        # 2. Load/train models
        match_model = self._get_match_model()
        player_model = self._get_player_model()
        player_calibrator = self._get_player_calibrator()

        # 3. Build team features
        team_features = self.feature_pipeline.get_team_features()

        # 4. Generate candidate legs
        all_legs: List[Leg] = []
        legs_meta: Dict[str, Dict] = {}  # leg_id -> extra display metadata

        for _, fixture in upcoming.iterrows():
            fixture_id = int(fixture["fixture_id"])
            legs, meta = self._generate_fixture_legs(
                fixture_id=fixture_id,
                fixture=fixture,
                team_features=team_features,
                match_model=match_model,
                player_model=player_model,
                player_calibrator=player_calibrator,
            )
            all_legs.extend(legs)
            legs_meta.update(meta)

        logger.info(f"Generated {len(all_legs)} candidate legs.")

        # 5. Rank legs
        value_legs = self.leg_ranker.get_value_legs(all_legs, top_n=20)
        safe_legs = self.leg_ranker.get_safe_legs(all_legs, top_n=20)

        # 6. Build multis
        value_multis = self.multi_builder.build(value_legs, mode="value", max_results=15)
        safe_multis = self.multi_builder.build(safe_legs, mode="safe", max_results=15)

        same_game_multis: List[Multi] = []
        for fixture_id in upcoming["fixture_id"].unique():
            sgm = self.multi_builder.build_same_game_multis(
                all_legs, fixture_id=int(fixture_id), max_results=5
            )
            same_game_multis.extend(sgm)

        # Build a leg dict lookup for multi enrichment
        leg_dict_lookup = {l.leg_id: self._leg_to_dict(l, legs_meta) for l in all_legs}

        elapsed = (datetime.utcnow() - start_time).total_seconds()
        logger.info(f"[{self.run_id}] Pipeline complete in {elapsed:.2f}s")

        # Data source info
        data_source = self._get_data_source_label()

        return {
            "run_id": self.run_id,
            "timestamp": datetime.utcnow().isoformat(),
            "data_source": data_source,
            "n_fixtures": len(upcoming),
            "n_candidate_legs": len(all_legs),
            "elapsed_seconds": round(elapsed, 2),
            "value_legs": [self._leg_to_dict(l, legs_meta) for l in value_legs],
            "safe_legs": [self._leg_to_dict(l, legs_meta) for l in safe_legs],
            "value_multis": [self._multi_to_dict(m, leg_dict_lookup) for m in value_multis],
            "safe_multis": [self._multi_to_dict(m, leg_dict_lookup) for m in safe_multis],
            "same_game_multis": [self._multi_to_dict(m, leg_dict_lookup) for m in same_game_multis],
        }

    # ------------------------------------------------------------------
    # Name lookups
    # ------------------------------------------------------------------

    def _build_lookups(self) -> None:
        """Build team/player/fixture name lookup maps."""
        teams_df = self.loader.load_teams_df()
        players_df = self.loader.load_players_df()
        fixtures_df = self.loader.load_fixtures_df()

        self._team_names = {}
        for _, row in teams_df.iterrows():
            try:
                self._team_names[int(row["team_id"])] = str(row["name"])
            except (ValueError, KeyError):
                pass

        # Also populate team names from fixtures_df home/away name columns
        # (SportradarLoader embeds names directly; avoids needing a separate teams endpoint)
        if "home_team_name" in fixtures_df.columns and "home_team_id" in fixtures_df.columns:
            for _, row in fixtures_df.iterrows():
                try:
                    hid = int(row["home_team_id"])
                    aid = int(row["away_team_id"])
                    if hid and hid not in self._team_names:
                        self._team_names[hid] = str(row["home_team_name"])
                    if aid and aid not in self._team_names:
                        self._team_names[aid] = str(row["away_team_name"])
                except (ValueError, KeyError, TypeError):
                    pass

        self._player_names = {}
        for _, row in players_df.iterrows():
            try:
                self._player_names[int(row["player_id"])] = str(row["name"])
            except (ValueError, KeyError):
                pass

        self._fixture_info = {}
        for _, row in fixtures_df.iterrows():
            try:
                fid = int(row["fixture_id"])
                date_str = str(row.get("date", ""))
                time_str = str(row.get("time", ""))
                start_time = f"{date_str} {time_str}".strip() if date_str else ""
                self._fixture_info[fid] = {
                    "home_team_id": int(row["home_team_id"]),
                    "away_team_id": int(row["away_team_id"]),
                    "start_time": start_time,
                    "season": int(row.get("season", 0)),
                    "round": int(row.get("round", 0)),
                }
            except (ValueError, KeyError):
                pass

        logger.debug(
            f"Lookups built: {len(self._team_names)} teams, "
            f"{len(self._player_names)} players, "
            f"{len(self._fixture_info)} fixtures"
        )

    def _team_name(self, team_id: Optional[int]) -> str:
        if team_id is None:
            return "Unknown"
        return self._team_names.get(team_id, f"Team {team_id}")

    def _player_name(self, player_id: Optional[int]) -> str:
        if player_id is None:
            return ""
        return self._player_names.get(player_id, f"Player {player_id}")

    def _match_label(self, fixture_id: int) -> str:
        info = self._fixture_info.get(fixture_id, {})
        home = self._team_name(info.get("home_team_id"))
        away = self._team_name(info.get("away_team_id"))
        return f"{home} vs {away}"

    def _start_time(self, fixture_id: int) -> str:
        info = self._fixture_info.get(fixture_id, {})
        return info.get("start_time", "")

    # ------------------------------------------------------------------
    # Leg generation
    # ------------------------------------------------------------------

    def _generate_fixture_legs(
        self,
        fixture_id: int,
        fixture: pd.Series,
        team_features: pd.DataFrame,
        match_model,
        player_model,
        player_calibrator,
    ) -> Tuple[List[Leg], Dict[str, Dict]]:
        """Generate all candidate legs for a single fixture.
        Returns (legs, meta_dict) where meta_dict maps leg_id -> display metadata.
        """
        legs: List[Leg] = []
        meta: Dict[str, Dict] = {}

        h2h_legs, h2h_meta = self._generate_h2h_legs(
            fixture_id, fixture, team_features, match_model
        )
        legs.extend(h2h_legs)
        meta.update(h2h_meta)

        if player_model is not None:
            p_legs, p_meta = self._generate_player_legs(
                fixture_id, fixture, player_model, player_calibrator
            )
            legs.extend(p_legs)
            meta.update(p_meta)

        return legs, meta

    def _generate_h2h_legs(
        self, fixture_id: int, fixture: pd.Series,
        team_features: pd.DataFrame, match_model
    ) -> Tuple[List[Leg], Dict[str, Dict]]:
        """Generate head-to-head win/loss legs."""
        legs = []
        meta = {}

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
            elo_prob = float(fixture.get("elo_win_prob_home", 0.5)) if "elo_win_prob_home" in fixture else 0.5
            cal_home_prob = elo_prob
            raw_home_prob = elo_prob

        cal_away_prob = 1.0 - cal_home_prob

        h2h_odds = self._get_best_odds(fixture_id, "head_to_head")
        home_id = int(fixture["home_team_id"])
        away_id = int(fixture["away_team_id"])
        home_name = self._team_name(home_id)
        away_name = self._team_name(away_id)
        match_label = f"{home_name} vs {away_name}"
        start_time = self._start_time(fixture_id)

        fixture_info = self._fixture_info.get(fixture_id, {})
        round_no = fixture_info.get("round", "?")
        season = fixture_info.get("season", "?")

        for selection, model_prob, team_id, team_name in [
            ("home_win", cal_home_prob, home_id, home_name),
            ("away_win", cal_away_prob, away_id, away_name),
        ]:
            market_data = h2h_odds.get(selection)
            if market_data is None:
                decimal_odds = round(1.0 / max(model_prob, 0.05), 2)
                vig_adj_prob = model_prob * 0.95
            else:
                decimal_odds = market_data["decimal_odds"]
                vig_adj_prob = market_data["vig_adj_prob"]

            edge = compute_edge(model_prob, vig_adj_prob)
            ev = compute_ev(model_prob, decimal_odds)
            confidence = compute_confidence_score(model_prob, vig_adj_prob, edge, n_historical_games=30)
            conf_label = _confidence_label(confidence)

            selection_label = _make_selection_label(selection, team_name=team_name)
            opponent = away_name if selection == "home_win" else home_name
            venue_note = "home advantage" if selection == "home_win" else "away challenge"
            explanation = (
                f"{team_name} to win vs {opponent} (Rnd {round_no}, {season}). "
                f"Model: {model_prob:.1%} | Market: {vig_adj_prob:.1%} | Edge: {edge:+.1%}. "
                f"Confidence is {conf_label.lower()} — factoring in {venue_note} and recent form."
            )

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
            meta[leg_id] = {
                "selection_label": selection_label,
                "match_label": match_label,
                "home_team_name": home_name,
                "away_team_name": away_name,
                "team_name": team_name,
                "player_name": "",
                "start_time": start_time,
                "market_label": "Head-to-Head",
                "vig_adj_prob": vig_adj_prob,
                "edge": edge,
                "confidence_label": conf_label,
            }

        return legs, meta

    def _generate_player_legs(
        self, fixture_id: int, fixture: pd.Series, player_model, player_calibrator
    ) -> Tuple[List[Leg], Dict[str, Dict]]:
        """Generate player disposals over/under legs."""
        legs = []
        meta = {}
        odds_df = self.loader.load_odds_df()
        if odds_df.empty:
            return legs, meta

        player_odds = odds_df[
            (odds_df["fixture_id"] == fixture_id) &
            (odds_df["market_type"] == "player_disposals")
        ]
        if player_odds.empty:
            return legs, meta

        home_id = int(fixture["home_team_id"])
        away_id = int(fixture["away_team_id"])
        home_name = self._team_name(home_id)
        away_name = self._team_name(away_id)
        match_label = f"{home_name} vs {away_name}"
        start_time = self._start_time(fixture_id)
        fixture_info = self._fixture_info.get(fixture_id, {})
        round_no = fixture_info.get("round", "?")
        season = fixture_info.get("season", "?")

        processed = {}
        for _, row in player_odds.iterrows():
            sel = str(row["selection"])
            parts = sel.split("_")
            if len(parts) < 4:
                continue
            try:
                pid = int(parts[1])
                direction = parts[2]
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

        for (pid, line), directions in processed.items():
            player_info = players[players["player_id"] == pid]
            team_id = int(player_info["team_id"].iloc[0]) if not player_info.empty else None
            player_name = self._player_name(pid)
            team_name = self._team_name(team_id)

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
                roll_mean = feat.get("roll_mean_5", 20.0)
                from scipy.stats import norm
                raw_over_prob = float(1.0 - norm.cdf(line, loc=roll_mean, scale=5.0))
                cal_over_prob = raw_over_prob

            cal_over_prob = float(np.clip(cal_over_prob, 0.05, 0.95))
            cal_under_prob = 1.0 - cal_over_prob
            roll_mean = feat.get("roll_mean_5", 20.0)
            n_games = feat.get("n_games", 0)
            form_trend = feat.get("form_trend", 0.0)
            trend_note = "trending up" if form_trend > 1 else ("trending down" if form_trend < -1 else "stable form")

            for direction, model_prob in [("over", cal_over_prob), ("under", cal_under_prob)]:
                if direction not in directions:
                    continue
                d = directions[direction]
                decimal_odds = d["decimal_odds"]
                selection = d["selection"]

                opp_sel = directions.get("under" if direction == "over" else "over", {})
                if opp_sel:
                    raw_imp = 1.0 / decimal_odds
                    raw_opp_imp = 1.0 / opp_sel.get("decimal_odds", decimal_odds)
                    total_imp = raw_imp + raw_opp_imp
                    vig_adj_prob = raw_imp / total_imp if total_imp > 0 else 0.5
                else:
                    vig_adj_prob = 1.0 / decimal_odds

                edge = compute_edge(model_prob, vig_adj_prob)
                ev = compute_ev(model_prob, decimal_odds)
                confidence = compute_confidence_score(model_prob, vig_adj_prob, edge, n_historical_games=n_games)
                conf_label = _confidence_label(confidence)

                selection_label = _make_selection_label(selection, player_name=player_name)
                avg_note = f"5-game avg: {roll_mean:.1f}"
                vs_line = "above" if direction == "over" else "below"
                explanation = (
                    f"{player_name} ({team_name}) {direction} {line} disposals in {match_label} (Rnd {round_no}). "
                    f"{avg_note} — consistently {vs_line} this line. "
                    f"Model: {model_prob:.1%} | Market: {vig_adj_prob:.1%} | Edge: {edge:+.1%}. "
                    f"Based on {int(n_games)} games, {trend_note}. Confidence: {conf_label}."
                )

                leg_id = "L_" + hashlib.md5(f"{fixture_id}_{selection}".encode()).hexdigest()[:8]

                legs.append(Leg(
                    leg_id=leg_id,
                    fixture_id=fixture_id,
                    player_id=pid,
                    team_id=team_id,
                    market_type="player_disposals",
                    selection=selection,
                    decimal_odds=decimal_odds,
                    calibrated_probability=model_prob,
                    ev=ev,
                    confidence_score=confidence,
                    explanation=explanation,
                ))
                meta[leg_id] = {
                    "selection_label": selection_label,
                    "match_label": match_label,
                    "home_team_name": home_name,
                    "away_team_name": away_name,
                    "team_name": team_name,
                    "player_name": player_name,
                    "start_time": start_time,
                    "market_label": "Player Disposals",
                    "vig_adj_prob": vig_adj_prob,
                    "edge": edge,
                    "confidence_label": conf_label,
                    "line": line,
                    "direction": direction,
                    "roll_mean_5": roll_mean,
                }

        return legs, meta

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_best_odds(self, fixture_id: int, market_type: str) -> Dict:
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
        model = self.registry.load("match_win_ensemble")
        if model is None:
            logger.info("No trained match model found, training now...")
            from app.services.training import TrainingService
            svc = TrainingService(self.loader)
            svc.run_match_model_training()
            model = self.registry.load("match_win_ensemble")
        return model

    def _get_player_model(self):
        model = self.registry.load("player_disposals_model")
        if model is None:
            logger.info("No trained player model found, training now...")
            from app.services.training import TrainingService
            svc = TrainingService(self.loader)
            svc.run_player_disposals_training()
            model = self.registry.load("player_disposals_model")
        return model

    def _get_player_calibrator(self):
        cal = self.registry.load("player_disposals_calibrator")
        if cal is None:
            cal = IsotonicCalibrator()
        return cal

    def _get_data_source_label(self) -> str:
        """Return a human-readable data source description."""
        mode = settings.effective_data_mode
        if mode == "live":
            return "Sportradar API (Live)"
        elif mode == "cache":
            return "Sportradar API (Cached)"
        else:
            return "Demo Data"

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def _leg_to_dict(self, leg: Leg, meta: Dict[str, Dict]) -> Dict:
        """Serialise a Leg to a fully human-readable dict."""
        m = meta.get(leg.leg_id, {})
        return {
            # IDs (kept for joins but never shown raw in UI)
            "leg_id": leg.leg_id,
            "fixture_id": leg.fixture_id,
            "player_id": leg.player_id,
            "team_id": leg.team_id,

            # Human-readable display fields
            "match_label": m.get("match_label") or self._match_label(leg.fixture_id),
            "home_team_name": m.get("home_team_name", ""),
            "away_team_name": m.get("away_team_name", ""),
            "team_name": m.get("team_name") or self._team_name(leg.team_id),
            "player_name": m.get("player_name", ""),
            "start_time": m.get("start_time") or self._start_time(leg.fixture_id),
            "market_type": leg.market_type,
            "market_label": m.get("market_label", leg.market_type.replace("_", " ").title()),
            "selection": leg.selection,
            "selection_label": m.get("selection_label") or _make_selection_label(
                leg.selection,
                team_name=m.get("team_name", ""),
                player_name=m.get("player_name", ""),
            ),

            # Pricing
            "decimal_odds": round(leg.decimal_odds, 2),
            "model_probability": round(leg.calibrated_probability, 4),
            "market_probability": round(m.get("vig_adj_prob", 1 / leg.decimal_odds), 4),
            "edge": round(m.get("edge", 0.0), 4),
            "ev": round(leg.ev, 4),
            "confidence_score": round(leg.confidence_score, 1),
            "confidence_label": m.get("confidence_label", _confidence_label(leg.confidence_score)),

            # Explanation
            "explanation": leg.explanation,
        }

    def _multi_to_dict(self, multi: Multi, leg_dict_lookup: Dict[str, Dict]) -> Dict:
        """Serialise a Multi with enriched leg details."""
        leg_summaries = []
        for lid in (multi.leg_ids or []):
            ld = leg_dict_lookup.get(lid)
            if ld:
                leg_summaries.append({
                    "leg_id": lid,
                    "match_label": ld.get("match_label", ""),
                    "selection_label": ld.get("selection_label", ""),
                    "decimal_odds": ld.get("decimal_odds"),
                    "model_probability": ld.get("model_probability"),
                    "confidence_label": ld.get("confidence_label", ""),
                })

        return {
            "multi_id": multi.multi_id,
            "n_legs": multi.n_legs,
            "multi_type": multi.multi_type,
            "combined_odds": round(multi.combined_odds, 2),
            "raw_probability": round(getattr(multi, "raw_probability", multi.adjusted_probability), 4),
            "adjusted_probability": round(multi.adjusted_probability, 4),
            "ev": round(multi.ev, 4),
            "correlation_score": round(multi.correlation_score, 3),
            "correlation_label": multi.correlation_label,
            "risk_score": round(multi.risk_score, 1),
            "explanation": multi.explanation,
            "leg_ids": multi.leg_ids,
            "legs": leg_summaries,
        }
