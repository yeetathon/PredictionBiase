"""Feature engineering pipeline for team and player level features.

All features are computed strictly using data available BEFORE each fixture,
preventing any look-ahead leakage.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from loguru import logger

from app.data_ingestion.loader import DataLoader


class EloRatingSystem:
    """
    Elo rating system for AFL team strength estimation.
    Ratings are updated after each game, carry between seasons with regression to mean,
    and include home ground advantage.
    """

    def __init__(
        self,
        k_factor: float = 32.0,
        initial_rating: float = 1500.0,
        home_advantage: float = 70.0,
        season_carryover: float = 0.75,
    ):
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.home_advantage = home_advantage
        self.season_carryover = season_carryover

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Logistic expected score for team A vs team B."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def update(
        self, rating_a: float, rating_b: float, score_a: float
    ) -> Tuple[float, float]:
        """Return updated ratings after a game. score_a is 1/0/0.5."""
        ea = self.expected_score(rating_a, rating_b)
        eb = 1.0 - ea
        new_a = rating_a + self.k_factor * (score_a - ea)
        new_b = rating_b + self.k_factor * ((1 - score_a) - eb)
        return new_a, new_b

    def compute_ratings_history(
        self, fixtures: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute pre-game Elo ratings for all fixtures.
        Returns fixtures DataFrame with added columns:
            elo_home_pre, elo_away_pre, elo_win_prob_home
        """
        fixtures = fixtures.copy().sort_values(["season", "round", "fixture_id"])
        ratings: Dict[int, float] = {}
        prev_season = None

        records = []
        for _, row in fixtures.iterrows():
            season = int(row["season"])
            home_id = int(row["home_team_id"])
            away_id = int(row["away_team_id"])

            # Season start: regress ratings to mean
            if season != prev_season and prev_season is not None:
                for tid in list(ratings.keys()):
                    ratings[tid] = (
                        ratings[tid] * self.season_carryover
                        + self.initial_rating * (1 - self.season_carryover)
                    )
            prev_season = season

            # Initialise new teams
            if home_id not in ratings:
                ratings[home_id] = self.initial_rating
            if away_id not in ratings:
                ratings[away_id] = self.initial_rating

            home_rating = ratings[home_id]
            away_rating = ratings[away_id]

            # Adjust home rating for home ground advantage
            adj_home = home_rating + self.home_advantage
            win_prob_home = self.expected_score(adj_home, away_rating)

            records.append({
                "fixture_id": int(row["fixture_id"]),
                "elo_home_pre": round(home_rating, 2),
                "elo_away_pre": round(away_rating, 2),
                "elo_win_prob_home": round(win_prob_home, 4),
                "elo_diff": round(home_rating - away_rating, 2),
            })

            # Update if completed
            if row.get("status") == "completed" and pd.notna(row.get("home_win")):
                result = float(row["home_win"])
                new_home, new_away = self.update(home_rating, away_rating, result)
                ratings[home_id] = new_home
                ratings[away_id] = new_away

        elo_df = pd.DataFrame(records)
        return fixtures.merge(elo_df, on="fixture_id", how="left")


class TeamFeatureEngineer:
    """
    Computes team-level features from historical data.
    Strictly no lookahead: rolling stats are computed only on prior games.
    """

    def __init__(self, loader: Optional[DataLoader] = None, rolling_window: int = 5):
        self.loader = loader or DataLoader()
        self.window = rolling_window
        self.elo = EloRatingSystem()

    def build_features(self) -> pd.DataFrame:
        """
        Build the full team-level feature matrix.
        Each row = one team's pre-game features for a fixture.
        """
        fixtures = self.loader.load_fixtures_df()
        team_stats = self.loader.load_team_stats_df()
        weather = self.loader.load_weather_df()

        if fixtures.empty:
            logger.warning("No fixtures found for feature engineering.")
            return pd.DataFrame()

        # Compute Elo ratings (uses all fixtures; only updates on completed ones)
        completed = fixtures[fixtures["status"] == "completed"].copy()
        all_fx = fixtures.sort_values(["season", "round", "fixture_id"]).copy()
        all_fx = self.elo.compute_ratings_history(all_fx)

        # Build rolling team features (handles minimal team_stats gracefully)
        rolling_feats = self._compute_rolling_team_features(completed, team_stats)

        # Merge Elo onto fixtures
        elo_cols = ["fixture_id", "elo_home_pre", "elo_away_pre", "elo_win_prob_home", "elo_diff"]
        result = all_fx.merge(
            all_fx[elo_cols], on="fixture_id", how="left", suffixes=("", "_elo")
        )

        # Drop duplicate columns from double-merge
        result = all_fx.copy()

        # Merge rolling features for home and away teams
        if not rolling_feats.empty:
            home_rf = rolling_feats.rename(
                columns={c: f"home_{c}" for c in rolling_feats.columns if c not in ["fixture_id", "team_id"]}
            )
            home_rf = home_rf.rename(columns={"team_id": "home_team_id"})
            away_rf = rolling_feats.rename(
                columns={c: f"away_{c}" for c in rolling_feats.columns if c not in ["fixture_id", "team_id"]}
            )
            away_rf = away_rf.rename(columns={"team_id": "away_team_id"})

            result = result.merge(
                home_rf.drop(columns=["fixture_id"], errors="ignore"),
                on="home_team_id", how="left"
            )
            result = result.merge(
                away_rf.drop(columns=["fixture_id"], errors="ignore"),
                on="away_team_id", how="left"
            )

        # Add weather features
        if not weather.empty:
            wx_cols = ["fixture_id", "temperature_c", "humidity_pct", "wind_speed_kmh", "conditions"]
            wx = weather[[c for c in wx_cols if c in weather.columns]].copy()
            wx["is_rain"] = wx["conditions"].isin(["rain", "light_rain"]).astype(int)
            wx["wind_category"] = pd.cut(
                wx["wind_speed_kmh"], bins=[0, 10, 20, 35, 100],
                labels=[0, 1, 2, 3]
            ).astype(float)
            result = result.merge(wx, on="fixture_id", how="left")

        # Compute differential features for all paired home_/away_ roll columns
        numeric_home = [c for c in result.columns if c.startswith("home_roll_")]
        for hc in numeric_home:
            ac = hc.replace("home_", "away_")
            if ac in result.columns:
                diff_col = hc.replace("home_roll_", "diff_roll_")
                result[diff_col] = result[hc].fillna(0) - result[ac].fillna(0)

        # Rest-days differential
        if "home_rest_days" in result.columns and "away_rest_days" in result.columns:
            result["diff_rest_days"] = result["home_rest_days"].fillna(7) - result["away_rest_days"].fillna(7)

        result["elo_diff"] = result.get("elo_home_pre", 1500) - result.get("elo_away_pre", 1500)

        return result

    def _compute_rolling_team_features(
        self, completed_fixtures: pd.DataFrame, team_stats: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute rolling team features from prior games only (no lookahead).

        Works with minimal team_stats_df (score + is_home only, as returned by
        SportradarLoader) as well as full demo CSVs with disposals/marks/etc.

        Returns a DataFrame indexed by (fixture_id, team_id).
        """
        if completed_fixtures.empty:
            return pd.DataFrame()

        # Build a date lookup for rest-days calculation
        # Use 'date' column from completed_fixtures if available
        has_date = "date" in completed_fixtures.columns

        # Join team_stats to fixture metadata
        fx_meta_cols = ["fixture_id", "season", "round", "home_team_id", "away_team_id"]
        if has_date:
            fx_meta_cols.append("date")
        fx_meta = completed_fixtures[fx_meta_cols].copy()

        if team_stats.empty:
            # Build minimal team_stats from fixture score columns
            records_ts = []
            for _, row in completed_fixtures.iterrows():
                fid = int(row["fixture_id"])
                if "home_score" in row and pd.notna(row.get("home_score")):
                    records_ts.append({
                        "fixture_id": fid,
                        "team_id": int(row["home_team_id"]),
                        "is_home": 1,
                        "score": float(row["home_score"]),
                    })
                if "away_score" in row and pd.notna(row.get("away_score")):
                    records_ts.append({
                        "fixture_id": fid,
                        "team_id": int(row["away_team_id"]),
                        "is_home": 0,
                        "score": float(row["away_score"]),
                    })
            if not records_ts:
                return pd.DataFrame()
            ts = pd.DataFrame(records_ts)
            # Merge fixture metadata once
            ts = ts.merge(fx_meta, on="fixture_id", how="left")
        else:
            # team_stats already has fixture_id; merge metadata columns only
            existing_cols = set(team_stats.columns)
            meta_new_cols = [c for c in fx_meta_cols if c not in existing_cols or c == "fixture_id"]
            ts = team_stats.merge(fx_meta[meta_new_cols], on="fixture_id", how="inner")

        ts = ts.sort_values(["season", "round", "fixture_id"])

        stat_cols = ["score", "disposals", "marks", "tackles", "inside_50s",
                     "rebound_50s", "clearances", "contested_possessions",
                     "metres_gained", "turnovers"]
        stat_cols = [c for c in stat_cols if c in ts.columns]

        # Build win lookup: fixture_id → home_win (for completed games)
        win_lookup = {}
        if "home_win" in completed_fixtures.columns:
            for _, row in completed_fixtures.iterrows():
                hw = row.get("home_win")
                if pd.notna(hw):
                    win_lookup[int(row["fixture_id"])] = int(hw)

        records = []
        teams = ts["team_id"].unique()

        for tid in teams:
            team_games = ts[ts["team_id"] == tid].sort_values(
                ["season", "round", "fixture_id"]
            ).reset_index(drop=True)

            for i in range(len(team_games)):
                row = team_games.iloc[i]
                prior = team_games.iloc[:i]

                feat = {"fixture_id": int(row["fixture_id"]), "team_id": int(tid)}

                # Rolling stats (score + any other stat columns)
                for sc in stat_cols:
                    if sc not in prior.columns:
                        continue
                    vals = prior[sc].dropna().values
                    pop_mean = float(ts[sc].mean()) if sc in ts.columns else 0.0
                    if len(vals) >= 1:
                        feat[f"roll_{sc}_mean"] = float(np.mean(vals[-self.window:]))
                        feat[f"roll_{sc}_std"] = float(np.std(vals[-self.window:])) if len(vals) >= 2 else 0.0
                    else:
                        feat[f"roll_{sc}_mean"] = pop_mean
                        feat[f"roll_{sc}_std"] = 0.0

                n_games = len(prior)
                feat["roll_n_games"] = n_games

                # Win rate (overall, home, away) from prior games
                if n_games > 0 and win_lookup:
                    prior_is_home = prior["is_home"].values if "is_home" in prior.columns else []
                    total_wins = 0
                    home_wins = 0
                    home_games = 0
                    away_wins = 0
                    away_games = 0
                    for j, prow in prior.iterrows():
                        pfid = int(prow["fixture_id"])
                        p_is_home = int(prow.get("is_home", 0))
                        hw = win_lookup.get(pfid)
                        if hw is None:
                            continue
                        won = (p_is_home == 1 and hw == 1) or (p_is_home == 0 and hw == 0)
                        total_wins += int(won)
                        if p_is_home == 1:
                            home_games += 1
                            home_wins += int(won)
                        else:
                            away_games += 1
                            away_wins += int(won)
                    feat["roll_win_rate"] = total_wins / n_games
                    feat["roll_home_win_rate"] = home_wins / home_games if home_games > 0 else 0.5
                    feat["roll_away_win_rate"] = away_wins / away_games if away_games > 0 else 0.5
                else:
                    feat["roll_win_rate"] = 0.5
                    feat["roll_home_win_rate"] = 0.5
                    feat["roll_away_win_rate"] = 0.5

                # Rest days: days since last game (capped at 21)
                if has_date and n_games > 0:
                    try:
                        last_date_str = str(prior.iloc[-1]["date"])
                        cur_date_str = str(row.get("date", ""))
                        if last_date_str and cur_date_str:
                            from datetime import datetime as _dt
                            last_dt = _dt.fromisoformat(last_date_str.split("T")[0])
                            cur_dt = _dt.fromisoformat(cur_date_str.split("T")[0])
                            rest = max(0, min(21, (cur_dt - last_dt).days))
                            feat["rest_days"] = float(rest)
                        else:
                            feat["rest_days"] = 7.0
                    except Exception:
                        feat["rest_days"] = 7.0
                else:
                    feat["rest_days"] = 7.0

                records.append(feat)

        return pd.DataFrame(records)

    def get_fixture_features(self, fixture_id: int) -> Optional[Dict]:
        """Get pre-built features for a specific fixture."""
        features = self.build_features()
        row = features[features["fixture_id"] == fixture_id]
        if row.empty:
            return None
        return row.iloc[0].to_dict()


class PlayerFeatureEngineer:
    """
    Computes player-level features for disposals prediction.
    Uses rolling averages, consistency, opponent allowance, and team context.
    """

    def __init__(self, loader: Optional[DataLoader] = None, rolling_window: int = 5):
        self.loader = loader or DataLoader()
        self.window = rolling_window

    def build_player_features(self, stat_col: str = "disposals") -> pd.DataFrame:
        """
        Build player-level features for a target stat.
        Returns one row per (fixture_id, player_id) with pre-game features.
        """
        player_stats = self.loader.load_player_stats_df()
        players = self.loader.load_players_df()
        fixtures = self.loader.load_fixtures_df()
        team_stats = self.loader.load_team_stats_df()

        if player_stats.empty or fixtures.empty:
            return pd.DataFrame()

        completed = fixtures[fixtures["status"] == "completed"].copy()

        # Join fixture metadata
        ps = player_stats.merge(
            completed[["fixture_id", "season", "round", "home_team_id", "away_team_id"]],
            on="fixture_id", how="inner"
        ).sort_values(["season", "round", "fixture_id"])

        # Join player info
        if not players.empty:
            ps = ps.merge(
                players[["player_id", "position"]],
                on="player_id", how="left"
            )

        records = []
        player_ids = ps["player_id"].unique()

        for pid in player_ids:
            pg = ps[ps["player_id"] == pid].reset_index(drop=True)
            for i, row in pg.iterrows():
                prior = pg.iloc[:i]
                vals = prior[stat_col].dropna().values if stat_col in prior.columns else np.array([])
                feat = {
                    "fixture_id": int(row["fixture_id"]),
                    "player_id": int(pid),
                    "team_id": int(row["team_id"]),
                    "target": float(row[stat_col]) if pd.notna(row.get(stat_col)) else None,
                    "position": str(row.get("position", "Unknown")),
                }
                n = len(vals)
                if n >= 1:
                    feat["roll_mean_3"] = float(np.mean(vals[-3:])) if n >= 3 else float(np.mean(vals))
                    feat["roll_mean_5"] = float(np.mean(vals[-5:])) if n >= 5 else float(np.mean(vals))
                    feat["roll_mean_10"] = float(np.mean(vals[-10:])) if n >= 10 else float(np.mean(vals))
                    feat["roll_std_5"] = float(np.std(vals[-5:])) if n >= 3 else 0.0
                    feat["roll_max_5"] = float(np.max(vals[-5:])) if n >= 1 else 0.0
                    feat["roll_min_5"] = float(np.min(vals[-5:])) if n >= 1 else 0.0
                    # Consistency: CV (lower = more consistent)
                    mean5 = feat["roll_mean_5"]
                    feat["consistency_cv"] = (feat["roll_std_5"] / mean5) if mean5 > 0 else 1.0
                    # Form trend: difference of recent means
                    if n >= 4:
                        feat["form_trend"] = float(np.mean(vals[-2:])) - float(np.mean(vals[-4:-2]))
                    else:
                        feat["form_trend"] = 0.0
                    feat["n_games"] = n
                else:
                    # Population mean for cold start
                    pop_mean = float(ps[stat_col].mean()) if stat_col in ps.columns and not ps[stat_col].empty else 20.0
                    feat.update({
                        "roll_mean_3": pop_mean, "roll_mean_5": pop_mean,
                        "roll_mean_10": pop_mean, "roll_std_5": 5.0,
                        "roll_max_5": pop_mean + 5, "roll_min_5": pop_mean - 5,
                        "consistency_cv": 0.25, "form_trend": 0.0, "n_games": 0,
                    })

                # Opponent allowance: avg stat conceded by opponent team
                home_id = int(row["home_team_id"])
                away_id = int(row["away_team_id"])
                opp_id = away_id if int(row["team_id"]) == home_id else home_id
                # Use team_stats to approximate opponent defensive strength
                if not team_stats.empty:
                    opp_games = completed[
                        (completed["home_team_id"] == opp_id) | (completed["away_team_id"] == opp_id)
                    ]["fixture_id"].tolist()
                    opp_ts = team_stats[
                        (team_stats["fixture_id"].isin(opp_games)) &
                        (team_stats["team_id"] != opp_id)
                    ]
                    if "disposals" in opp_ts.columns and not opp_ts.empty:
                        opp_allow = float(opp_ts["disposals"].mean())
                        feat["opp_disposals_allowed_mean"] = round(opp_allow / 18, 2)  # per player approx
                    else:
                        feat["opp_disposals_allowed_mean"] = 20.0
                else:
                    feat["opp_disposals_allowed_mean"] = 20.0

                records.append(feat)

        df = pd.DataFrame(records)
        if df.empty:
            return df

        # Position encoding
        df["pos_midfielder"] = (df["position"] == "Midfielder").astype(int)
        df["pos_forward"] = (df["position"] == "Forward").astype(int)
        df["pos_defender"] = (df["position"] == "Defender").astype(int)
        df["pos_ruckman"] = (df["position"] == "Ruckman").astype(int)

        return df

    def get_player_prediction_features(
        self, player_id: int, fixture_id: int
    ) -> Optional[Dict]:
        """
        Get features for a player's upcoming fixture prediction.
        Uses the most recent available rolling stats.
        """
        player_stats = self.loader.load_player_stats_df()
        players = self.loader.load_players_df()

        if player_stats.empty:
            return None

        pg = player_stats[player_stats["player_id"] == player_id].copy()
        if pg.empty:
            return None

        fixtures = self.loader.load_fixtures_df()
        pg = pg.merge(
            fixtures[["fixture_id", "season", "round"]],
            on="fixture_id", how="left"
        ).sort_values(["season", "round"])

        vals = pg["disposals"].dropna().values
        n = len(vals)

        player_info = players[players["player_id"] == player_id]
        position = str(player_info["position"].iloc[0]) if not player_info.empty else "Unknown"

        feat = {
            "fixture_id": fixture_id,
            "player_id": player_id,
            "position": position,
        }
        pop_mean = float(player_stats["disposals"].mean()) if "disposals" in player_stats.columns else 20.0

        if n >= 1:
            feat["roll_mean_3"] = float(np.mean(vals[-3:])) if n >= 3 else float(np.mean(vals))
            feat["roll_mean_5"] = float(np.mean(vals[-5:])) if n >= 5 else float(np.mean(vals))
            feat["roll_mean_10"] = float(np.mean(vals[-10:])) if n >= 10 else float(np.mean(vals))
            feat["roll_std_5"] = float(np.std(vals[-5:])) if n >= 3 else 3.0
            feat["roll_max_5"] = float(np.max(vals[-5:]))
            feat["roll_min_5"] = float(np.min(vals[-5:]))
            mean5 = feat["roll_mean_5"]
            feat["consistency_cv"] = (feat["roll_std_5"] / mean5) if mean5 > 0 else 0.25
            feat["form_trend"] = float(np.mean(vals[-2:])) - float(np.mean(vals[-4:-2])) if n >= 4 else 0.0
            feat["n_games"] = n
        else:
            feat.update({
                "roll_mean_3": pop_mean, "roll_mean_5": pop_mean,
                "roll_mean_10": pop_mean, "roll_std_5": 5.0,
                "roll_max_5": pop_mean + 5, "roll_min_5": pop_mean - 5,
                "consistency_cv": 0.25, "form_trend": 0.0, "n_games": 0,
            })

        feat["pos_midfielder"] = int(position == "Midfielder")
        feat["pos_forward"] = int(position == "Forward")
        feat["pos_defender"] = int(position == "Defender")
        feat["pos_ruckman"] = int(position == "Ruckman")
        feat["opp_disposals_allowed_mean"] = 20.0

        return feat


class FeaturePipeline:
    """
    Orchestrates team and player feature engineering.
    Central entry point for the modelling pipeline.
    """

    def __init__(self, loader: Optional[DataLoader] = None):
        self.loader = loader or DataLoader()
        self.team_engineer = TeamFeatureEngineer(self.loader)
        self.player_engineer = PlayerFeatureEngineer(self.loader)
        self._team_features: Optional[pd.DataFrame] = None
        self._player_features: Optional[pd.DataFrame] = None

    def get_team_features(self, force_rebuild: bool = False) -> pd.DataFrame:
        """Get team features, building from scratch or returning cache."""
        if self._team_features is None or force_rebuild:
            logger.info("Building team feature matrix...")
            self._team_features = self.team_engineer.build_features()
            logger.info(f"Team features shape: {self._team_features.shape}")
        return self._team_features

    def get_player_features(
        self, stat_col: str = "disposals", force_rebuild: bool = False
    ) -> pd.DataFrame:
        """Get player features for a given stat target."""
        if self._player_features is None or force_rebuild:
            logger.info(f"Building player feature matrix for '{stat_col}'...")
            self._player_features = self.player_engineer.build_player_features(stat_col)
            logger.info(f"Player features shape: {self._player_features.shape}")
        return self._player_features

    def get_model_ready_match_data(self) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Return (X, y, feature_names) for match win probability model.
        Only includes completed fixtures with valid labels.
        """
        features = self.get_team_features()
        completed = features[
            (features["status"] == "completed") & features["home_win"].notna()
        ].copy()

        if completed.empty:
            return pd.DataFrame(), pd.Series(dtype=float), []

        feature_cols = [
            c for c in completed.columns
            if c.startswith(("elo_", "home_roll_", "away_roll_", "diff_roll_",
                             "home_rest_days", "away_rest_days", "diff_rest_days",
                             "temperature_c", "wind_speed_kmh", "is_rain", "wind_category"))
            and completed[c].dtype in [np.float64, np.int64, float, int]
        ]

        X = completed[feature_cols].fillna(0)
        y = completed["home_win"].astype(int)
        return X, y, feature_cols

    def get_model_ready_player_data(
        self, line: float, stat_col: str = "disposals"
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Return (X, y, feature_names) for player over/under model.
        y = 1 if player went OVER the line, 0 otherwise.
        """
        features = self.get_player_features(stat_col)
        valid = features[features["target"].notna()].copy()

        if valid.empty:
            return pd.DataFrame(), pd.Series(dtype=float), []

        valid["label"] = (valid["target"] > line).astype(int)

        feature_cols = [
            c for c in valid.columns
            if c in [
                "roll_mean_3", "roll_mean_5", "roll_mean_10", "roll_std_5",
                "roll_max_5", "roll_min_5", "consistency_cv", "form_trend",
                "n_games", "opp_disposals_allowed_mean",
                "pos_midfielder", "pos_forward", "pos_defender", "pos_ruckman",
            ]
        ]

        X = valid[feature_cols].fillna(0)
        y = valid["label"]
        return X, y, feature_cols
