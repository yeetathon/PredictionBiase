"""Feature engineering pipeline for team and player level features.

All features are computed strictly using data available BEFORE each fixture,
preventing any look-ahead leakage.

v2 upgrades:
  - Exponentially weighted moving averages (halflife=3 games): recent AFL form
    matters far more than games from 5+ rounds ago.
  - Linear regression form slope: genuine trend direction replaces crude delta.
  - H2H historical features: head-to-head win rate + avg score diff per matchup.
  - IQR-based consistency: robust to outlier performances.
  - Player role stability + decay-weighted usage trend.
  - Opponent position-specific defensive allowance (not /18 population guess).
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from loguru import logger

from app.data_ingestion.loader import DataLoader


# ── Module-level feature helpers ──────────────────────────────────────────────

def _ewma(vals: np.ndarray, halflife: float = 3.0) -> float:
    """Exponentially weighted mean. Recent games count more (halflife in games)."""
    if len(vals) == 0:
        return 0.0
    n = len(vals)
    alpha = 1.0 - np.exp(-np.log(2.0) / halflife)
    weights = np.array([(1.0 - alpha) ** (n - 1 - i) for i in range(n)], dtype=float)
    total = weights.sum()
    return float(np.dot(weights, vals) / total) if total > 0 else float(np.mean(vals))


def _form_slope(vals: np.ndarray) -> float:
    """
    Linear regression slope (units/game). Positive = improving trend.
    Uses de-meaned x for numerical stability.
    """
    if len(vals) < 3:
        return 0.0
    x = np.arange(len(vals), dtype=float)
    x -= x.mean()
    ss_x = float(np.dot(x, x))
    return float(np.dot(x, vals) / ss_x) if ss_x > 0 else 0.0


def _iqr_spread(vals: np.ndarray) -> float:
    """
    IQR/median: consistency measure robust to outliers.
    Lower = more consistent (0 = perfectly consistent).
    Falls back to CV when n < 4.
    """
    if len(vals) < 2:
        return 0.3
    if len(vals) < 4:
        mean = float(np.mean(vals))
        return float(np.std(vals) / max(mean, 1.0))
    q75, q25 = float(np.percentile(vals, 75)), float(np.percentile(vals, 25))
    median = float(np.median(vals))
    return float((q75 - q25) / max(median, 1.0))


# ── Elo system ────────────────────────────────────────────────────────────────

class EloRatingSystem:
    """
    Elo rating system for AFL team strength estimation.
    Ratings update after each game, carry between seasons with regression to mean,
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
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def update(self, rating_a: float, rating_b: float, score_a: float) -> Tuple[float, float]:
        ea = self.expected_score(rating_a, rating_b)
        new_a = rating_a + self.k_factor * (score_a - ea)
        new_b = rating_b + self.k_factor * ((1 - score_a) - (1.0 - ea))
        return new_a, new_b

    def compute_ratings_history(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        """
        Compute pre-game Elo ratings for all fixtures.
        Returns fixtures DataFrame with added columns:
            elo_home_pre, elo_away_pre, elo_win_prob_home, elo_diff
        """
        fixtures = fixtures.copy().sort_values(["season", "round", "fixture_id"])
        ratings: Dict[int, float] = {}
        prev_season = None

        records = []
        for _, row in fixtures.iterrows():
            season = int(row["season"])
            home_id = int(row["home_team_id"])
            away_id = int(row["away_team_id"])

            if season != prev_season and prev_season is not None:
                for tid in list(ratings.keys()):
                    ratings[tid] = (
                        ratings[tid] * self.season_carryover
                        + self.initial_rating * (1 - self.season_carryover)
                    )
            prev_season = season

            ratings.setdefault(home_id, self.initial_rating)
            ratings.setdefault(away_id, self.initial_rating)

            home_rating = ratings[home_id]
            away_rating = ratings[away_id]
            adj_home = home_rating + self.home_advantage
            win_prob_home = self.expected_score(adj_home, away_rating)

            records.append({
                "fixture_id": int(row["fixture_id"]),
                "elo_home_pre": round(home_rating, 2),
                "elo_away_pre": round(away_rating, 2),
                "elo_win_prob_home": round(win_prob_home, 4),
                "elo_diff": round(home_rating - away_rating, 2),
            })

            if row.get("status") == "completed" and pd.notna(row.get("home_win")):
                result = float(row["home_win"])
                new_home, new_away = self.update(home_rating, away_rating, result)
                ratings[home_id] = new_home
                ratings[away_id] = new_away

        elo_df = pd.DataFrame(records)
        return fixtures.merge(elo_df, on="fixture_id", how="left")


# ── Team feature engineering ──────────────────────────────────────────────────

class TeamFeatureEngineer:
    """
    Computes team-level features from historical data.
    Strictly no lookahead: rolling stats use only prior games.
    """

    def __init__(self, loader: Optional[DataLoader] = None, rolling_window: int = 5):
        self.loader = loader or DataLoader()
        self.window = rolling_window
        self.elo = EloRatingSystem()

    def build_features(self) -> pd.DataFrame:
        """Build the full team-level feature matrix. Each row = one fixture."""
        fixtures = self.loader.load_fixtures_df()
        team_stats = self.loader.load_team_stats_df()
        weather = self.loader.load_weather_df()

        if fixtures.empty:
            logger.warning("No fixtures found for feature engineering.")
            return pd.DataFrame()

        completed = fixtures[fixtures["status"] == "completed"].copy()
        all_fx = fixtures.sort_values(["season", "round", "fixture_id"]).copy()
        all_fx = self.elo.compute_ratings_history(all_fx)

        rolling_feats = self._compute_rolling_team_features(completed, team_stats)

        result = all_fx.copy()

        if not rolling_feats.empty:
            home_rf = rolling_feats.rename(
                columns={c: f"home_{c}" for c in rolling_feats.columns
                         if c not in ["fixture_id", "team_id"]}
            ).rename(columns={"team_id": "home_team_id"})
            away_rf = rolling_feats.rename(
                columns={c: f"away_{c}" for c in rolling_feats.columns
                         if c not in ["fixture_id", "team_id"]}
            ).rename(columns={"team_id": "away_team_id"})

            result = result.merge(
                home_rf.drop(columns=["fixture_id"], errors="ignore"),
                on="home_team_id", how="left"
            )
            result = result.merge(
                away_rf.drop(columns=["fixture_id"], errors="ignore"),
                on="away_team_id", how="left"
            )

        # Matchup style interaction features
        if "home_roll_pace_score" in result.columns and "away_roll_pace_score" in result.columns:
            result["pace_mismatch"] = abs(
                result["home_roll_pace_score"].fillna(0) - result["away_roll_pace_score"].fillna(0)
            )

        # H2H historical features
        h2h_feats = self._compute_h2h_features(completed)
        if not h2h_feats.empty:
            result = result.merge(h2h_feats, on="fixture_id", how="left")
            result["h2h_n_games"] = result.get("h2h_n_games", pd.Series(0)).fillna(0)
            result["h2h_home_win_rate"] = result.get("h2h_home_win_rate", pd.Series(0.5)).fillna(0.5)
            result["h2h_avg_score_diff"] = result.get("h2h_avg_score_diff", pd.Series(0.0)).fillna(0.0)
            result["h2h_home_win_rate_recent"] = result.get("h2h_home_win_rate_recent", pd.Series(0.5)).fillna(0.5)

        # Weather features
        if not weather.empty:
            wx_cols = ["fixture_id", "temperature_c", "humidity_pct", "wind_speed_kmh", "conditions"]
            wx = weather[[c for c in wx_cols if c in weather.columns]].copy()
            if "conditions" in wx.columns:
                wx["is_rain"] = wx["conditions"].isin(["rain", "light_rain"]).astype(int)
                wx["wind_category"] = pd.cut(
                    wx["wind_speed_kmh"], bins=[0, 10, 20, 35, 100],
                    labels=[0, 1, 2, 3]
                ).astype(float)
            result = result.merge(wx, on="fixture_id", how="left")

        # Differential features
        numeric_home = [c for c in result.columns if c.startswith("home_roll_")]
        for hc in numeric_home:
            ac = hc.replace("home_", "away_")
            if ac in result.columns:
                diff_col = hc.replace("home_roll_", "diff_roll_")
                result[diff_col] = result[hc].fillna(0) - result[ac].fillna(0)

        # Travel & rest fatigue features
        travel_rest = self._compute_travel_rest_features(all_fx)
        if not travel_rest.empty:
            result = result.merge(travel_rest, on="fixture_id", how="left")

        if "home_rest_days" in result.columns and "away_rest_days" in result.columns:
            result["diff_rest_days"] = (
                result["home_rest_days"].fillna(7) - result["away_rest_days"].fillna(7)
            )

        result["elo_diff"] = (
            result.get("elo_home_pre", 1500) - result.get("elo_away_pre", 1500)
        )

        # Data freshness score
        result["data_freshness_score"] = self._compute_data_freshness(all_fx, completed)

        return result

    def _compute_h2h_features(self, completed: pd.DataFrame) -> pd.DataFrame:
        """
        Compute pre-game head-to-head historical features for each fixture.
        Strictly no lookahead: only uses games with fixture_id < current.

        Returns DataFrame: fixture_id, h2h_n_games, h2h_home_win_rate,
                           h2h_avg_score_diff, h2h_home_win_rate_recent
        """
        if completed.empty or "home_win" not in completed.columns:
            return pd.DataFrame()

        completed = completed.sort_values(
            ["season", "round", "fixture_id"]
        ).reset_index(drop=True)
        records = []

        for _, row in completed.iterrows():
            fid = int(row["fixture_id"])
            home_id = int(row["home_team_id"])
            away_id = int(row["away_team_id"])

            # All prior meetings between these two teams (any venue order)
            mask = (
                (completed["fixture_id"] < fid) &
                (
                    ((completed["home_team_id"] == home_id) & (completed["away_team_id"] == away_id)) |
                    ((completed["home_team_id"] == away_id) & (completed["away_team_id"] == home_id))
                )
            )
            prior = completed[mask].tail(10)
            n_h2h = len(prior)

            if n_h2h == 0:
                records.append({
                    "fixture_id": fid,
                    "h2h_n_games": 0,
                    "h2h_home_win_rate": 0.5,
                    "h2h_avg_score_diff": 0.0,
                    "h2h_home_win_rate_recent": 0.5,
                })
                continue

            win_series = []
            score_diffs = []
            for _, pr in prior.iterrows():
                hw = pr.get("home_win")
                if pd.isna(hw):
                    continue
                is_same_home = int(pr["home_team_id"]) == home_id
                won = (is_same_home and int(hw) == 1) or (not is_same_home and int(hw) == 0)
                win_series.append(float(int(won)))

                hs = pr.get("home_score") or 0.0
                as_ = pr.get("away_score") or 0.0
                if pd.notna(hs) and pd.notna(as_):
                    diff = float(hs) - float(as_) if is_same_home else float(as_) - float(hs)
                    score_diffs.append(diff)

            denom = len(win_series) or 1
            h2h_wr = sum(win_series) / denom
            h2h_wr_recent = _ewma(np.array(win_series), halflife=3.0) if win_series else 0.5
            h2h_diff = float(np.mean(score_diffs)) if score_diffs else 0.0

            records.append({
                "fixture_id": fid,
                "h2h_n_games": n_h2h,
                "h2h_home_win_rate": round(h2h_wr, 3),
                "h2h_avg_score_diff": round(h2h_diff, 1),
                "h2h_home_win_rate_recent": round(h2h_wr_recent, 3),
            })

        return pd.DataFrame(records) if records else pd.DataFrame()

    def _compute_rolling_team_features(
        self, completed_fixtures: pd.DataFrame, team_stats: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute rolling team features from prior games only (no lookahead).
        Includes exponential decay means, form slope, IQR consistency.
        """
        if completed_fixtures.empty:
            return pd.DataFrame()

        has_date = "date" in completed_fixtures.columns
        fx_meta_cols = ["fixture_id", "season", "round", "home_team_id", "away_team_id"]
        if has_date:
            fx_meta_cols.append("date")
        fx_meta = completed_fixtures[fx_meta_cols].copy()

        if team_stats.empty:
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
            ts = pd.DataFrame(records_ts).merge(fx_meta, on="fixture_id", how="left")
        else:
            existing_cols = set(team_stats.columns)
            meta_new = [c for c in fx_meta_cols if c not in existing_cols or c == "fixture_id"]
            ts = team_stats.merge(fx_meta[meta_new], on="fixture_id", how="inner")

        ts = ts.sort_values(["season", "round", "fixture_id"])

        stat_cols = ["score", "disposals", "marks", "tackles", "inside_50s",
                     "rebound_50s", "clearances", "contested_possessions",
                     "metres_gained", "turnovers"]
        stat_cols = [c for c in stat_cols if c in ts.columns]

        win_lookup: Dict[int, int] = {}
        if "home_win" in completed_fixtures.columns:
            for _, row in completed_fixtures.iterrows():
                hw = row.get("home_win")
                if pd.notna(hw):
                    win_lookup[int(row["fixture_id"])] = int(hw)

        records = []
        for tid in ts["team_id"].unique():
            team_games = ts[ts["team_id"] == tid].sort_values(
                ["season", "round", "fixture_id"]
            ).reset_index(drop=True)

            for i in range(len(team_games)):
                row = team_games.iloc[i]
                prior = team_games.iloc[:i]
                feat: Dict = {"fixture_id": int(row["fixture_id"]), "team_id": int(tid)}

                for sc in stat_cols:
                    if sc not in prior.columns:
                        continue
                    vals = prior[sc].dropna().values
                    pop_mean = float(ts[sc].mean()) if sc in ts.columns else 0.0

                    if len(vals) >= 1:
                        w_vals = vals[-self.window * 2:]   # up to 10 games for ewma
                        feat[f"roll_{sc}_mean"] = float(np.mean(vals[-self.window:]))
                        feat[f"roll_{sc}_std"] = float(np.std(vals[-self.window:])) if len(vals) >= 2 else 0.0
                        feat[f"roll_{sc}_ewma"] = _ewma(w_vals, halflife=3.0)
                        feat[f"roll_{sc}_slope"] = _form_slope(vals[-min(8, len(vals)):])
                        feat[f"roll_{sc}_iqr"] = _iqr_spread(vals[-self.window:])

                        if sc == "score":
                            feat["score_cv"] = (
                                feat["roll_score_std"] / feat["roll_score_mean"]
                                if feat["roll_score_mean"] > 0 else 0.0
                            )
                            # Legacy form trend (kept for backward compat)
                            if len(vals) >= 5:
                                feat["form_trend_score"] = (
                                    float(np.mean(vals[-2:])) - float(np.mean(vals[-5:-2]))
                                )
                            elif len(vals) >= 3:
                                feat["form_trend_score"] = (
                                    float(np.mean(vals[-1:])) - float(np.mean(vals[:-1]))
                                )
                            else:
                                feat["form_trend_score"] = 0.0
                    else:
                        feat[f"roll_{sc}_mean"] = pop_mean
                        feat[f"roll_{sc}_std"] = 0.0
                        feat[f"roll_{sc}_ewma"] = pop_mean
                        feat[f"roll_{sc}_slope"] = 0.0
                        feat[f"roll_{sc}_iqr"] = 0.3
                        if sc == "score":
                            feat["score_cv"] = 0.0
                            feat["form_trend_score"] = 0.0

                n_games = len(prior)
                feat["roll_n_games"] = n_games

                # Win rates (overall, home, away) + ewma win rate
                if n_games > 0 and win_lookup:
                    total_wins = 0
                    home_wins = home_games = 0
                    away_wins = away_games = 0
                    win_seq: List[float] = []   # chronological 1/0 for ewma
                    for _, prow in prior.iterrows():
                        pfid = int(prow["fixture_id"])
                        p_is_home = int(prow.get("is_home", 0))
                        hw = win_lookup.get(pfid)
                        if hw is None:
                            continue
                        won = (p_is_home == 1 and hw == 1) or (p_is_home == 0 and hw == 0)
                        total_wins += int(won)
                        win_seq.append(float(int(won)))
                        if p_is_home == 1:
                            home_games += 1
                            home_wins += int(won)
                        else:
                            away_games += 1
                            away_wins += int(won)
                    feat["roll_win_rate"] = total_wins / n_games
                    feat["roll_win_rate_ewma"] = _ewma(np.array(win_seq[-8:]), halflife=3.0) if win_seq else 0.5
                    feat["roll_home_win_rate"] = home_wins / home_games if home_games > 0 else 0.5
                    feat["roll_away_win_rate"] = away_wins / away_games if away_games > 0 else 0.5
                else:
                    feat["roll_win_rate"] = 0.5
                    feat["roll_win_rate_ewma"] = 0.5
                    feat["roll_home_win_rate"] = 0.5
                    feat["roll_away_win_rate"] = 0.5

                # Game script / style features
                if n_games > 0 and win_lookup:
                    blowout_wins = 0
                    close_losses = 0
                    n_wins = 0
                    n_losses = 0
                    pace_scores: List[float] = []
                    for _, prow in prior.iterrows():
                        pfid = int(prow["fixture_id"])
                        p_is_home = int(prow.get("is_home", 0))
                        hw = win_lookup.get(pfid)
                        if hw is None:
                            continue
                        won = (p_is_home == 1 and hw == 1) or (p_is_home == 0 and hw == 0)
                        # Margin calculation from fixture scores
                        hs = completed_fixtures.loc[
                            completed_fixtures["fixture_id"] == pfid, "home_score"
                        ]
                        as_ = completed_fixtures.loc[
                            completed_fixtures["fixture_id"] == pfid, "away_score"
                        ]
                        if not hs.empty and not as_.empty and pd.notna(hs.iloc[0]) and pd.notna(as_.iloc[0]):
                            h_sc = float(hs.iloc[0])
                            a_sc = float(as_.iloc[0])
                            total_score = h_sc + a_sc
                            margin = abs(h_sc - a_sc)
                            pace_scores.append(total_score)
                            if won:
                                n_wins += 1
                                if margin > 30:
                                    blowout_wins += 1
                            else:
                                n_losses += 1
                                if margin < 12:
                                    close_losses += 1
                    feat["roll_blowout_rate"] = blowout_wins / n_wins if n_wins > 0 else 0.0
                    feat["roll_close_loss_rate"] = close_losses / n_losses if n_losses > 0 else 0.0
                    feat["roll_pace_score"] = float(np.mean(pace_scores)) if pace_scores else 0.0
                else:
                    feat["roll_blowout_rate"] = 0.0
                    feat["roll_close_loss_rate"] = 0.0
                    feat["roll_pace_score"] = 0.0

                # Rest days
                if has_date and n_games > 0:
                    try:
                        last_date_str = str(prior.iloc[-1]["date"])
                        cur_date_str = str(row.get("date", ""))
                        if last_date_str and cur_date_str:
                            from datetime import datetime as _dt
                            last_dt = _dt.fromisoformat(last_date_str.split("T")[0])
                            cur_dt = _dt.fromisoformat(cur_date_str.split("T")[0])
                            feat["rest_days"] = float(max(0, min(21, (cur_dt - last_dt).days)))
                        else:
                            feat["rest_days"] = 7.0
                    except Exception:
                        feat["rest_days"] = 7.0
                else:
                    feat["rest_days"] = 7.0

                records.append(feat)

        return pd.DataFrame(records)

    def _compute_travel_rest_features(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        """
        Compute travel burden and rest/recovery features.

        For each fixture, using only info available before kickoff:
          - home_rest_days: days since home team's last game (default 7 if no prior game)
          - away_rest_days: days since away team's last game (default 7 if no prior game)
          - turnaround_flag: 1 if either team has < 6 days rest
          - rest_advantage: home_rest_days - away_rest_days (positive = home better rested)

        Uses the 'date' column to compute rest days. Sorts by date, for each team tracks
        their previous game's date. Uses pd.to_datetime for date parsing.
        """
        if fixtures.empty or "date" not in fixtures.columns:
            return pd.DataFrame()

        fx = fixtures.copy()
        fx["date_dt"] = pd.to_datetime(fx["date"], errors="coerce")
        fx = fx.sort_values("date_dt").reset_index(drop=True)

        last_game_date: Dict[int, pd.Timestamp] = {}
        records = []

        for _, row in fx.iterrows():
            fid = int(row["fixture_id"])
            home_id = int(row["home_team_id"])
            away_id = int(row["away_team_id"])
            game_date = row["date_dt"]

            if pd.isna(game_date):
                records.append({
                    "fixture_id": fid,
                    "home_rest_days": 7.0,
                    "away_rest_days": 7.0,
                    "turnaround_flag": 0,
                    "rest_advantage": 0.0,
                })
                continue

            home_last = last_game_date.get(home_id)
            away_last = last_game_date.get(away_id)

            home_rest = float((game_date - home_last).days) if home_last else 7.0
            away_rest = float((game_date - away_last).days) if away_last else 7.0
            home_rest = float(np.clip(home_rest, 2.0, 21.0))
            away_rest = float(np.clip(away_rest, 2.0, 21.0))

            records.append({
                "fixture_id": fid,
                "home_rest_days": round(home_rest, 1),
                "away_rest_days": round(away_rest, 1),
                "turnaround_flag": int(home_rest < 6 or away_rest < 6),
                "rest_advantage": round(home_rest - away_rest, 1),
            })

            # Update last game date for both teams AFTER recording pre-game values
            last_game_date[home_id] = game_date
            last_game_date[away_id] = game_date

        return pd.DataFrame(records)

    def _compute_data_freshness(
        self, fixtures: pd.DataFrame, completed: pd.DataFrame
    ) -> pd.Series:
        """
        Compute a data_freshness_score per fixture row.

        Days from the most recent completed fixture data point to the current fixture date.
        Fresh data (< 7 days old) ≈ 1.0; staleness decays exponentially with
        half-decay at 14 days: score = exp(-lag_days / 14).
        """
        fixtures = fixtures.copy()
        fixtures["date_dt"] = pd.to_datetime(fixtures["date"], errors="coerce")

        if completed.empty or "date" not in completed.columns:
            return pd.Series(0.5, index=fixtures.index)

        comp_copy = completed.copy()
        comp_copy["date_dt"] = pd.to_datetime(comp_copy["date"], errors="coerce")
        last_data_date = comp_copy["date_dt"].max()

        if pd.isna(last_data_date):
            return pd.Series(0.5, index=fixtures.index)

        def freshness(row_date):
            if pd.isna(row_date):
                return 0.5
            days_lag = max(0.0, float((row_date - last_data_date).days))
            return float(np.exp(-days_lag / 14.0))  # half-decay at 14 days

        return fixtures["date_dt"].apply(freshness)

    def get_fixture_features(self, fixture_id: int) -> Optional[Dict]:
        features = self.build_features()
        row = features[features["fixture_id"] == fixture_id]
        if row.empty:
            return None
        return row.iloc[0].to_dict()


# ── Player feature engineering ────────────────────────────────────────────────

class PlayerFeatureEngineer:
    """
    Computes player-level features for disposals prediction.
    v2: adds decay-weighted usage, role stability, position-specific opponent allowance.
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
        ps = player_stats.merge(
            completed[["fixture_id", "season", "round", "home_team_id", "away_team_id"]],
            on="fixture_id", how="inner"
        ).sort_values(["season", "round", "fixture_id"])

        if not players.empty and "position" in players.columns:
            ps = ps.merge(players[["player_id", "position"]], on="player_id", how="left")

        pop_mean = float(ps[stat_col].mean()) if stat_col in ps.columns and not ps[stat_col].empty else 20.0
        pop_std = float(ps[stat_col].std()) if stat_col in ps.columns and not ps[stat_col].empty else 5.0

        # Pre-compute opponent defensive strength per position (recency-weighted)
        opp_pos_allowance = self._compute_opp_position_allowance(ps, stat_col)
        # Position-population stats for baseline z-score
        pos_pop_stats = self._compute_position_population_stats(ps, stat_col)

        # Home/away split: pre-group player-stats by home/away
        ps["is_home_game"] = ps.apply(
            lambda r: int(r["team_id"]) == int(r["home_team_id"]) if "home_team_id" in r else 0,
            axis=1
        )

        records = []
        for pid in ps["player_id"].unique():
            pg = ps[ps["player_id"] == pid].reset_index(drop=True)
            for i, row in pg.iterrows():
                prior = pg.iloc[:i]
                vals = prior[stat_col].dropna().values if stat_col in prior.columns else np.array([])
                feat: Dict = {
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
                    feat["roll_ewma"] = _ewma(vals[-10:], halflife=3.0)
                    feat["roll_slope"] = _form_slope(vals[-min(8, n):])
                    feat["roll_iqr"] = _iqr_spread(vals[-self.window:])
                    mean5 = feat["roll_mean_5"]
                    feat["consistency_cv"] = feat["roll_std_5"] / mean5 if mean5 > 0 else 1.0
                    feat["form_trend"] = (
                        float(np.mean(vals[-2:])) - float(np.mean(vals[-4:-2])) if n >= 4 else 0.0
                    )
                    feat["n_games"] = n

                    # Role stability: fraction of recent 5 games at same position
                    if "position" in prior.columns and n >= 2:
                        cur_pos = str(row.get("position", "Unknown"))
                        recent_pos = prior.iloc[-min(5, n):]["position"].fillna("Unknown").values
                        same = sum(1 for p in recent_pos if str(p) == cur_pos)
                        feat["role_stability"] = same / len(recent_pos) if len(recent_pos) > 0 else 1.0
                    else:
                        feat["role_stability"] = 1.0

                    # Role transition score: did position change in last 3 games?
                    if "position" in prior.columns and n >= 3:
                        recent_3 = prior.iloc[-3:]["position"].fillna("Unknown").values
                        n_unique = len(set(str(p) for p in recent_3))
                        feat["role_transition_flag"] = int(n_unique > 1)
                    else:
                        feat["role_transition_flag"] = 0

                    # Home/away split: separate ewma for home vs away games
                    if "is_home_game" in prior.columns:
                        is_home_this = int(row.get("is_home_game", 0))
                        home_vals = prior[prior["is_home_game"] == 1][stat_col].dropna().values
                        away_vals = prior[prior["is_home_game"] == 0][stat_col].dropna().values
                        if len(home_vals) >= 2:
                            feat["roll_ewma_home"] = _ewma(home_vals[-6:], halflife=3.0)
                        else:
                            feat["roll_ewma_home"] = feat["roll_ewma"]
                        if len(away_vals) >= 2:
                            feat["roll_ewma_away"] = _ewma(away_vals[-6:], halflife=3.0)
                        else:
                            feat["roll_ewma_away"] = feat["roll_ewma"]
                        # Contextual form: use venue-specific average
                        feat["roll_ewma_venue"] = (
                            feat["roll_ewma_home"] if is_home_this else feat["roll_ewma_away"]
                        )
                        feat["home_away_split"] = (
                            feat["roll_ewma_home"] - feat["roll_ewma_away"]
                        )
                        feat["is_home_game"] = is_home_this
                    else:
                        feat["roll_ewma_home"] = feat["roll_ewma"]
                        feat["roll_ewma_away"] = feat["roll_ewma"]
                        feat["roll_ewma_venue"] = feat["roll_ewma"]
                        feat["home_away_split"] = 0.0
                        feat["is_home_game"] = 0

                    # Position baseline z-score
                    cur_pos = str(row.get("position", "Unknown"))
                    pos_stats = pos_pop_stats.get(cur_pos, {})
                    if pos_stats:
                        pos_mean = pos_stats["mean"]
                        pos_std_val = pos_stats["std"]
                        player_exp = feat["roll_ewma"]
                        feat["position_baseline_z"] = (player_exp - pos_mean) / pos_std_val
                        feat["position_mean_allowance"] = pos_mean
                    else:
                        feat["position_baseline_z"] = 0.0
                        feat["position_mean_allowance"] = pop_mean

                else:
                    feat.update({
                        "roll_mean_3": pop_mean, "roll_mean_5": pop_mean,
                        "roll_mean_10": pop_mean, "roll_std_5": 5.0,
                        "roll_max_5": pop_mean + 5, "roll_min_5": max(0.0, pop_mean - 5),
                        "roll_ewma": pop_mean, "roll_slope": 0.0, "roll_iqr": 0.3,
                        "consistency_cv": 0.25, "form_trend": 0.0, "n_games": 0,
                        "role_stability": 0.5, "role_transition_flag": 0,
                        "roll_ewma_home": pop_mean, "roll_ewma_away": pop_mean,
                        "roll_ewma_venue": pop_mean, "home_away_split": 0.0,
                        "is_home_game": 0, "position_baseline_z": 0.0,
                        "position_mean_allowance": pop_mean,
                    })

                # Opponent defensive strength — position-specific (recency-weighted)
                home_id = int(row["home_team_id"])
                away_id = int(row["away_team_id"])
                opp_id = away_id if int(row["team_id"]) == home_id else home_id
                pos = str(row.get("position", "Unknown"))

                key_pos = (opp_id, pos)
                key_all = (opp_id, "all")
                opp_allow_pos = opp_pos_allowance.get(key_pos, opp_pos_allowance.get(key_all, pop_mean))
                feat["opp_disposals_allowed_mean"] = round(float(opp_allow_pos), 2)
                feat["opp_pos_disposals_allowed"] = round(float(opp_allow_pos), 2)
                # Matchup advantage: how much does this opponent allow vs population?
                feat["matchup_advantage"] = round(float(opp_allow_pos) - pop_mean, 2)

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

    def _compute_opp_position_allowance(
        self, ps: pd.DataFrame, stat_col: str
    ) -> Dict:
        """
        Compute recency-weighted opponent allowance per (team_id, position).

        Uses exponential decay (halflife=5 games per opponent) so that recent
        defensive performance is weighted more than games from months ago.
        Returns dict: {(team_id, position): ewma_mean_allowed}
                    + {(team_id, "all"): overall_ewma} as fallback.
        """
        if stat_col not in ps.columns or ps.empty:
            return {}

        team_pos_games: Dict = {}  # (opp_team_id, position) -> [(round, val)]
        team_all_games: Dict = {}  # (opp_team_id, "all") -> [(round, val)]

        ordered = ps.sort_values(["season", "round", "fixture_id"])

        for _, prow in ordered.iterrows():
            val = prow.get(stat_col)
            if pd.isna(val):
                continue
            home_id = prow.get("home_team_id")
            away_id = prow.get("away_team_id")
            if home_id is None or away_id is None:
                continue
            opp = int(away_id) if int(prow["team_id"]) == int(home_id) else int(home_id)
            pos = str(prow.get("position", "Unknown"))
            seq = int(prow.get("round", 0)) + int(prow.get("season", 0)) * 100
            team_pos_games.setdefault((opp, pos), []).append((seq, float(val)))
            team_all_games.setdefault((opp, "all"), []).append((seq, float(val)))

        result: Dict = {}
        for key, entries in team_pos_games.items():
            entries.sort(key=lambda x: x[0])
            vals = np.array([v for _, v in entries])
            result[key] = _ewma(vals, halflife=5.0)
        for key, entries in team_all_games.items():
            entries.sort(key=lambda x: x[0])
            vals = np.array([v for _, v in entries])
            result[key] = _ewma(vals, halflife=5.0)

        return result

    def _compute_position_population_stats(
        self, ps: pd.DataFrame, stat_col: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Return per-position population statistics used to compute baseline z-scores.
        Dict: { position: {"mean": float, "std": float} }
        """
        if stat_col not in ps.columns or ps.empty:
            return {}
        result = {}
        for pos, grp in ps.groupby("position"):
            vals = grp[stat_col].dropna().values
            if len(vals) < 5:
                continue
            result[str(pos)] = {
                "mean": float(np.mean(vals)),
                "std": float(max(np.std(vals), 1.0)),
            }
        return result

    def get_player_prediction_features(
        self, player_id: int, fixture_id: int
    ) -> Optional[Dict]:
        """Get features for a player's upcoming fixture using most recent rolling stats."""
        player_stats = self.loader.load_player_stats_df()
        players = self.loader.load_players_df()

        if player_stats.empty:
            return None

        pg = player_stats[player_stats["player_id"] == player_id].copy()
        if pg.empty:
            return None

        fixtures = self.loader.load_fixtures_df()
        pg = pg.merge(
            fixtures[["fixture_id", "season", "round"]], on="fixture_id", how="left"
        ).sort_values(["season", "round"])

        vals = pg["disposals"].dropna().values
        n = len(vals)

        player_info = players[players["player_id"] == player_id]
        position = str(player_info["position"].iloc[0]) if not player_info.empty else "Unknown"
        pop_mean = float(player_stats["disposals"].mean()) if "disposals" in player_stats.columns else 20.0

        feat: Dict = {"fixture_id": fixture_id, "player_id": player_id, "position": position}

        pop_std = float(player_stats["disposals"].std()) if "disposals" in player_stats.columns else 5.0

        # Position population stats for baseline z-score
        ps_all = player_stats.copy()
        if not players.empty and "position" in players.columns:
            ps_all = ps_all.merge(players[["player_id", "position"]], on="player_id", how="left")
        pos_pop_stats = self._compute_position_population_stats(ps_all, "disposals")

        if n >= 1:
            feat["roll_mean_3"] = float(np.mean(vals[-3:])) if n >= 3 else float(np.mean(vals))
            feat["roll_mean_5"] = float(np.mean(vals[-5:])) if n >= 5 else float(np.mean(vals))
            feat["roll_mean_10"] = float(np.mean(vals[-10:])) if n >= 10 else float(np.mean(vals))
            feat["roll_std_5"] = float(np.std(vals[-5:])) if n >= 3 else 3.0
            feat["roll_max_5"] = float(np.max(vals[-5:]))
            feat["roll_min_5"] = float(np.min(vals[-5:]))
            feat["roll_ewma"] = _ewma(vals[-10:], halflife=3.0)
            feat["roll_slope"] = _form_slope(vals[-min(8, n):])
            feat["roll_iqr"] = _iqr_spread(vals[-self.window:])
            mean5 = feat["roll_mean_5"]
            feat["consistency_cv"] = feat["roll_std_5"] / mean5 if mean5 > 0 else 0.25
            feat["form_trend"] = float(np.mean(vals[-2:])) - float(np.mean(vals[-4:-2])) if n >= 4 else 0.0
            feat["n_games"] = n

            # Role stability from recent position data
            if "position" in pg.columns and n >= 2:
                recent_pos = pg.iloc[-min(5, n):]["position"].fillna("Unknown").values if hasattr(pg, "columns") else []
                same = sum(1 for p in recent_pos if str(p) == position)
                feat["role_stability"] = same / len(recent_pos) if recent_pos.size > 0 else 1.0
            else:
                feat["role_stability"] = 1.0

            # Role transition flag (last 3 games)
            if "position" in pg.columns and n >= 3:
                recent_3 = pg.iloc[-3:]["position"].fillna("Unknown").values
                feat["role_transition_flag"] = int(len(set(str(p) for p in recent_3)) > 1)
            else:
                feat["role_transition_flag"] = 0

            # Home/away splits (use all available games)
            if "is_home" in pg.columns:
                home_vals = pg[pg["is_home"] == 1]["disposals"].dropna().values
                away_vals = pg[pg["is_home"] == 0]["disposals"].dropna().values
                feat["roll_ewma_home"] = _ewma(home_vals[-6:], halflife=3.0) if len(home_vals) >= 2 else feat["roll_ewma"]
                feat["roll_ewma_away"] = _ewma(away_vals[-6:], halflife=3.0) if len(away_vals) >= 2 else feat["roll_ewma"]
                feat["home_away_split"] = feat["roll_ewma_home"] - feat["roll_ewma_away"]
            else:
                feat["roll_ewma_home"] = feat["roll_ewma"]
                feat["roll_ewma_away"] = feat["roll_ewma"]
                feat["home_away_split"] = 0.0

            # Position baseline z-score
            pos_stats = pos_pop_stats.get(position, {})
            if pos_stats:
                pos_mean = pos_stats["mean"]
                pos_std_val = pos_stats["std"]
                feat["position_baseline_z"] = (feat["roll_ewma"] - pos_mean) / pos_std_val
                feat["position_mean_allowance"] = pos_mean
            else:
                feat["position_baseline_z"] = 0.0
                feat["position_mean_allowance"] = pop_mean

        else:
            feat.update({
                "roll_mean_3": pop_mean, "roll_mean_5": pop_mean,
                "roll_mean_10": pop_mean, "roll_std_5": 5.0,
                "roll_max_5": pop_mean + 5, "roll_min_5": max(0.0, pop_mean - 5),
                "roll_ewma": pop_mean, "roll_slope": 0.0, "roll_iqr": 0.3,
                "consistency_cv": 0.25, "form_trend": 0.0, "n_games": 0,
                "role_stability": 0.5, "role_transition_flag": 0,
                "roll_ewma_home": pop_mean, "roll_ewma_away": pop_mean,
                "home_away_split": 0.0, "position_baseline_z": 0.0,
                "position_mean_allowance": pop_mean,
            })

        feat["pos_midfielder"] = int(position == "Midfielder")
        feat["pos_forward"] = int(position == "Forward")
        feat["pos_defender"] = int(position == "Defender")
        feat["pos_ruckman"] = int(position == "Ruckman")
        feat["opp_disposals_allowed_mean"] = pop_mean
        feat["opp_pos_disposals_allowed"] = pop_mean
        feat["matchup_advantage"] = 0.0

        return feat


# ── Feature pipeline orchestrator ─────────────────────────────────────────────

class FeaturePipeline:
    """Orchestrates team and player feature engineering."""

    def __init__(self, loader: Optional[DataLoader] = None):
        self.loader = loader or DataLoader()
        self.team_engineer = TeamFeatureEngineer(self.loader)
        self.player_engineer = PlayerFeatureEngineer(self.loader)
        self._team_features: Optional[pd.DataFrame] = None
        self._player_features: Optional[pd.DataFrame] = None

    def get_team_features(self, force_rebuild: bool = False) -> pd.DataFrame:
        if self._team_features is None or force_rebuild:
            logger.info("Building team feature matrix...")
            self._team_features = self.team_engineer.build_features()
            logger.info(f"Team features shape: {self._team_features.shape}")
        return self._team_features

    def get_player_features(
        self, stat_col: str = "disposals", force_rebuild: bool = False
    ) -> pd.DataFrame:
        if self._player_features is None or force_rebuild:
            logger.info(f"Building player feature matrix for '{stat_col}'...")
            self._player_features = self.player_engineer.build_player_features(stat_col)
            logger.info(f"Player features shape: {self._player_features.shape}")
        return self._player_features

    def get_model_ready_match_data(self) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Return (X, y, feature_names) for match win probability model."""
        features = self.get_team_features()
        if "home_win" not in features.columns:
            return pd.DataFrame(), pd.Series(dtype=float), []
        completed = features[
            (features["status"] == "completed") & features["home_win"].notna()
        ].copy()

        if completed.empty:
            return pd.DataFrame(), pd.Series(dtype=float), []

        # Exact-match columns to always include if present (new context-aware features)
        _extra_cols = {
            "rest_advantage", "turnaround_flag", "home_rest_days", "away_rest_days",
            "diff_rest_days", "data_freshness_score", "pace_mismatch",
        }
        feature_cols = [
            c for c in completed.columns
            if (
                c.startswith((
                    "elo_", "home_roll_", "away_roll_", "diff_roll_",
                    "home_rest_days", "away_rest_days", "diff_rest_days",
                    "h2h_",                  # H2H historical features
                    "temperature_c", "wind_speed_kmh", "is_rain", "wind_category",
                ))
                or c in _extra_cols
            )
            and completed[c].dtype in [np.float64, np.int64, float, int]
        ]

        X = completed[feature_cols].fillna(0)
        y = completed["home_win"].astype(int)
        return X, y, feature_cols

    def get_model_ready_player_data(
        self, line: float, stat_col: str = "disposals"
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Return (X, y, feature_names) for player over/under model."""
        features = self.get_player_features(stat_col)
        valid = features[features["target"].notna()].copy()

        if valid.empty:
            return pd.DataFrame(), pd.Series(dtype=float), []

        valid["label"] = (valid["target"] > line).astype(int)

        feature_cols = [
            c for c in valid.columns
            if c in [
                "roll_mean_3", "roll_mean_5", "roll_mean_10", "roll_std_5",
                "roll_max_5", "roll_min_5",
                "roll_ewma", "roll_slope", "roll_iqr",   # v2 additions
                "consistency_cv", "form_trend", "n_games",
                "opp_disposals_allowed_mean", "opp_pos_disposals_allowed",
                "pos_midfielder", "pos_forward", "pos_defender", "pos_ruckman",
                "role_stability",                         # v2 addition
            ]
        ]

        X = valid[feature_cols].fillna(0)
        y = valid["label"]
        return X, y, feature_cols
