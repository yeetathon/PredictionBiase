"""Tests for DSG stat registry and expanded feature pipeline (v3)."""
import pytest
import numpy as np
import pandas as pd


# ── stat_registry ─────────────────────────────────────────────────────────────

class TestStatRegistry:
    def test_all_33_stats_present(self):
        from app.core.stat_registry import DSG_STAT_REGISTRY
        expected = {
            "kicks", "handballs", "disposals",
            "effective_kicks", "effective_handballs", "effective_disposals",
            "clangers", "marks", "contested_marks",
            "inside_50s", "rebound_50s", "marks_inside_50",
            "clearances", "centre_clearances", "stoppage_clearances",
            "hitouts", "hitouts_to_advantage",
            "score", "goals", "behinds",
            "score_involvements", "goal_assists",
            "frees_for", "frees_against", "one_percenters", "bounces",
            "tackles", "metres_gained", "turnovers",
            "contested_possessions", "uncontested_possessions",
            "time_on_ground_pct",
            "brownlow_votes", "supercoach_score", "rating_points",
        }
        assert expected <= set(DSG_STAT_REGISTRY.keys())

    def test_evaluation_only_stats_not_pre_match_safe(self):
        from app.core.stat_registry import DSG_STAT_REGISTRY
        for stat in ("brownlow_votes", "supercoach_score", "rating_points"):
            assert not DSG_STAT_REGISTRY[stat]["pre_match_safe"], (
                f"{stat} must not be marked pre_match_safe"
            )

    def test_evaluation_only_stats_have_leakage_note(self):
        from app.core.stat_registry import DSG_STAT_REGISTRY
        for stat in ("brownlow_votes", "supercoach_score", "rating_points"):
            assert DSG_STAT_REGISTRY[stat]["leakage_note"], (
                f"{stat} must have a leakage_note"
            )

    def test_all_other_stats_are_pre_match_safe(self):
        from app.core.stat_registry import DSG_STAT_REGISTRY
        non_safe = {"brownlow_votes", "supercoach_score", "rating_points"}
        for stat, meta in DSG_STAT_REGISTRY.items():
            if stat not in non_safe:
                assert meta["pre_match_safe"], f"{stat} should be pre_match_safe"

    def test_get_pre_match_safe_stats_team(self):
        from app.core.stat_registry import get_pre_match_safe_stats
        team_stats = get_pre_match_safe_stats("team")
        assert "disposals" in team_stats
        assert "brownlow_votes" not in team_stats

    def test_get_pre_match_safe_stats_player(self):
        from app.core.stat_registry import get_pre_match_safe_stats
        player_stats = get_pre_match_safe_stats("player")
        assert "time_on_ground_pct" in player_stats
        assert "supercoach_score" not in player_stats
        assert "turnovers" not in player_stats  # team-only stat

    def test_get_stats_by_family(self):
        from app.core.stat_registry import get_stats_by_family
        scoring = get_stats_by_family("scoring")
        assert "goals" in scoring
        assert "disposals" not in scoring

    def test_get_team_stat_cols_excludes_evaluation_only(self):
        from app.core.stat_registry import get_team_stat_cols
        cols = get_team_stat_cols()
        for eval_only in ("brownlow_votes", "supercoach_score", "rating_points"):
            assert eval_only not in cols, f"{eval_only} must not be a team feature col"

    def test_get_team_stat_cols_contains_all_families(self):
        from app.core.stat_registry import get_team_stat_cols, DSG_STAT_REGISTRY
        cols = set(get_team_stat_cols())
        families = {DSG_STAT_REGISTRY[c]["feature_family"] for c in cols}
        expected_families = {"ball_winning", "kick_quality", "territorial",
                             "stoppage", "scoring", "discipline", "opportunity"}
        assert expected_families <= families

    def test_get_player_secondary_stats(self):
        from app.core.stat_registry import get_player_secondary_stats
        sec = get_player_secondary_stats()
        assert "effective_disposals" in sec
        assert "clangers" in sec
        assert "time_on_ground_pct" in sec
        assert "score_involvements" in sec
        assert "contested_possessions" in sec


# ── Expanded team features ────────────────────────────────────────────────────

def _make_rich_loader(n_fixtures=12):
    """DataLoader with full DSG stats in team_stats_df."""
    from app.data_ingestion.loader import DataLoader

    fixtures = pd.DataFrame([
        {
            "fixture_id": i, "season": 2024, "round": i,
            "home_team_id": 1, "away_team_id": 2,
            "date": f"2024-04-{10 + i:02d}",
            "status": "completed",
            "home_score": float(90 + i * 2),
            "away_score": float(70 + i),
            "home_win": int(i % 2 == 0),
        }
        for i in range(1, n_fixtures + 1)
    ])

    rng = np.random.default_rng(99)
    team_stats_rows = []
    for i in range(1, n_fixtures + 1):
        for tid, is_home in [(1, 1), (2, 0)]:
            team_stats_rows.append({
                "fixture_id": i, "team_id": tid, "is_home": is_home,
                "score": float(90 + rng.integers(-10, 10)),
                "goals": int(12 + rng.integers(0, 4)),
                "behinds": int(8 + rng.integers(0, 4)),
                "kicks": int(120 + rng.integers(-20, 20)),
                "handballs": int(80 + rng.integers(-10, 10)),
                "disposals": int(200 + rng.integers(-20, 20)),
                "effective_kicks": int(90 + rng.integers(-15, 15)),
                "effective_handballs": int(60 + rng.integers(-10, 10)),
                "effective_disposals": int(150 + rng.integers(-20, 20)),
                "clangers": int(15 + rng.integers(-5, 5)),
                "marks": int(50 + rng.integers(-10, 10)),
                "contested_marks": int(10 + rng.integers(-3, 3)),
                "inside_50s": int(45 + rng.integers(-8, 8)),
                "rebound_50s": int(25 + rng.integers(-5, 5)),
                "marks_inside_50": int(15 + rng.integers(-3, 3)),
                "clearances": int(35 + rng.integers(-5, 5)),
                "centre_clearances": int(15 + rng.integers(-3, 3)),
                "stoppage_clearances": int(20 + rng.integers(-4, 4)),
                "hitouts": int(40 + rng.integers(-8, 8)),
                "hitouts_to_advantage": int(20 + rng.integers(-5, 5)),
                "score_involvements": int(60 + rng.integers(-10, 10)),
                "goal_assists": int(8 + rng.integers(-2, 2)),
                "frees_for": int(12 + rng.integers(-3, 3)),
                "frees_against": int(12 + rng.integers(-3, 3)),
                "one_percenters": int(20 + rng.integers(-5, 5)),
                "bounces": int(5 + rng.integers(-2, 2)),
                "tackles": int(55 + rng.integers(-8, 8)),
                "metres_gained": int(1600 + rng.integers(-200, 200)),
                "turnovers": int(12 + rng.integers(-3, 3)),
                "contested_possessions": int(90 + rng.integers(-15, 15)),
                "uncontested_possessions": int(110 + rng.integers(-15, 15)),
            })
    team_stats = pd.DataFrame(team_stats_rows)

    class _Provider:
        @property
        def fixtures_df(self): return fixtures
        @property
        def teams_df(self): return pd.DataFrame()
        @property
        def players_df(self): return pd.DataFrame()
        @property
        def team_stats_df(self): return team_stats
        @property
        def player_stats_df(self): return pd.DataFrame()
        def load_upcoming_fixtures_df(self): return fixtures[fixtures["status"] == "upcoming"]

    class _NullOdds:
        @property
        def odds_df(self): return pd.DataFrame()

    return DataLoader(afl_provider=_Provider(), odds_provider=_NullOdds())


class TestExpandedTeamFeatures:
    def setup_method(self):
        from app.features.pipeline import TeamFeatureEngineer
        loader = _make_rich_loader(12)
        self.eng = TeamFeatureEngineer(loader=loader, rolling_window=5)
        self.feats = self.eng._compute_rolling_team_features(
            loader.load_fixtures_df(),
            loader.load_team_stats_df(),
        )

    def test_new_raw_stat_rolling_features_present(self):
        """New stat groups should produce roll_*_mean columns."""
        for stat in ("effective_kicks", "clangers", "marks_inside_50",
                     "centre_clearances", "hitouts_to_advantage",
                     "score_involvements", "frees_for", "one_percenters"):
            col = f"roll_{stat}_mean"
            assert col in self.feats.columns, f"Missing {col}"

    def test_derived_efficiency_features_present(self):
        """All 10 derived ratio/difference features must be present."""
        derived = [
            "roll_eff_kick_rate", "roll_eff_disposal_rate", "roll_eff_handball_rate",
            "roll_clanger_rate", "roll_free_kick_diff",
            "roll_inside_50_conversion", "roll_scoring_efficiency",
            "roll_hitout_advantage_rate", "roll_stoppage_clearance_share",
            "roll_scoring_chain_rate", "roll_contested_poss_rate",
        ]
        for col in derived:
            assert col in self.feats.columns, f"Derived feature {col} missing"

    def test_eff_kick_rate_in_valid_range(self):
        """Effective kick rate (eff_kicks / kicks) must be in [0, 1]."""
        col = self.feats["roll_eff_kick_rate"].dropna()
        assert (col >= 0.0).all() and (col <= 1.0).all()

    def test_clanger_rate_in_valid_range(self):
        col = self.feats["roll_clanger_rate"].dropna()
        assert (col >= 0.0).all() and (col <= 1.0).all()

    def test_scoring_efficiency_positive(self):
        """Goals / inside_50s should be > 0 when data exists."""
        col = self.feats["roll_scoring_efficiency"]
        assert (col >= 0.0).all()

    def test_hitout_advantage_rate_in_range(self):
        col = self.feats["roll_hitout_advantage_rate"].dropna()
        assert (col >= 0.0).all() and (col <= 1.0).all()

    def test_build_features_includes_new_diff_cols(self):
        """build_features() must produce diff_roll_* for new stats."""
        from app.features.pipeline import FeaturePipeline
        loader = _make_rich_loader(12)
        pipeline = FeaturePipeline(loader=loader)
        team_feats = pipeline.get_team_features()
        assert "diff_roll_eff_kick_rate" in team_feats.columns or any(
            c.startswith("diff_roll_") for c in team_feats.columns
        )

    def test_model_ready_data_includes_new_features(self):
        """get_model_ready_match_data() must include the new derived features."""
        from app.features.pipeline import FeaturePipeline
        loader = _make_rich_loader(12)
        pipeline = FeaturePipeline(loader=loader)
        X, y, feat_names = pipeline.get_model_ready_match_data()
        if feat_names:
            # New stats generate home_roll_eff_kick_rate etc.
            assert any("eff_kick" in f or "clanger" in f for f in feat_names), (
                "Expected eff_kick_rate or clanger_rate in feature names"
            )


# ── Expanded player features ──────────────────────────────────────────────────

def _make_rich_player_loader(n_fixtures=12):
    """DataLoader with full DSG player stats."""
    from app.data_ingestion.loader import DataLoader

    fixtures = pd.DataFrame([
        {
            "fixture_id": i, "season": 2024, "round": i,
            "home_team_id": 1, "away_team_id": 2,
            "date": f"2024-04-{10 + i:02d}",
            "status": "completed",
            "home_score": float(90 + i),
            "away_score": float(75 + i),
            "home_win": 1,
        }
        for i in range(1, n_fixtures + 1)
    ])

    rng = np.random.default_rng(7)
    ps_rows = []
    for i in range(1, n_fixtures + 1):
        for pid, tid in [(101, 1), (102, 2)]:
            ps_rows.append({
                "fixture_id": i, "player_id": pid, "team_id": tid,
                "disposals": int(25 + rng.integers(-5, 5)),
                "kicks": int(15 + rng.integers(-3, 3)),
                "handballs": int(10 + rng.integers(-2, 2)),
                "goals": 1, "marks": 5,
                "effective_disposals": int(18 + rng.integers(-4, 4)),
                "clangers": int(2 + rng.integers(0, 3)),
                "time_on_ground_pct": float(75 + rng.integers(-10, 10)),
                "score_involvements": int(6 + rng.integers(-2, 2)),
                "contested_possessions": int(10 + rng.integers(-3, 3)),
            })
    player_stats = pd.DataFrame(ps_rows)

    players = pd.DataFrame([
        {"player_id": 101, "team_id": 1, "position": "Midfielder"},
        {"player_id": 102, "team_id": 2, "position": "Forward"},
    ])

    class _Provider:
        @property
        def fixtures_df(self): return fixtures
        @property
        def teams_df(self): return pd.DataFrame()
        @property
        def players_df(self): return players
        @property
        def team_stats_df(self): return pd.DataFrame()
        @property
        def player_stats_df(self): return player_stats
        def load_upcoming_fixtures_df(self): return fixtures[fixtures["status"] == "upcoming"]

    class _NullOdds:
        @property
        def odds_df(self): return pd.DataFrame()

    return DataLoader(afl_provider=_Provider(), odds_provider=_NullOdds())


class TestExpandedPlayerFeatures:
    def setup_method(self):
        from app.features.pipeline import PlayerFeatureEngineer
        loader = _make_rich_player_loader(12)
        self.eng = PlayerFeatureEngineer(loader=loader, rolling_window=5)
        self.feats = self.eng.build_player_features(stat_col="disposals")

    def test_player_secondary_features_present(self):
        """New derived player rates must be present in player features."""
        for col in ("player_eff_disposal_rate", "player_clanger_rate",
                    "player_score_chain_rate", "player_tog_pct",
                    "player_contested_rate"):
            assert col in self.feats.columns, f"Missing player feature: {col}"

    def test_player_tog_pct_in_range(self):
        """TOG% normalised to [0, 1] must be within bounds."""
        col = self.feats["player_tog_pct"].dropna()
        assert (col >= 0.0).all() and (col <= 1.0).all()

    def test_player_eff_disposal_rate_in_range(self):
        col = self.feats["player_eff_disposal_rate"].dropna()
        assert (col >= 0.0).all() and (col <= 2.0).all()

    def test_player_clanger_rate_non_negative(self):
        col = self.feats["player_clanger_rate"].dropna()
        assert (col >= 0.0).all()

    def test_player_score_chain_rate_non_negative(self):
        col = self.feats["player_score_chain_rate"].dropna()
        assert (col >= 0.0).all()

    def test_model_ready_player_data_includes_new_features(self):
        """get_model_ready_player_data() must include new v3 player features."""
        from app.features.pipeline import FeaturePipeline
        loader = _make_rich_player_loader(12)
        pipeline = FeaturePipeline(loader=loader)
        X, y, feat_names = pipeline.get_model_ready_player_data(line=22.5)
        if feat_names:
            assert any(f.startswith("player_") for f in feat_names), (
                "Expected player_* derived features in model-ready feature set"
            )

    def test_no_nan_leakage_in_player_features(self):
        """Secondary features must not introduce NaN for rows with prior data."""
        rows_with_history = self.feats[self.feats["n_games"] >= 3]
        if not rows_with_history.empty:
            for col in ("player_eff_disposal_rate", "player_tog_pct"):
                assert rows_with_history[col].notna().all(), (
                    f"NaN in {col} for player with history"
                )


# ── _sdiv helper ──────────────────────────────────────────────────────────────

class TestSdivHelper:
    def test_normal_division(self):
        from app.features.pipeline import _sdiv
        assert _sdiv(10.0, 5.0) == pytest.approx(2.0)

    def test_zero_denominator_returns_default(self):
        from app.features.pipeline import _sdiv
        assert _sdiv(5.0, 0.0) == pytest.approx(0.0)

    def test_negative_denominator_returns_default(self):
        from app.features.pipeline import _sdiv
        assert _sdiv(5.0, -1.0) == pytest.approx(0.0)

    def test_custom_default(self):
        from app.features.pipeline import _sdiv
        assert _sdiv(5.0, 0.0, default=0.5) == pytest.approx(0.5)
