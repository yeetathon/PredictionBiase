"""
Multi-signal prediction engine — v2.

Computes independent signals for each AFL match outcome and measures
their agreement. High agreement → higher confidence. Disagreement →
lower confidence / outright rejection.

v2 additions:
  H2H Signal: head-to-head historical record between the specific two teams.
  Prediction intervals: 80% CI derived from signal distribution.
  Decay-weighted scoring: EWMA scores weight recent AFL form more heavily.
  Improved player signals: decay-weighted short-form + matchup signal.

Signals (H2H):
  1. Elo          — long-run strength (elo_win_prob_home)
  2. Form         — rolling win-rate differential (venue-adjusted, ewma-weighted)
  3. Scoring      — recent scoring power + trend (ewma, not simple mean)
  4. H2H          — head-to-head historical record (last 10 meetings)
  5. Market       — bookmaker consensus (vig-adjusted, when available)

Signals (Player disposals):
  1. Short-form   — ewma 3-game average vs line
  2. Medium-form  — ewma 5-game average vs line
  3. Trend        — form slope direction
  4. Matchup      — opponent defensive allowance vs line
  5. ML model     — XGBoost prediction (when available)
  6. Market       — bookmaker line (when available)
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from loguru import logger


@dataclass
class Signal:
    name: str
    probability: float      # P(home win) or P(over)
    weight: float           # base contribution weight
    reliability: float      # 0–1; down-weights uncertain/sparse signals
    explanation: str


@dataclass
class SignalResult:
    """All signal outputs for one leg direction."""
    signals: List[Signal]
    consensus_probability: float    # reliability-weighted combination
    signal_agreement: float         # 0–1; 1 = perfect agreement
    prediction_variance: float      # variance of probabilities across signals
    data_completeness: float        # 0–1; overall data quality
    n_active_signals: int           # signals with reliability > 0.1
    top_factors: List[str]          # human-readable bullets
    explanation: str                # summary line
    # v2: prediction interval (80% confidence)
    prediction_low: float = 0.0     # 10th percentile of signal distribution
    prediction_high: float = 1.0    # 90th percentile of signal distribution


class SignalEngine:
    """
    Computes independent prediction signals and a consensus probability.

    Usage:
        engine = SignalEngine()
        home_result, away_result = engine.compute_h2h_signals(features_dict, market_probs)

    features_dict keys (from TeamFeatureEngineer.build_features()):
        elo_win_prob_home, elo_home_pre, elo_away_pre, elo_diff,
        home_roll_win_rate, away_roll_win_rate,
        home_roll_win_rate_ewma, away_roll_win_rate_ewma,        [v2]
        home_roll_home_win_rate, away_roll_away_win_rate,
        home_roll_score_mean, away_roll_score_mean,
        home_roll_score_ewma, away_roll_score_ewma,              [v2]
        home_roll_score_slope, away_roll_score_slope,            [v2]
        home_roll_score_std, away_roll_score_std,
        home_roll_n_games, away_roll_n_games,
        home_form_trend_score, away_form_trend_score,
        home_rest_days, away_rest_days, diff_rest_days,
        h2h_n_games, h2h_home_win_rate,                         [v2]
        h2h_home_win_rate_recent, h2h_avg_score_diff            [v2]
    """

    # std above this → agreement → 0
    _AGREEMENT_SCALE = 0.15

    def __init__(self):
        try:
            from app.pricing.signal_weights import get_signal_weight_store
            self._weight_store = get_signal_weight_store()
        except Exception:
            self._weight_store = None

    def _get_signal_weight(self, market_type: str, signal_name: str, default: float) -> float:
        """Get learned weight for a signal, falling back to hardcoded default."""
        if self._weight_store is None:
            return default
        try:
            weights = self._weight_store.get_weights(market_type)
            return float(weights.get(signal_name, default))
        except Exception:
            return default

    def compute_h2h_signals(
        self,
        features: Dict,
        market_probs: Optional[Dict[str, float]] = None,
    ) -> Tuple[SignalResult, SignalResult]:
        """
        Compute all signals for a home-vs-away fixture.
        Returns (home_win_result, away_win_result).
        """
        n_home = int(float(features.get("home_roll_n_games", 0) or 0))
        n_away = int(float(features.get("away_roll_n_games", 0) or 0))
        n_games = min(n_home, n_away)
        data_completeness = self._data_completeness(features, n_games)

        home_sigs: List[Signal] = []

        # ── Signal 1: Elo ─────────────────────────────────────────────────
        elo_prob = float(np.clip(features.get("elo_win_prob_home") or 0.5, 0.05, 0.95))
        elo_diff = float(features.get("elo_diff") or 0.0)
        elo_home = float(features.get("elo_home_pre") or 1500.0)
        elo_away = float(features.get("elo_away_pre") or 1500.0)
        elo_reliability = float(np.clip(n_games / 20.0, 0.1, 1.0))

        if abs(elo_diff) < 15:
            elo_exp = f"Elo: near-even matchup (diff={elo_diff:+.0f} pts), P(home)={elo_prob:.1%}"
        else:
            stronger = "Home" if elo_diff > 0 else "Away"
            elo_exp = (
                f"Elo: {stronger} stronger by {abs(elo_diff):.0f} pts "
                f"({elo_home:.0f} vs {elo_away:.0f}), P(home)={elo_prob:.1%}"
            )
        home_sigs.append(Signal("elo", elo_prob,
                                self._get_signal_weight("head_to_head", "elo", 0.25),
                                elo_reliability, elo_exp))

        # ── Signal 2: Form (ewma win-rate, venue-adjusted, rest-adjusted) ──
        home_wr = float(features.get("home_roll_win_rate_ewma")
                        or features.get("home_roll_win_rate") or 0.5)
        away_wr = float(features.get("away_roll_win_rate_ewma")
                        or features.get("away_roll_win_rate") or 0.5)
        home_home_wr = float(features.get("home_roll_home_win_rate") or home_wr)
        away_away_wr = float(features.get("away_roll_away_win_rate") or away_wr)
        # 60% overall + 40% venue-specific
        home_form = 0.60 * home_wr + 0.40 * home_home_wr
        away_form = 0.60 * away_wr + 0.40 * away_away_wr
        form_diff = home_form - away_form
        # Rest advantage: clip to ±0.08 boost (±2.4 days advantage = ±0.08)
        rest_diff = float(features.get("rest_advantage", 0.0))
        rest_adj = float(np.clip(rest_diff / 30.0, -0.08, 0.08))
        form_diff_adj = form_diff + rest_adj
        form_prob = float(np.clip(1.0 / (1.0 + np.exp(-form_diff_adj * 5.0)), 0.10, 0.90))
        form_reliability = float(np.clip(n_games / 12.0, 0.05, 1.0))
        rest_note = f" | rest adv {rest_diff:+.1f}d" if abs(rest_diff) >= 2 else ""
        form_exp = (
            f"Form (ewma): home {home_wr:.0%} WR (at-home {home_home_wr:.0%}) "
            f"vs away {away_wr:.0%} WR (away {away_away_wr:.0%}){rest_note}; P(home)={form_prob:.1%}"
        )
        home_sigs.append(Signal("form", form_prob,
                                self._get_signal_weight("head_to_head", "form", 0.20),
                                form_reliability, form_exp))

        # ── Signal 3: Scoring power + slope (v2: ewma not simple mean) ────
        # Use ewma scores if available, fall back to simple mean
        home_sc = float(features.get("home_roll_score_ewma")
                        or features.get("home_roll_score_mean") or 0.0)
        away_sc = float(features.get("away_roll_score_ewma")
                        or features.get("away_roll_score_mean") or 0.0)
        home_sc_std = float(features.get("home_roll_score_std") or 15.0)
        away_sc_std = float(features.get("away_roll_score_std") or 15.0)
        home_slope = float(features.get("home_roll_score_slope")
                           or features.get("home_form_trend_score") or 0.0)
        away_slope = float(features.get("away_roll_score_slope")
                           or features.get("away_form_trend_score") or 0.0)

        scoring_prob = 0.5
        scoring_reliability = 0.0
        scoring_exp = "Scoring: insufficient score data"

        if home_sc > 0 and away_sc > 0:
            # Slope-adjusted expected scores (slope in pts/game, project 1 game ahead)
            home_adj = home_sc + home_slope * 0.5
            away_adj = away_sc + away_slope * 0.5
            sc_diff = home_adj - away_adj
            combined_std = float(max(20.0, np.sqrt(home_sc_std ** 2 + away_sc_std ** 2)))
            from scipy.stats import norm as _norm
            scoring_prob = float(np.clip(_norm.cdf(sc_diff / combined_std), 0.10, 0.90))
            scoring_reliability = float(np.clip(n_games / 10.0, 0.05, 1.0))

            trend_note = ""
            if abs(home_slope - away_slope) > 3:
                if home_slope > away_slope:
                    trend_note = f" (Home improving: +{home_slope:.1f} pts/game)"
                else:
                    trend_note = f" (Away improving: +{away_slope:.1f} pts/game)"

            scoring_exp = (
                f"Scoring (ewma): home {home_sc:.1f} (slope {home_slope:+.1f}) "
                f"vs away {away_sc:.1f} (slope {away_slope:+.1f}); "
                f"P(home)={scoring_prob:.1%}.{trend_note}"
            )
        home_sigs.append(Signal("scoring", scoring_prob,
                                self._get_signal_weight("head_to_head", "scoring", 0.25),
                                scoring_reliability, scoring_exp))

        # ── Signal 4: H2H historical record (v2 NEW) ──────────────────────
        h2h_n = int(float(features.get("h2h_n_games") or 0))
        if h2h_n >= 3:
            # Use recent-weighted H2H win rate when available
            h2h_wr = float(features.get("h2h_home_win_rate_recent")
                           or features.get("h2h_home_win_rate") or 0.5)
            h2h_diff = float(features.get("h2h_avg_score_diff") or 0.0)
            # Logistic transform: home team dominates H2H → higher probability
            h2h_prob = float(np.clip(
                1.0 / (1.0 + np.exp(-(h2h_wr - 0.5) * 6.0 + h2h_diff / 40.0)),
                0.10, 0.90
            ))
            # Reliability scales with H2H sample size
            h2h_reliability = float(np.clip((h2h_n - 2) / 8.0, 0.1, 0.85))
            h2h_exp = (
                f"H2H ({h2h_n} meetings): home win rate {h2h_wr:.0%} "
                f"(avg margin {h2h_diff:+.1f}); P(home)={h2h_prob:.1%}"
            )
            home_sigs.append(Signal("h2h", h2h_prob,
                                    self._get_signal_weight("head_to_head", "h2h", 0.20),
                                    h2h_reliability, h2h_exp))

        # ── Signal 5: Market consensus ────────────────────────────────────
        if market_probs and "home_win" in market_probs:
            mkt_p = float(np.clip(market_probs["home_win"], 0.05, 0.95))
            home_sigs.append(Signal(
                "market", mkt_p,
                self._get_signal_weight("head_to_head", "market", 0.20),
                0.90,
                f"Bookmaker consensus: P(home)={mkt_p:.1%} (vig-adjusted)",
            ))

        # ── Build results ─────────────────────────────────────────────────
        home_result = self._build_result(home_sigs, data_completeness, corr_table=self._SIGNAL_CORRELATIONS)
        away_sigs = [
            Signal(s.name, float(np.clip(1.0 - s.probability, 0.05, 0.95)),
                   s.weight, s.reliability, s.explanation)
            for s in home_sigs
        ]
        away_result = self._build_result(away_sigs, data_completeness, corr_table=self._SIGNAL_CORRELATIONS)

        logger.debug(
            "SignalEngine H2H: home={:.1%} (agree={:.0%}, var={:.4f}, n_sig={}) | away={:.1%}",
            home_result.consensus_probability,
            home_result.signal_agreement,
            home_result.prediction_variance,
            home_result.n_active_signals,
            away_result.consensus_probability,
        )

        return home_result, away_result

    # ------------------------------------------------------------------
    # Player disposals signals
    # ------------------------------------------------------------------

    # Player signal correlations: short/medium form share underlying data
    _PLAYER_SIGNAL_CORRELATIONS = {
        ("short_form_ewma", "medium_form"): 0.55,   # both from same rolling window
        ("short_form_ewma", "slope_trend"): 0.35,   # slope derived from same vals
        ("medium_form", "slope_trend"): 0.30,
        ("venue_split", "short_form_ewma"): 0.20,   # venue split from same games
        ("matchup", "short_form_ewma"): 0.10,       # partially correlated environment
        ("ml_model", "medium_form"): 0.25,          # ML trained on same features
        ("ml_model", "short_form_ewma"): 0.20,
    }

    def compute_player_disposal_signals(
        self,
        player_features: Dict,
        line: float,
        model_over_prob: Optional[float] = None,
        market_probs: Optional[Dict[str, float]] = None,
    ) -> Tuple[SignalResult, SignalResult]:
        """
        Multi-signal consensus for player disposals over/under.

        v3 additions:
          - Venue split signal (home/away ewma)
          - Position baseline signal (vs position population)
          - Matchup advantage signal (recency-weighted opp allowance)
          - Learned signal weights from SignalWeightStore
          - Player-specific correlation penalties
          - Role-stability reduces all signal reliabilities uniformly
        """
        n_games = int(float(player_features.get("n_games") or 0))
        role_stability = float(player_features.get("role_stability") or 1.0)
        role_transition = int(player_features.get("role_transition_flag") or 0)
        # Transition flag → immediate reliability penalty on top of stability
        transition_penalty = 0.70 if role_transition else 1.0
        data_completeness = (
            float(np.clip(n_games / 15.0, 0.0, 1.0)) * role_stability * transition_penalty
        )

        over_sigs: List[Signal] = []

        ewma_val = float(player_features.get("roll_ewma") or 0.0)
        mean_3 = float(player_features.get("roll_mean_3") or 0.0)
        mean_5 = float(player_features.get("roll_mean_5") or 0.0)
        std_5 = float(player_features.get("roll_std_5") or 5.0)
        roll_slope = float(player_features.get("roll_slope")
                           or player_features.get("form_trend") or 0.0)
        roll_iqr = float(player_features.get("roll_iqr") or 0.3)
        ewma_venue = float(player_features.get("roll_ewma_venue") or ewma_val or mean_5)
        home_away_split = float(player_features.get("home_away_split") or 0.0)
        pos_baseline_z = float(player_features.get("position_baseline_z") or 0.0)
        pos_mean_allow = float(player_features.get("position_mean_allowance") or mean_5 or 20.0)
        matchup_adv = float(player_features.get("matchup_advantage") or 0.0)

        # Effective std: IQR-based is more robust; blend with rolling std
        roll_iqr_std = roll_iqr * max(mean_5, 1.0)
        eff_std = max(4.0, 0.5 * std_5 + 0.5 * roll_iqr_std)
        from scipy.stats import norm as _norm

        # ── Signal 1: Short-form EWMA (~3 games, decay-weighted) ──────────
        short_mean = ewma_val if ewma_val > 0 and n_games >= 3 else mean_3
        if short_mean > 0 and n_games >= 3:
            short_prob = float(np.clip(_norm.sf(line, loc=short_mean, scale=eff_std), 0.05, 0.95))
            s1_rel = float(np.clip(n_games / 8.0, 0.2, 1.0)) * role_stability * transition_penalty
            over_sigs.append(Signal(
                "short_form_ewma", short_prob,
                self._get_signal_weight("player_disposals", "short_form", 0.30),
                s1_rel,
                f"EWMA {short_mean:.1f} vs line {line}; P(over)={short_prob:.1%} (std≈{eff_std:.1f})",
            ))

        # ── Signal 2: Medium-form (5-game mean) ───────────────────────────
        if mean_5 > 0 and n_games >= 5:
            med_prob = float(np.clip(_norm.sf(line, loc=mean_5, scale=eff_std), 0.05, 0.95))
            s2_rel = float(np.clip(n_games / 12.0, 0.2, 1.0)) * role_stability * transition_penalty
            over_sigs.append(Signal(
                "medium_form", med_prob,
                self._get_signal_weight("player_disposals", "medium_form", 0.20),
                s2_rel,
                f"5-game avg {mean_5:.1f} vs line {line}; P(over)={med_prob:.1%}",
            ))

        # ── Signal 3: Slope-adjusted trend ────────────────────────────────
        if mean_5 > 0 and n_games >= 5 and abs(roll_slope) > 0.3:
            projected = mean_5 + roll_slope * 1.5
            trend_prob = float(np.clip(_norm.sf(line, loc=projected, scale=eff_std), 0.05, 0.95))
            s3_rel = float(np.clip(n_games / 15.0, 0.1, 0.75)) * role_stability * transition_penalty
            direction = "↑ improving" if roll_slope > 0 else "↓ declining"
            over_sigs.append(Signal(
                "slope_trend", trend_prob,
                self._get_signal_weight("player_disposals", "trend", 0.12),
                s3_rel,
                f"Slope {roll_slope:+.1f}/game → projected {projected:.1f} ({direction}); "
                f"P(over)={trend_prob:.1%}",
            ))

        # ── Signal 4: Venue split (home vs away historical ewma) ──────────
        if abs(home_away_split) > 1.5 and n_games >= 4:
            venue_prob = float(np.clip(_norm.sf(line, loc=ewma_venue, scale=eff_std), 0.05, 0.95))
            # Reliability scales with |split| — strong splits are informative
            s4_rel = float(np.clip(abs(home_away_split) / 5.0, 0.15, 0.65)) * role_stability
            venue_note = "at home" if home_away_split > 0 and player_features.get("is_home_game") else "away"
            over_sigs.append(Signal(
                "venue_split", venue_prob,
                self._get_signal_weight("player_disposals", "venue_split", 0.12),
                s4_rel,
                f"Venue ({venue_note}) ewma {ewma_venue:.1f} vs line {line}; "
                f"split={home_away_split:+.1f}; P(over)={venue_prob:.1%}",
            ))

        # ── Signal 5: Matchup vs opponent position-specific allowance ─────
        opp_allow = float(player_features.get("opp_pos_disposals_allowed")
                          or player_features.get("opp_disposals_allowed_mean") or mean_5)
        if opp_allow > 0 and n_games >= 3:
            matchup_prob = float(np.clip(_norm.sf(line, loc=opp_allow, scale=eff_std), 0.05, 0.95))
            # Reliability: higher when opp allowance is clearly generous/restrictive
            s5_rel = float(np.clip(0.40 + abs(matchup_adv) / 8.0, 0.15, 0.70))
            matchup_note = (
                f"generous (+{matchup_adv:.1f})" if matchup_adv > 2
                else (f"restrictive ({matchup_adv:.1f})" if matchup_adv < -2 else "neutral")
            )
            over_sigs.append(Signal(
                "matchup", matchup_prob,
                self._get_signal_weight("player_disposals", "matchup", 0.15),
                s5_rel,
                f"Opp allows {opp_allow:.1f} ({matchup_note}); P(over)={matchup_prob:.1%}",
            ))

        # ── Signal 6: Position baseline plausibility ──────────────────────
        # Low |z| → player is well-anchored to position norms → reliable signal
        # High |z| → outlier player or unusual prediction → lower weight
        if pos_mean_allow > 0 and n_games >= 3:
            baseline_prob = float(np.clip(_norm.sf(line, loc=pos_mean_allow, scale=eff_std), 0.05, 0.95))
            # Weight diminishes for extreme outliers (|z| > 1.5 player)
            z_pen = float(np.clip(1.0 - abs(pos_baseline_z) / 3.0, 0.20, 0.80))
            s6_rel = float(np.clip(n_games / 20.0, 0.1, 0.60)) * z_pen
            over_sigs.append(Signal(
                "position_baseline", baseline_prob,
                self._get_signal_weight("player_disposals", "position_baseline", 0.08),
                s6_rel,
                f"Position avg {pos_mean_allow:.1f} vs line {line} (z={pos_baseline_z:+.2f}); "
                f"P(over)={baseline_prob:.1%}",
            ))

        # ── Signal 7: Disposal quality / efficiency ───────────────────────
        # Uses DSG secondary stat rates to adjust projected disposal output.
        # High eff_disposal_rate + low clanger_rate + high contested_rate → upside.
        _AVG_EFF_RATE = 0.65
        _AVG_CLANGER_RATE = 0.08
        _AVG_CONTESTED_RATE = 0.45
        eff_rate = float(player_features.get("player_eff_disposal_rate") or 0.0)
        clanger_rate_val = float(player_features.get("player_clanger_rate") or 0.0)
        contested_rate_val = float(player_features.get("player_contested_rate") or 0.0)
        if eff_rate > 0 and n_games >= 5 and mean_5 > 0:
            eff_adj = (eff_rate - _AVG_EFF_RATE) / 0.15
            clang_adj = -((clanger_rate_val - _AVG_CLANGER_RATE) / 0.05)
            cont_adj = (contested_rate_val - _AVG_CONTESTED_RATE) / 0.15
            quality_score = float(np.clip(eff_adj * 0.5 + clang_adj * 0.3 + cont_adj * 0.2, -1.0, 1.0))
            quality_adjusted_mean = mean_5 + quality_score * 2.0
            quality_prob = float(np.clip(_norm.sf(line, loc=quality_adjusted_mean, scale=eff_std), 0.05, 0.95))
            s7_rel = float(np.clip(n_games / 10.0, 0.2, 0.65)) * role_stability * transition_penalty
            direction = "quality↑" if quality_score > 0.2 else ("quality↓" if quality_score < -0.2 else "neutral")
            over_sigs.append(Signal(
                "disposal_quality", quality_prob,
                self._get_signal_weight("player_disposals", "quality", 0.08),
                s7_rel,
                f"Quality: eff={eff_rate:.2f}, clang={clanger_rate_val:.2f}, "
                f"cont={contested_rate_val:.2f} → {direction} proj={quality_adjusted_mean:.1f}; "
                f"P(over)={quality_prob:.1%}",
            ))

        # ── Signal 8: ML model ────────────────────────────────────────────
        if model_over_prob is not None:
            ml_prob = float(np.clip(model_over_prob, 0.05, 0.95))
            ml_rel = float(np.clip(n_games / 15.0, 0.1, 0.9)) * role_stability * transition_penalty
            over_sigs.append(Signal(
                "ml_model", ml_prob,
                self._get_signal_weight("player_disposals", "ml", 0.10),
                ml_rel,
                f"XGBoost: P(over {line})={ml_prob:.1%}",
            ))

        # ── Signal 8: Market consensus ────────────────────────────────────
        sel_key = f"player_over_{line}"
        if market_probs and sel_key in market_probs:
            mkt_p = float(np.clip(market_probs[sel_key], 0.05, 0.95))
            over_sigs.append(Signal(
                "market", mkt_p,
                self._get_signal_weight("player_disposals", "market", 0.10),
                0.85,
                f"Market: P(over {line})={mkt_p:.1%} (vig-adjusted)",
            ))

        if not over_sigs:
            empty = SignalResult(
                signals=[], consensus_probability=0.5,
                signal_agreement=0.0, prediction_variance=0.0,
                data_completeness=data_completeness, n_active_signals=0,
                top_factors=["No player data available"],
                explanation="No player signals computed.",
                prediction_low=0.2, prediction_high=0.8,
            )
            return empty, empty

        over_result = self._build_result(
            over_sigs, data_completeness,
            corr_table=self._PLAYER_SIGNAL_CORRELATIONS,
        )
        under_sigs = [
            Signal(s.name, float(np.clip(1.0 - s.probability, 0.05, 0.95)),
                   s.weight, s.reliability, s.explanation)
            for s in over_sigs
        ]
        under_result = self._build_result(
            under_sigs, data_completeness,
            corr_table=self._PLAYER_SIGNAL_CORRELATIONS,
        )
        return over_result, under_result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    # Known inter-signal correlations (not true independence — penalise redundancy)
    _SIGNAL_CORRELATIONS = {
        ("elo", "form"): 0.40,
        ("form", "scoring"): 0.30,
        ("h2h", "elo"): 0.30,
        ("market", "elo"): 0.25,
        ("market", "form"): 0.20,
    }

    def _compute_effective_weights(
        self,
        signals: List[Signal],
        corr_table: Optional[Dict] = None,
    ) -> List[float]:
        """
        Reduce weights for signals that share information with others.
        corr_table: override the default H2H correlation table (e.g. for player signals).
        """
        if corr_table is None:
            corr_table = self._SIGNAL_CORRELATIONS
        names = [s.name for s in signals]
        weights = [s.weight * s.reliability for s in signals]
        total = sum(weights)
        if total < 1e-6:
            return weights

        effective = list(weights)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                pair = tuple(sorted([names[i], names[j]]))
                r = corr_table.get(pair, 0.0)
                if r > 0:
                    penalty = r * min(effective[i], effective[j]) * 0.3
                    effective[i] = max(0.0, effective[i] - penalty)
                    effective[j] = max(0.0, effective[j] - penalty)

        total_eff = sum(effective)
        if total_eff > 1e-6:
            effective = [e / total_eff for e in effective]
        return effective

    def _build_result(
        self,
        signals: List[Signal],
        data_completeness: float,
        corr_table: Optional[Dict] = None,
    ) -> SignalResult:
        """Compute correlation-penalised consensus, agreement metrics, prediction interval."""
        if not signals:
            return SignalResult(
                signals=[], consensus_probability=0.5,
                signal_agreement=0.0, prediction_variance=0.0,
                data_completeness=data_completeness, n_active_signals=0,
                top_factors=["Insufficient data"],
                explanation="No signals available.",
                prediction_low=0.2, prediction_high=0.8,
            )

        eff_weights = self._compute_effective_weights(signals, corr_table=corr_table)
        total_w = sum(eff_weights)
        n_active = sum(1 for w in eff_weights if w > 0.02)

        if total_w < 0.02:
            return SignalResult(
                signals=signals,
                consensus_probability=0.5,
                signal_agreement=0.0,
                prediction_variance=float(np.var([s.probability for s in signals])),
                data_completeness=data_completeness,
                n_active_signals=0,
                top_factors=["All signals have near-zero reliability"],
                explanation="No reliable signals — insufficient data.",
                prediction_low=0.2, prediction_high=0.8,
            )

        norm_w = [w / total_w for w in eff_weights]
        probs = [s.probability for s in signals]
        consensus = float(np.clip(np.dot(norm_w, probs), 0.05, 0.95))

        # Agreement: 1 − (std / scale)
        prob_std = float(np.std(probs))
        signal_agreement = float(np.clip(1.0 - prob_std / self._AGREEMENT_SCALE, 0.0, 1.0))
        prediction_variance = float(np.var(probs))

        # v2: prediction interval — weighted percentiles of signal distribution
        # Use signal probabilities weighted by reliability as a distribution
        # 10th / 90th percentile gives an 80% confidence interval
        if len(probs) >= 3:
            sorted_probs = sorted(probs)
            pred_low = float(np.percentile(sorted_probs, 10))
            pred_high = float(np.percentile(sorted_probs, 90))
        else:
            pred_low = max(0.05, consensus - 1.5 * prob_std)
            pred_high = min(0.95, consensus + 1.5 * prob_std)

        ranked = sorted(
            [(s, w) for s, w in zip(signals, eff_weights) if w > 0.02],
            key=lambda x: x[1], reverse=True,
        )
        top_factors = [s.explanation for s, _ in ranked[:3]]

        explanation = (
            f"Consensus: {consensus:.1%} from {n_active} signals "
            f"[{', '.join(s.name + '=' + f'{s.probability:.1%}' for s in signals)}]. "
            f"Agreement: {signal_agreement:.0%}, variance: {prediction_variance:.4f}. "
            f"80% CI: [{pred_low:.1%}, {pred_high:.1%}]."
        )

        return SignalResult(
            signals=signals,
            consensus_probability=consensus,
            signal_agreement=signal_agreement,
            prediction_variance=prediction_variance,
            data_completeness=data_completeness,
            n_active_signals=n_active,
            top_factors=top_factors,
            explanation=explanation,
            prediction_low=round(pred_low, 4),
            prediction_high=round(pred_high, 4),
        )

    @staticmethod
    def _data_completeness(features: Dict, n_games: int) -> float:
        """
        Score data quality 0–1.
          0.4 from game-count (capped at 10+)
          0.3 from feature presence
          0.2 from Elo separation (clear favourite = more predictable)
          0.1 from H2H data availability (v2)
        """
        score = float(np.clip(n_games / 10.0, 0.0, 0.4))

        # Check presence of key predictive features; fall back through three tiers:
        #   1. diff_roll_* (pruned feature set used since v3.1)
        #   2. home_/away_ ewma versions
        #   3. home_/away_ mean versions (legacy fallback)
        key_feats_diff = [
            "elo_win_prob_home",
            "diff_roll_score_ewma", "diff_roll_inside_50s_mean",
            "diff_roll_clearances_mean", "diff_roll_win_rate_ewma",
        ]
        key_feats_ha_ewma = [
            "elo_win_prob_home",
            "home_roll_score_ewma", "away_roll_score_ewma",
            "home_roll_win_rate_ewma", "away_roll_win_rate_ewma",
        ]
        fallback_feats = [
            "elo_win_prob_home",
            "home_roll_score_mean", "away_roll_score_mean",
            "home_roll_win_rate", "away_roll_win_rate",
        ]
        if any(features.get(f) for f in key_feats_diff):
            check_feats = key_feats_diff
        elif any(features.get(f) for f in key_feats_ha_ewma):
            check_feats = key_feats_ha_ewma
        else:
            check_feats = fallback_feats

        n_present = sum(
            1 for f in check_feats
            if features.get(f) is not None and float(features.get(f) or 0) != 0.0
        )
        score += (n_present / len(check_feats)) * 0.3

        elo_diff = abs(float(features.get("elo_diff") or 0.0))
        score += float(np.clip(elo_diff / 200.0, 0.0, 0.2))

        # H2H bonus (v2)
        h2h_n = int(float(features.get("h2h_n_games") or 0))
        score += float(np.clip(h2h_n / 10.0, 0.0, 0.1))

        # Data freshness: stale data reduces confidence in rolling features
        freshness = float(features.get("data_freshness_score", 1.0) or 1.0)
        score = score * (0.70 + 0.30 * freshness)

        return float(np.clip(score, 0.0, 1.0))
