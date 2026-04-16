"""
Multi-signal prediction engine.

Computes 4 independent signals for each AFL match outcome and measures
their agreement. High agreement → higher confidence. Disagreement →
lower confidence, or outright rejection at the quality gate.

Signals (in order of independence):
  1. Elo          — long-run team strength rating (uses elo_win_prob_home)
  2. Form         — rolling win-rate differential (no scores, no Elo)
  3. Scoring      — recent scoring power differential + trend
  4. Market       — bookmaker consensus (vig-adjusted, if available)

Each signal has a base weight and a reliability score (0–1) that
down-weights it automatically when data is sparse or missing.

The consensus probability is the reliability-weighted average.
Signal agreement = 1 − normalised std of probabilities across signals.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from loguru import logger


@dataclass
class Signal:
    name: str
    probability: float      # P(home win) from this signal
    weight: float           # base contribution weight
    reliability: float      # 0–1; down-weights uncertain/sparse signals
    explanation: str


@dataclass
class SignalResult:
    """All signal outputs for one leg direction (home win or away win)."""
    signals: List[Signal]
    consensus_probability: float   # reliability-weighted combination
    signal_agreement: float        # 0–1; 1 = perfect agreement across signals
    prediction_variance: float     # variance of raw probabilities across signals
    data_completeness: float       # 0–1; overall data quality for this fixture
    n_active_signals: int          # signals with reliability > 0.1
    top_factors: List[str]         # human-readable explanation bullets
    explanation: str               # summary line


class SignalEngine:
    """
    Computes independent prediction signals and a consensus probability.

    Usage:
        engine = SignalEngine()
        home_result, away_result = engine.compute_h2h_signals(features_dict, market_probs)

    features_dict keys (from fixture row of TeamFeatureEngineer.build_features()):
        elo_win_prob_home, elo_home_pre, elo_away_pre, elo_diff,
        home_roll_win_rate, away_roll_win_rate,
        home_roll_home_win_rate, away_roll_away_win_rate,
        home_roll_score_mean, away_roll_score_mean,
        home_roll_score_std, away_roll_score_std,
        home_roll_n_games, away_roll_n_games,
        home_form_trend_score, away_form_trend_score,
        home_rest_days, away_rest_days,
        diff_rest_days

    market_probs keys: {"home_win": 0.52, "away_win": 0.48}  (vig-adjusted)
    """

    # Normalisation constant for signal agreement: std above this → agreement → 0
    _AGREEMENT_SCALE = 0.15

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
        elo_prob = float(features.get("elo_win_prob_home") or 0.5)
        elo_prob = float(np.clip(elo_prob, 0.05, 0.95))
        elo_diff = float(features.get("elo_diff") or 0.0)
        elo_home = float(features.get("elo_home_pre") or 1500.0)
        elo_away = float(features.get("elo_away_pre") or 1500.0)
        # Reliability: full at 20 completed games; zero at 0
        elo_reliability = float(np.clip(n_games / 20.0, 0.1, 1.0))

        if abs(elo_diff) < 15:
            elo_exp = (
                f"Elo: near-even matchup (diff={elo_diff:+.0f} pts), "
                f"P(home)={elo_prob:.1%}"
            )
        else:
            stronger = "Home" if elo_diff > 0 else "Away"
            elo_exp = (
                f"Elo: {stronger} team stronger by {abs(elo_diff):.0f} pts "
                f"({elo_home:.0f} vs {elo_away:.0f}), P(home)={elo_prob:.1%}"
            )
        home_sigs.append(Signal(
            name="elo",
            probability=elo_prob,
            weight=0.30,
            reliability=elo_reliability,
            explanation=elo_exp,
        ))

        # ── Signal 2: Form (win-rate based, no scores) ────────────────────
        home_wr = float(features.get("home_roll_win_rate") or 0.5)
        away_wr = float(features.get("away_roll_win_rate") or 0.5)
        # Venue-specific win rates, blended with overall
        home_home_wr = float(features.get("home_roll_home_win_rate") or home_wr)
        away_away_wr = float(features.get("away_roll_away_win_rate") or away_wr)
        # 60% overall + 40% venue-specific
        home_form = 0.6 * home_wr + 0.4 * home_home_wr
        away_form = 0.6 * away_wr + 0.4 * away_away_wr
        form_diff = home_form - away_form           # –1 to +1
        # Logistic transform: diff of ±0.3 → prob of ~0.65/0.35
        form_prob = float(1.0 / (1.0 + np.exp(-form_diff * 5.0)))
        form_prob = float(np.clip(form_prob, 0.10, 0.90))
        form_reliability = float(np.clip(n_games / 12.0, 0.05, 1.0))
        form_exp = (
            f"Form: home {home_wr:.0%} WR (at-home {home_home_wr:.0%}) vs "
            f"away {away_wr:.0%} WR (away {away_away_wr:.0%}); "
            f"P(home)={form_prob:.1%}"
        )
        home_sigs.append(Signal(
            name="form",
            probability=form_prob,
            weight=0.25,
            reliability=form_reliability,
            explanation=form_exp,
        ))

        # ── Signal 3: Scoring power + trend ──────────────────────────────
        home_sc = float(features.get("home_roll_score_mean") or 0.0)
        away_sc = float(features.get("away_roll_score_mean") or 0.0)
        home_sc_std = float(features.get("home_roll_score_std") or 15.0)
        away_sc_std = float(features.get("away_roll_score_std") or 15.0)
        home_trend = float(features.get("home_form_trend_score") or 0.0)
        away_trend = float(features.get("away_form_trend_score") or 0.0)

        scoring_prob = 0.5
        scoring_reliability = 0.0
        scoring_exp = "Scoring: insufficient score data"

        if home_sc > 0 and away_sc > 0:
            # Trend-adjusted expected scores
            home_adj = home_sc + home_trend * 0.3
            away_adj = away_sc + away_trend * 0.3
            sc_diff = home_adj - away_adj
            # Combined std of score differential (typical AFL: ~30–40 pts spread)
            combined_std = float(np.sqrt(home_sc_std ** 2 + away_sc_std ** 2))
            combined_std = max(20.0, combined_std)
            from scipy.stats import norm as _norm
            scoring_prob = float(_norm.cdf(sc_diff / combined_std))
            scoring_prob = float(np.clip(scoring_prob, 0.10, 0.90))
            scoring_reliability = float(np.clip(n_games / 10.0, 0.05, 1.0))

            trend_note = ""
            if abs(home_trend - away_trend) > 8:
                if home_trend > away_trend:
                    trend_note = f" (Home on upward trend: {home_trend:+.1f} pts)"
                else:
                    trend_note = f" (Away on upward trend: {away_trend:+.1f} pts)"

            scoring_exp = (
                f"Scoring: home avg {home_sc:.1f} (trend {home_trend:+.1f}) vs "
                f"away avg {away_sc:.1f} (trend {away_trend:+.1f}); "
                f"P(home)={scoring_prob:.1%}.{trend_note}"
            )

        home_sigs.append(Signal(
            name="scoring",
            probability=scoring_prob,
            weight=0.25,
            reliability=scoring_reliability,
            explanation=scoring_exp,
        ))

        # ── Signal 4: Market consensus ────────────────────────────────────
        if market_probs and "home_win" in market_probs:
            mkt_p = float(np.clip(market_probs["home_win"], 0.05, 0.95))
            home_sigs.append(Signal(
                name="market",
                probability=mkt_p,
                weight=0.20,
                reliability=0.90,   # bookmakers are well-calibrated on average
                explanation=f"Bookmaker consensus: P(home)={mkt_p:.1%} (vig-adjusted)",
            ))

        # ── Build results ─────────────────────────────────────────────────
        home_result = self._build_result(home_sigs, data_completeness)

        # Away is the complement of every home signal
        away_sigs = [
            Signal(
                name=s.name,
                probability=float(np.clip(1.0 - s.probability, 0.05, 0.95)),
                weight=s.weight,
                reliability=s.reliability,
                explanation=s.explanation,
            )
            for s in home_sigs
        ]
        away_result = self._build_result(away_sigs, data_completeness)

        logger.debug(
            "SignalEngine H2H: home={:.1%} (agree={:.0%}, var={:.4f}) | "
            "away={:.1%} (agree={:.0%})",
            home_result.consensus_probability,
            home_result.signal_agreement,
            home_result.prediction_variance,
            away_result.consensus_probability,
            away_result.signal_agreement,
        )

        return home_result, away_result

    # ------------------------------------------------------------------
    # Player disposals signals
    # ------------------------------------------------------------------

    def compute_player_disposal_signals(
        self,
        player_features: Dict,
        line: float,
        model_over_prob: Optional[float] = None,
        market_probs: Optional[Dict[str, float]] = None,
    ) -> Tuple[SignalResult, SignalResult]:
        """
        Compute signals for a player disposals over/under leg.
        Returns (over_result, under_result).

        player_features keys:
            roll_mean_3, roll_mean_5, roll_mean_10,
            roll_std_5, form_trend, consistency_cv, n_games
        """
        n_games = int(float(player_features.get("n_games") or 0))
        data_completeness = float(np.clip(n_games / 15.0, 0.0, 1.0))

        over_sigs: List[Signal] = []

        mean_3 = float(player_features.get("roll_mean_3") or 0.0)
        mean_5 = float(player_features.get("roll_mean_5") or 0.0)
        mean_10 = float(player_features.get("roll_mean_10") or mean_5)
        std_5 = float(player_features.get("roll_std_5") or 5.0)
        trend = float(player_features.get("form_trend") or 0.0)
        cv = float(player_features.get("consistency_cv") or 0.3)

        # ── Signal 1: Short-run form (3-game average) ─────────────────────
        if mean_3 > 0 and n_games >= 3:
            from scipy.stats import norm as _norm
            short_prob = float(_norm.sf(line, loc=mean_3, scale=max(4.0, std_5)))
            short_prob = float(np.clip(short_prob, 0.05, 0.95))
            s1_rel = float(np.clip(n_games / 8.0, 0.2, 1.0))
            over_sigs.append(Signal(
                name="short_form",
                probability=short_prob,
                weight=0.35,
                reliability=s1_rel,
                explanation=(
                    f"3-game avg {mean_3:.1f} vs line {line}; "
                    f"P(over)={short_prob:.1%} (std={std_5:.1f})"
                ),
            ))

        # ── Signal 2: Medium-run form (5-game average) ────────────────────
        if mean_5 > 0 and n_games >= 5:
            from scipy.stats import norm as _norm
            med_prob = float(_norm.sf(line, loc=mean_5, scale=max(4.0, std_5)))
            med_prob = float(np.clip(med_prob, 0.05, 0.95))
            s2_rel = float(np.clip(n_games / 12.0, 0.2, 1.0))
            over_sigs.append(Signal(
                name="medium_form",
                probability=med_prob,
                weight=0.30,
                reliability=s2_rel,
                explanation=(
                    f"5-game avg {mean_5:.1f} vs line {line}; "
                    f"P(over)={med_prob:.1%}"
                ),
            ))

        # ── Signal 3: Trend-adjusted (form trend applied to 5-game avg) ───
        if mean_5 > 0 and n_games >= 5:
            from scipy.stats import norm as _norm
            trend_adj_mean = mean_5 + trend * 0.4
            trend_prob = float(_norm.sf(line, loc=trend_adj_mean, scale=max(4.0, std_5)))
            trend_prob = float(np.clip(trend_prob, 0.05, 0.95))
            s3_rel = float(np.clip(n_games / 15.0, 0.1, 0.8))
            over_sigs.append(Signal(
                name="trend",
                probability=trend_prob,
                weight=0.20,
                reliability=s3_rel,
                explanation=(
                    f"Trend-adjusted avg {trend_adj_mean:.1f} (trend={trend:+.1f}); "
                    f"P(over)={trend_prob:.1%}"
                ),
            ))

        # ── Signal 4: ML model prediction (if provided) ───────────────────
        if model_over_prob is not None:
            ml_prob = float(np.clip(model_over_prob, 0.05, 0.95))
            ml_rel = float(np.clip(n_games / 15.0, 0.1, 0.9))
            over_sigs.append(Signal(
                name="ml_model",
                probability=ml_prob,
                weight=0.15,
                reliability=ml_rel,
                explanation=f"XGBoost regression: P(over {line})={ml_prob:.1%}",
            ))

        # ── Signal 5: Market ──────────────────────────────────────────────
        sel_key = f"player_over_{line}"
        if market_probs and sel_key in market_probs:
            mkt_p = float(np.clip(market_probs[sel_key], 0.05, 0.95))
            over_sigs.append(Signal(
                name="market",
                probability=mkt_p,
                weight=0.15,
                reliability=0.85,
                explanation=f"Market: P(over {line})={mkt_p:.1%}",
            ))

        if not over_sigs:
            empty = SignalResult(
                signals=[], consensus_probability=0.5,
                signal_agreement=0.0, prediction_variance=0.0,
                data_completeness=data_completeness, n_active_signals=0,
                top_factors=["No player data available"],
                explanation="No player signals computed.",
            )
            return empty, empty

        over_result = self._build_result(over_sigs, data_completeness)
        under_sigs = [
            Signal(
                name=s.name,
                probability=float(np.clip(1.0 - s.probability, 0.05, 0.95)),
                weight=s.weight,
                reliability=s.reliability,
                explanation=s.explanation,
            )
            for s in over_sigs
        ]
        under_result = self._build_result(under_sigs, data_completeness)
        return over_result, under_result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_result(
        self, signals: List[Signal], data_completeness: float
    ) -> SignalResult:
        """
        Compute reliability-weighted consensus and agreement metrics.
        """
        if not signals:
            return SignalResult(
                signals=[],
                consensus_probability=0.5,
                signal_agreement=0.0,
                prediction_variance=0.0,
                data_completeness=data_completeness,
                n_active_signals=0,
                top_factors=["Insufficient data"],
                explanation="No signals available.",
            )

        # Effective weight = base_weight × reliability
        eff_weights = [s.weight * s.reliability for s in signals]
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
            )

        norm_w = [w / total_w for w in eff_weights]
        probs = [s.probability for s in signals]
        consensus = float(np.dot(norm_w, probs))
        consensus = float(np.clip(consensus, 0.05, 0.95))

        # Agreement: 1 − (std / scale); scale = 0.15
        prob_std = float(np.std(probs))
        signal_agreement = float(np.clip(1.0 - prob_std / self._AGREEMENT_SCALE, 0.0, 1.0))
        prediction_variance = float(np.var(probs))

        # Top factors: highest-weight active signals
        ranked = sorted(
            [(s, w) for s, w in zip(signals, eff_weights) if w > 0.02],
            key=lambda x: x[1], reverse=True,
        )
        top_factors = [s.explanation for s, _ in ranked[:3]]

        explanation = (
            f"Consensus: {consensus:.1%} from {n_active} signals "
            f"[{', '.join(s.name + '=' + f'{s.probability:.1%}' for s in signals)}]. "
            f"Agreement: {signal_agreement:.0%}, variance: {prediction_variance:.4f}."
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
        )

    @staticmethod
    def _data_completeness(features: Dict, n_games: int) -> float:
        """
        Score data quality 0–1.
          0.5 from game-count  (capped at 10+ games)
          0.3 from feature presence
          0.2 from Elo separation (clear favourite = more predictable)
        """
        score = float(np.clip(n_games / 10.0, 0.0, 0.5))

        key_feats = [
            "elo_win_prob_home",
            "home_roll_score_mean", "away_roll_score_mean",
            "home_roll_win_rate", "away_roll_win_rate",
        ]
        n_present = sum(
            1 for f in key_feats
            if features.get(f) is not None and float(features.get(f) or 0) != 0.0
        )
        score += (n_present / len(key_feats)) * 0.3

        elo_diff = abs(float(features.get("elo_diff") or 0.0))
        score += float(np.clip(elo_diff / 200.0, 0.0, 0.2))

        return float(np.clip(score, 0.0, 1.0))
