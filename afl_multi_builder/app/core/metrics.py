"""Core metrics: Brier score, log loss, calibration, EV calculations."""
import numpy as np
from typing import List, Optional
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.calibration import calibration_curve


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Brier score (lower is better, 0=perfect, 0.25=baseline)."""
    return float(brier_score_loss(y_true, y_prob))


def compute_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute log loss (lower is better)."""
    # Clip probabilities to avoid log(0)
    y_prob_clipped = np.clip(y_prob, 1e-7, 1 - 1e-7)
    return float(log_loss(y_true, y_prob_clipped))


def compute_calibration_bins(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Compute probability bin analysis for calibration reporting.
    Returns dict with bin midpoints, mean predicted probabilities,
    fraction of positives, and counts.
    """
    fraction_of_positives, mean_predicted = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )
    return {
        "mean_predicted": mean_predicted.tolist(),
        "fraction_of_positives": fraction_of_positives.tolist(),
        "brier_score": compute_brier_score(y_true, y_prob),
        "log_loss": compute_log_loss(y_true, y_prob),
        "n_samples": int(len(y_true)),
    }


def implied_probability(decimal_odds: float) -> float:
    """Convert decimal odds to implied probability."""
    if decimal_odds <= 1.0:
        return 1.0
    return 1.0 / decimal_odds


def remove_vig(probs: List[float]) -> List[float]:
    """
    Remove bookmaker overround (vig) from a list of implied probabilities.
    Normalises so they sum to 1.0.
    """
    total = sum(probs)
    if total <= 0:
        return probs
    return [p / total for p in probs]


def compute_edge(model_prob: float, market_prob: float) -> float:
    """
    Edge = model probability - market (vig-removed) probability.
    Positive edge = model believes outcome is underpriced.
    """
    return model_prob - market_prob


def compute_ev(model_prob: float, decimal_odds: float) -> float:
    """
    Expected value per unit staked.
    EV = (model_prob * decimal_odds) - 1
    Positive EV = profitable long-run bet.
    """
    return (model_prob * decimal_odds) - 1.0


def compute_kelly_fraction(
    model_prob: float,
    decimal_odds: float,
    fraction: float = 0.25,
) -> float:
    """
    Fractional Kelly criterion stake size.
    Kelly = (model_prob * decimal_odds - 1) / (decimal_odds - 1)
    Returns fraction of bankroll to stake (clipped at 0).
    """
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    kelly = (model_prob * b - (1 - model_prob)) / b
    return max(0.0, kelly * fraction)


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Simple accuracy: predicted class matches actual."""
    return float(np.mean(y_true == y_pred))


def compute_roi(
    outcomes: np.ndarray,
    odds: np.ndarray,
    stake: float = 1.0,
) -> dict:
    """
    Compute ROI/yield metrics for a series of bets.
    outcomes: 1 if won, 0 if lost.
    odds: decimal odds for each bet.
    """
    total_staked = len(outcomes) * stake
    returns = np.where(outcomes == 1, odds * stake, 0.0)
    profit = returns.sum() - total_staked
    roi = profit / total_staked if total_staked > 0 else 0.0
    hit_rate = float(np.mean(outcomes))
    return {
        "n_bets": int(len(outcomes)),
        "total_staked": float(total_staked),
        "total_returns": float(returns.sum()),
        "profit": float(profit),
        "roi": float(roi),
        "hit_rate": float(hit_rate),
        "avg_odds": float(np.mean(odds)),
    }


def compute_confidence_score(
    model_prob: float,
    market_prob: float,
    edge: float,
    n_historical_games: int = 0,
) -> float:
    """
    Legacy confidence score (0-100). Kept for backward compatibility.
    Prefer compute_signal_aware_confidence() when signal data is available.
    Higher when: edge is large, model agrees with market direction,
    sufficient data available.
    """
    edge_score = min(40.0, max(0.0, edge * 400))
    agreement = 1.0 - abs(model_prob - market_prob) * 2
    agreement_score = max(0.0, agreement * 30)
    data_score = min(30.0, n_historical_games * 0.5)
    return round(edge_score + agreement_score + data_score, 1)


def compute_signal_aware_confidence(
    signal_agreement: float,
    edge: float,
    data_completeness: float,
    prediction_variance: float,
    n_active_signals: int = 0,
) -> float:
    """
    Signal-based confidence score (0-100).

    Rewards:
      - Signal agreement: all independent signals pointing the same way (0-40 pts)
      - Edge vs vig-adjusted market probability (0-30 pts)
      - Data completeness: sufficient historical games and feature coverage (0-20 pts)

    Penalises:
      - High prediction variance across signals (up to 25 pts)
      - Fewer than 2 active signals (10 pt penalty)

    Typical values:
      4 agreeing signals (agree=0.90), 8% edge, completeness=0.8, var=0.01
        → 36 + 16 + 16 − 4 = 64 pts (Medium-High)
      3 disagreeing signals (agree=0.45), 5% edge, completeness=0.5, var=0.04
        → 18 + 10 + 10 − 16 = 22 pts (Low — likely rejected)
    """
    # Signal agreement component (0-40)
    agreement_score = float(signal_agreement) * 40.0

    # Edge component (0-30): scaled so 5% edge ≈ 10 pts, 15% edge ≈ 30 pts
    edge_score = min(30.0, max(0.0, float(edge) * 200.0))

    # Data quality component (0-20)
    quality_score = float(data_completeness) * 20.0

    # Variance penalty: std 0.05 → var 0.0025 → 1 pt; std 0.25 → var 0.0625 → 25 pts
    variance_penalty = min(25.0, float(prediction_variance) * 400.0)

    # Thin signal penalty (< 2 active signals)
    thin_penalty = 10.0 if n_active_signals < 2 else 0.0

    raw = agreement_score + edge_score + quality_score - variance_penalty - thin_penalty
    return round(max(0.0, min(100.0, raw)), 1)


def compute_signal_agreement(probabilities: list) -> float:
    """
    Compute 0-1 agreement score from a list of signal probabilities.
    1.0 = all signals identical; 0.0 = maximum disagreement (std ≥ 0.15).
    """
    if len(probabilities) < 2:
        return 0.0
    std = float(np.std(probabilities))
    return float(np.clip(1.0 - std / 0.15, 0.0, 1.0))


def compute_prediction_variance(probabilities: list) -> float:
    """Variance of probabilities across independent signals."""
    if len(probabilities) < 2:
        return 0.0
    return float(np.var(probabilities))
