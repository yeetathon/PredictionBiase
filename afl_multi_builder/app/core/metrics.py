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
    Composite confidence score (0-100).
    Higher when: edge is large, model agrees with market direction,
    sufficient data available.
    """
    # Edge component: scaled 0-40
    edge_score = min(40.0, max(0.0, edge * 400))

    # Agreement component: high when model and market are aligned
    agreement = 1.0 - abs(model_prob - market_prob) * 2
    agreement_score = max(0.0, agreement * 30)

    # Data sufficiency component (0-30)
    data_score = min(30.0, n_historical_games * 0.5)

    return round(edge_score + agreement_score + data_score, 1)
