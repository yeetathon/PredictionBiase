"""General utility functions."""
import hashlib
import json
from datetime import datetime, date
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division that returns default if denominator is zero."""
    if denominator == 0:
        return default
    return numerator / denominator


def rolling_mean(values: List[float], window: int) -> List[float]:
    """Compute rolling mean with minimum periods = 1."""
    s = pd.Series(values)
    return s.rolling(window=window, min_periods=1).mean().tolist()


def rolling_std(values: List[float], window: int, min_periods: int = 3) -> List[float]:
    """Compute rolling standard deviation."""
    s = pd.Series(values)
    return s.rolling(window=window, min_periods=min_periods).std().fillna(0).tolist()


def compute_exponential_moving_average(
    values: List[float], span: int = 5
) -> List[float]:
    """Compute exponential weighted moving average."""
    s = pd.Series(values)
    return s.ewm(span=span, adjust=False).mean().tolist()


def winsorize(
    values: np.ndarray, lower_pct: float = 0.01, upper_pct: float = 0.99
) -> np.ndarray:
    """Winsorize values to remove extreme outliers."""
    lower = np.quantile(values, lower_pct)
    upper = np.quantile(values, upper_pct)
    return np.clip(values, lower, upper)


def generate_id(*args) -> str:
    """Generate a deterministic ID from arguments."""
    key = "_".join(str(a) for a in args)
    return hashlib.md5(key.encode()).hexdigest()[:12]


def chunk_list(lst: List[Any], size: int) -> List[List[Any]]:
    """Split a list into chunks of given size."""
    return [lst[i : i + size] for i in range(0, len(lst), size)]


def flatten_dict(d: Dict, parent_key: str = "", sep: str = "_") -> Dict:
    """Flatten nested dictionary."""
    items: List = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def format_odds(decimal_odds: float) -> str:
    """Format decimal odds for display."""
    return f"${decimal_odds:.2f}"


def format_probability(prob: float) -> str:
    """Format probability as percentage string."""
    return f"{prob * 100:.1f}%"


def format_edge(edge: float) -> str:
    """Format edge with sign for display."""
    sign = "+" if edge >= 0 else ""
    return f"{sign}{edge * 100:.1f}%"


def format_ev(ev: float) -> str:
    """Format EV with sign for display."""
    sign = "+" if ev >= 0 else ""
    return f"{sign}{ev * 100:.1f}%"


def season_from_date(dt: datetime) -> int:
    """Infer AFL season year from date."""
    # AFL season runs March-October
    if dt.month >= 1 and dt.month <= 2:
        return dt.year - 1
    return dt.year


def round_to_half(value: float) -> float:
    """Round to nearest 0.5 (for line/total markets)."""
    return round(value * 2) / 2


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max."""
    return max(min_val, min(max_val, value))


def prob_to_american(prob: float) -> int:
    """Convert probability to American odds."""
    if prob >= 0.5:
        return int(-prob / (1 - prob) * 100)
    else:
        return int((1 - prob) / prob * 100)


def american_to_decimal(american: int) -> float:
    """Convert American odds to decimal."""
    if american > 0:
        return (american / 100) + 1
    else:
        return (100 / abs(american)) + 1


def decimal_to_american(decimal_odds: float) -> int:
    """Convert decimal odds to American format."""
    if decimal_odds >= 2.0:
        return int((decimal_odds - 1) * 100)
    else:
        return int(-100 / (decimal_odds - 1))
