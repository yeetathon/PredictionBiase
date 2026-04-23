"""Lightweight feature importance: permutation importance + correlation ranking."""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional
from loguru import logger


class FeatureImportanceAnalyzer:
    """
    Post-training feature importance analysis.

    Provides two fast methods:
      permutation_importance — shuffles each feature and measures accuracy drop
      correlation_ranking    — absolute Pearson correlation with target (no model needed)

    Both return DataFrames sorted descending by importance so callers can easily
    slice the top-N features or log them.
    """

    def __init__(self, n_repeats: int = 5, random_state: int = 42):
        self.n_repeats = n_repeats
        self.random_state = random_state

    def permutation_importance(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        scoring: str = "accuracy",
    ) -> pd.DataFrame:
        """
        Compute permutation feature importance using sklearn.

        Returns DataFrame[feature, importance_mean, importance_std] sorted
        descending by importance_mean.  Returns empty DataFrame on failure.
        """
        if X.empty or len(X) < 10:
            return pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])
        try:
            from sklearn.inspection import permutation_importance as _pi

            result = _pi(
                model, X, y,
                n_repeats=self.n_repeats,
                random_state=self.random_state,
                scoring=scoring,
            )
            df = pd.DataFrame({
                "feature": X.columns.tolist(),
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
            return df
        except Exception as exc:
            logger.warning("Permutation importance failed: {}", exc)
            return pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])

    def correlation_ranking(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        top_n: int = 30,
    ) -> pd.DataFrame:
        """
        Rank features by absolute Pearson correlation with target.

        Fast drop-in alternative when the dataset is too small for reliable
        permutation importance.  Returns DataFrame[feature, abs_corr, corr].
        """
        if X.empty:
            return pd.DataFrame(columns=["feature", "abs_corr", "corr"])
        rows = []
        y_vals = y.values.astype(float)
        for col in X.columns:
            try:
                c = float(np.corrcoef(X[col].fillna(0).values, y_vals)[0, 1])
                if np.isnan(c):
                    c = 0.0
            except Exception:
                c = 0.0
            rows.append({"feature": col, "abs_corr": abs(c), "corr": c})
        df = (
            pd.DataFrame(rows)
            .sort_values("abs_corr", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )
        return df

    def log_top_features(
        self,
        importance_df: pd.DataFrame,
        market: str = "",
        top_n: int = 20,
    ) -> None:
        """Log the top N most important features."""
        if importance_df.empty:
            return
        tag = f"[{market}] " if market else ""
        top = importance_df.head(top_n)
        logger.info("{}Top {} features:", tag, min(top_n, len(top)))
        val_col = "importance_mean" if "importance_mean" in top.columns else "abs_corr"
        std_col = "importance_std" if "importance_std" in top.columns else None
        for _, row in top.iterrows():
            val = float(row.get(val_col, 0.0))
            std = float(row.get(std_col, 0.0)) if std_col else 0.0
            logger.info("  {:<50s}  {:.4f} ± {:.4f}", row["feature"], val, std)
