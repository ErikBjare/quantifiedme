"""LightGBM baseline model for QS prediction.

Quick predictive baseline to reveal which features actually matter
before investing in Bayesian models. Predicts next-day outcomes
(e.g. work hours) from substance history + screen time patterns.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .features import build_feature_frame

logger = logging.getLogger(__name__)


@dataclass
class BaselineResult:
    """Results from training a baseline model."""

    train_rmse: float
    test_rmse: float
    train_mae: float
    test_mae: float
    train_r2: float
    test_r2: float
    feature_importance: pd.Series
    target_col: str
    n_train: int
    n_val: int
    n_test: int
    target_mean: float
    target_std: float
    predictions: pd.Series = field(repr=False)

    def summary(self) -> str:
        lines = [
            f"Baseline model: {self.target_col}",
            f"  Train: n={self.n_train}, RMSE={self.train_rmse:.3f}, MAE={self.train_mae:.3f}, R²={self.train_r2:.3f}",
            f"  Test:  n={self.n_test} (val={self.n_val}), RMSE={self.test_rmse:.3f}, MAE={self.test_mae:.3f}, R²={self.test_r2:.3f}",
            f"  Target: mean={self.target_mean:.3f}, std={self.target_std:.3f}",
            "  Top 10 features:",
        ]
        for feat, imp in self.feature_importance.head(10).items():
            lines.append(f"    {feat}: {imp:.0f}")
        return "\n".join(lines)


def load_csv_export(path: str | Path) -> pd.DataFrame:
    """Load the privacy-safe CSV export from qs-export.py.

    Args:
        path: Path to the CSV file.

    Returns:
        DataFrame indexed by date with time:* and tag:* columns.
    """
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    df.index = df.index.date  # type: ignore[attr-defined]  # normalize to date objects
    df = df.sort_index()
    return df


def train_baseline(
    df: pd.DataFrame,
    target_col: str = "time:Work",
    test_fraction: float = 0.2,
    top_n_substances: int = 15,
) -> BaselineResult:
    """Train a LightGBM model predicting next-day target.

    Uses time-based split (no shuffling — respects temporal ordering).

    Args:
        df: Raw DataFrame from CSV export or load_all_df().
        target_col: Column to predict.
        test_fraction: Fraction of data to hold out (from the end).
        top_n_substances: Number of top substances to include.

    Returns:
        BaselineResult with metrics and feature importances.
    """
    import lightgbm as lgb  # type: ignore[import-untyped]

    X, y = build_feature_frame(df, target_col=target_col, top_n_substances=top_n_substances)

    # Sanitize feature names for LightGBM (no special JSON chars)
    import re as _re

    clean_names = {c: _re.sub(r"[^a-zA-Z0-9_]", "_", c) for c in X.columns}

    # Check for collisions: distinct original names mapping to same sanitized name
    reverse: dict[str, str] = {}
    for orig, clean in clean_names.items():
        if clean in reverse and reverse[clean] != orig:
            raise ValueError(
                f"Feature name collision after sanitization: "
                f"'{orig}' and '{reverse[clean]}' both map to '{clean}'"
            )
        reverse[clean] = orig

    X = X.rename(columns=clean_names)

    # Time-based 3-way split: train / validation (early stopping) / test (metrics)
    # This prevents early stopping from biasing test metrics (Greptile P1).
    val_fraction = test_fraction
    train_end = int(len(X) * (1 - test_fraction - val_fraction))
    val_end = int(len(X) * (1 - test_fraction))
    X_train, X_val, X_test = X.iloc[:train_end], X.iloc[train_end:val_end], X.iloc[val_end:]
    y_train, y_val, y_test = y.iloc[:train_end], y.iloc[train_end:val_end], y.iloc[val_end:]

    logger.info(
        f"Split: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test"
    )
    logger.info(f"Features: {len(X.columns)}")

    # Train LightGBM (early stopping on validation set, NOT test set)
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbosity": -1,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )

    # Predictions on held-out test set (not used during training at all)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Metrics
    from sklearn.metrics import (  # type: ignore[import-untyped]
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )

    train_rmse = float(np.sqrt(mean_squared_error(y_train, y_pred_train)))
    test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    train_mae = float(mean_absolute_error(y_train, y_pred_train))
    test_mae = float(mean_absolute_error(y_test, y_pred_test))
    train_r2 = float(r2_score(y_train, y_pred_train))
    test_r2 = float(r2_score(y_test, y_pred_test))

    # Feature importance (map back to original names)
    reverse_names = {v: k for k, v in clean_names.items()}
    importance = pd.Series(
        model.feature_importance(importance_type="gain"),
        index=[reverse_names.get(c, c) for c in X.columns],
    ).sort_values(ascending=False)

    return BaselineResult(
        train_rmse=train_rmse,
        test_rmse=test_rmse,
        train_mae=train_mae,
        test_mae=test_mae,
        train_r2=train_r2,
        test_r2=test_r2,
        feature_importance=importance,
        target_col=target_col,
        n_train=len(X_train),
        n_val=len(X_val),
        n_test=len(X_test),
        target_mean=float(y.mean()),
        target_std=float(y.std()),
        predictions=pd.Series(y_pred_test, index=y_test.index),
    )


def run_baseline(
    csv_path: str | Path,
    target_col: str = "time:Work",
) -> BaselineResult:
    """Convenience: load CSV and train baseline in one call."""
    df = load_csv_export(csv_path)
    result = train_baseline(df, target_col=target_col)
    print(result.summary())
    return result


def run_diagnostic(
    csv_path: str | Path,
    targets: list[str] | None = None,
) -> dict[str, BaselineResult]:
    """Run baseline across multiple targets and compare.

    Args:
        csv_path: Path to CSV export.
        targets: List of target columns. Default: common screen time categories.

    Returns:
        Dict mapping target name to BaselineResult.
    """
    df = load_csv_export(csv_path)

    if targets is None:
        # Auto-detect meaningful targets (time:* with enough variance)
        candidates = [c for c in df.columns if c.startswith("time:")]
        targets = []
        for col in candidates:
            if df[col].std() > 0.1 and df[col].mean() > 0.05:
                targets.append(col)
        targets.sort(key=lambda c: -df[c].mean())
        targets = targets[:8]  # top 8 by mean hours

    results = {}
    print(f"Running diagnostic across {len(targets)} targets...\n")
    print(f"{'Target':<25} {'Train R²':>10} {'Test R²':>10} {'Test RMSE':>10} {'Test MAE':>10} {'Mean':>8} {'Std':>8} {'N':>6}")
    print("-" * 95)

    for target in targets:
        try:
            result = train_baseline(df, target_col=target)
            results[target] = result
            print(
                f"{target:<25} {result.train_r2:>10.3f} {result.test_r2:>10.3f} "
                f"{result.test_rmse:>10.3f} {result.test_mae:>10.3f} "
                f"{result.target_mean:>8.2f} {result.target_std:>8.2f} "
                f"{result.n_train + result.n_val + result.n_test:>6}"
            )
        except Exception:
            logger.warning(f"Failed for {target}", exc_info=True)

    # Summary: which features appear most across targets
    if results:
        print("\n--- Cross-Target Feature Importance (top features across all targets) ---")
        all_importances: dict[str, float] = {}
        for result in results.values():
            for feat, imp in result.feature_importance.head(10).items():
                key = str(feat)
                all_importances[key] = all_importances.get(key, 0) + imp

        sorted_feats = sorted(all_importances.items(), key=lambda x: -x[1])
        for feat, total_imp in sorted_feats[:15]:
            print(f"  {feat}: {total_imp:.0f}")

    return results
