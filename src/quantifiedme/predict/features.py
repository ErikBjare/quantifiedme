"""Feature transforms for predictive QS models.

Core idea: raw binary substance tags (tag:caffeine = 0/1) lose temporal
information. A decay kernel encodes *how much* of a substance is still
active, based on pharmacological half-lives.

    substance_active(t) = Σ dose_i * exp(-(t - t_i) / τ)

Since QSlang data is daily binary (not timestamped doses), we use a
simplified multi-day decay: each day's binary flag is treated as a unit
dose, and the kernel sums contributions from the past `window` days.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Substance half-lives in days (converted from hours for daily resolution).
# These are pharmacological half-lives; behavioral effects may differ.
# Using ~2x half-life as the effective τ for daily aggregation.
SUBSTANCE_DECAY_DAYS: dict[str, float] = {
    "caffeine": 0.5,  # ~5-6h half-life, clears within a day
    "alcohol": 0.5,  # acute: ~3-4h, but next-day hangover effect ~1d
    "cannabinoids": 1.5,  # THC: 6-12h acute, fat-soluble chronic load
    "nicotine": 0.25,  # ~2h half-life, very fast clearance
    "psychedelics": 2.0,  # acute 6-12h, afterglow/tolerance multi-day
    "stimulants": 0.5,  # phenylpiracetam etc, ~3-5h
    "nootropics": 1.0,  # varies widely, conservative default
    "sleepaids": 0.5,  # melatonin etc, clears by morning
    "dissociatives": 1.0,
    "empathogens": 2.0,  # MDMA-like, multi-day recovery
    "gabaergics": 0.5,
    "benzos": 1.5,  # longer half-life benzos
    "depressants": 0.5,
}

# Default decay for substances not in the table above
DEFAULT_DECAY_DAYS = 1.0


def decay_kernel(
    series: pd.Series,
    tau: float,
    window: int = 7,
) -> pd.Series:
    """Apply exponential decay kernel to a binary time series.

    For each day t, computes:
        kernel(t) = Σ_{k=0}^{window-1} series[t-k] * exp(-k / τ)

    Args:
        series: Binary (0/1) daily series indexed by date.
        tau: Decay constant in days. Larger = slower decay.
        window: Number of past days to consider.

    Returns:
        Series of same length with decay-weighted values.
    """
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")

    weights = np.exp(-np.arange(window) / tau)
    values = series.fillna(0).values.astype(float)

    # Vectorized convolution: pad with zeros on the left, convolve with weights
    padded = np.concatenate([np.zeros(window - 1), values])
    result_values = np.array(
        [np.dot(padded[i : i + window][::-1], weights) for i in range(len(values))]
    )

    return pd.Series(result_values, index=series.index, dtype=float)


def build_substance_features(
    df: pd.DataFrame,
    top_n: int | None = None,
    min_frequency: float = 0.01,
    window: int = 7,
) -> pd.DataFrame:
    """Build decay kernel features for substance tags.

    Args:
        df: DataFrame with `tag:*` columns (binary 0/1).
        top_n: If set, keep only the N most frequent substances.
        min_frequency: Minimum fraction of days with usage to include.
        window: Lookback window for decay kernels.

    Returns:
        DataFrame with `decay:*` columns for each substance.
    """
    tag_cols = [c for c in df.columns if c.startswith("tag:")]
    if not tag_cols:
        return pd.DataFrame(index=df.index)

    # Filter by frequency
    frequencies = df[tag_cols].mean()
    active_tags = frequencies[frequencies >= min_frequency].index.tolist()

    if top_n is not None:
        active_tags = frequencies[active_tags].nlargest(top_n).index.tolist()

    features = pd.DataFrame(index=df.index)

    for col in active_tags:
        substance = col.removeprefix("tag:")
        tau = SUBSTANCE_DECAY_DAYS.get(substance, DEFAULT_DECAY_DAYS)

        # Raw binary (today)
        features[f"decay:{substance}:today"] = df[col].fillna(0)

        # Decay kernel (accumulated exposure)
        features[f"decay:{substance}:kernel"] = decay_kernel(df[col], tau=tau, window=window)

        # Simple trailing sum (how many of past N days)
        features[f"decay:{substance}:count_{window}d"] = (
            df[col].fillna(0).rolling(window, min_periods=1).sum()
        )

    return features


def build_screentime_features(
    df: pd.DataFrame,
    lag_days: list[int] | None = None,
    exclude_col: str | None = None,
) -> pd.DataFrame:
    """Build lag features for screen time categories.

    Args:
        df: DataFrame with `time:*` columns (hours).
        lag_days: Which lags to create. Default [1, 2, 3, 7].
        exclude_col: Column to exclude (avoids duplicating AR features for the target).

    Returns:
        DataFrame with lag and rolling features for key categories.
    """
    if lag_days is None:
        lag_days = [1, 2, 3, 7]

    # Key categories worth modeling (high variance, meaningful)
    key_categories = [
        "time:Work",
        "time:Programming",
        "time:Media",
        "time:Social Media",
        "time:Games",
    ]

    # Filter to categories that exist in the data, excluding the target
    # (AR features handle the target with more comprehensive rolling stats)
    available = [c for c in key_categories if c in df.columns and c != exclude_col]
    features = pd.DataFrame(index=df.index)

    for col in available:
        name = col.removeprefix("time:")

        for lag in lag_days:
            features[f"lag:{name}:d-{lag}"] = df[col].shift(lag)

        # 7-day rolling mean
        features[f"roll:{name}:7d_mean"] = (
            df[col].rolling(7, min_periods=1).mean()
        )

        # 7-day rolling std (consistency/volatility)
        features[f"roll:{name}:7d_std"] = (
            df[col].rolling(7, min_periods=1).std().fillna(0)
        )

    return features


def build_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build calendar/temporal features from the date index.

    Returns:
        DataFrame with day-of-week, weekend flag, etc.
    """
    dates = pd.DatetimeIndex(df.index)
    features = pd.DataFrame(index=df.index)

    features["dow"] = dates.dayofweek  # 0=Monday
    features["is_weekend"] = (dates.dayofweek >= 5).astype(int)
    features["month"] = dates.month
    features["day_of_year"] = dates.dayofyear

    # Cyclical encoding for day of week
    features["dow_sin"] = np.sin(2 * np.pi * dates.dayofweek / 7)
    features["dow_cos"] = np.cos(2 * np.pi * dates.dayofweek / 7)

    return features


def build_autoregressive_features(
    df: pd.DataFrame,
    target_col: str,
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """Build autoregressive features for the target variable.

    The target's own recent history is often the strongest predictor.

    Args:
        df: DataFrame containing the target column.
        target_col: Column to build AR features for.
        lags: Which lags to create. Default [1, 2, 3, 7].

    Returns:
        DataFrame with lag and rolling features for the target.
    """
    if lags is None:
        lags = [1, 2, 3, 7]

    if target_col not in df.columns:
        logger.warning(f"Target column '{target_col}' not found — skipping AR features")
        return pd.DataFrame(index=df.index)

    features = pd.DataFrame(index=df.index)
    name = target_col.replace(":", "_")

    for lag in lags:
        features[f"ar:{name}:d-{lag}"] = df[target_col].shift(lag)

    # Rolling statistics
    features[f"ar:{name}:7d_mean"] = df[target_col].rolling(7, min_periods=1).mean()
    features[f"ar:{name}:7d_std"] = df[target_col].rolling(7, min_periods=1).std().fillna(0)
    features[f"ar:{name}:14d_mean"] = df[target_col].rolling(14, min_periods=1).mean()

    return features


def build_feature_frame(
    df: pd.DataFrame,
    target_col: str = "time:Work",
    top_n_substances: int | None = 15,
    lag_days: list[int] | None = None,
    substance_window: int = 7,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build the full feature frame for prediction.

    Combines substance decay kernels, screen time lags, and temporal
    features. The target is shifted by +1 day (predict tomorrow's value
    from today's features).

    Args:
        df: Raw DataFrame from load_all_df() or CSV export.
        target_col: Column to predict (default: time:Work).
        top_n_substances: Number of top substances to include.
        lag_days: Lag days for screen time features.
        substance_window: Lookback window for substance kernels.

    Returns:
        (X, y) tuple where X is the feature matrix and y is the
        next-day target. Rows with NaN target are dropped.
    """
    substance_features = build_substance_features(
        df,
        top_n=top_n_substances,
        window=substance_window,
    )
    screentime_features = build_screentime_features(df, lag_days=lag_days, exclude_col=target_col)
    temporal_features = build_temporal_features(df)
    ar_features = build_autoregressive_features(df, target_col, lags=lag_days)

    X = pd.concat([substance_features, screentime_features, temporal_features, ar_features], axis=1)

    # Target: next-day value (shift -1 so today's features predict tomorrow)
    y = df[target_col].shift(-1)

    # Drop rows where target is NaN (last row, plus any gaps)
    valid = y.notna() & X.notna().all(axis=1)
    X = X.loc[valid]
    y = y.loc[valid]

    return X, y
