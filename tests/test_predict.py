"""Tests for the predictive QS framework."""

import numpy as np
import pandas as pd
import pytest

from quantifiedme.predict.features import (
    build_feature_frame,
    build_screentime_features,
    build_substance_features,
    build_temporal_features,
    decay_kernel,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a minimal DataFrame mimicking the QS export structure."""
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    rng = np.random.default_rng(42)

    df = pd.DataFrame(
        {
            "time:Work": rng.uniform(0, 8, 30),
            "time:Programming": rng.uniform(0, 5, 30),
            "time:Media": rng.uniform(0, 4, 30),
            "time:Social Media": rng.uniform(0, 2, 30),
            "time:Games": rng.uniform(0, 3, 30),
            "tag:caffeine": rng.choice([0, 1], 30, p=[0.3, 0.7]),
            "tag:alcohol": rng.choice([0, 1], 30, p=[0.7, 0.3]),
            "tag:cannabinoids": rng.choice([0, 1], 30, p=[0.8, 0.2]),
            "tag:nicotine": rng.choice([0, 1], 30, p=[0.9, 0.1]),
            # Rare substance — should be filtered out by min_frequency
            "tag:psychedelics": np.zeros(30),
        },
        index=dates.date,
    )
    return df


class TestDecayKernel:
    def test_no_exposure_gives_zero(self):
        series = pd.Series([0, 0, 0, 0, 0])
        result = decay_kernel(series, tau=1.0, window=3)
        assert (result == 0).all()

    def test_single_exposure_decays(self):
        series = pd.Series([1, 0, 0, 0, 0])
        result = decay_kernel(series, tau=1.0, window=5)
        # Day 0: 1.0 (immediate)
        assert result.iloc[0] == pytest.approx(1.0)
        # Day 1: exp(-1/1) ≈ 0.368
        assert result.iloc[1] == pytest.approx(np.exp(-1), abs=0.01)
        # Day 4: exp(-4/1) ≈ 0.018
        assert result.iloc[4] == pytest.approx(np.exp(-4), abs=0.01)

    def test_daily_exposure_accumulates(self):
        series = pd.Series([1, 1, 1, 1, 1])
        result = decay_kernel(series, tau=1.0, window=5)
        # Each day accumulates more than a single exposure
        assert result.iloc[4] > result.iloc[0]

    def test_larger_tau_slower_decay(self):
        series = pd.Series([1, 0, 0, 0, 0])
        fast = decay_kernel(series, tau=0.5, window=5)
        slow = decay_kernel(series, tau=2.0, window=5)
        # Slow decay retains more at day 2
        assert slow.iloc[2] > fast.iloc[2]


class TestSubstanceFeatures:
    def test_creates_expected_columns(self, sample_df: pd.DataFrame):
        features = build_substance_features(sample_df, top_n=3, window=7)
        # Should have today, kernel, and count columns per substance
        assert any("decay:" in c and ":today" in c for c in features.columns)
        assert any("decay:" in c and ":kernel" in c for c in features.columns)
        assert any("decay:" in c and ":count_7d" in c for c in features.columns)

    def test_filters_rare_substances(self, sample_df: pd.DataFrame):
        features = build_substance_features(sample_df, min_frequency=0.05)
        # psychedelics has 0 frequency, should be excluded
        assert not any("psychedelics" in c for c in features.columns)

    def test_top_n_limits_substances(self, sample_df: pd.DataFrame):
        features = build_substance_features(sample_df, top_n=2)
        substances = {c.split(":")[1] for c in features.columns}
        assert len(substances) <= 2


class TestScreentimeFeatures:
    def test_creates_lag_features(self, sample_df: pd.DataFrame):
        features = build_screentime_features(sample_df)
        assert "lag:Work:d-1" in features.columns
        assert "lag:Work:d-7" in features.columns

    def test_creates_rolling_features(self, sample_df: pd.DataFrame):
        features = build_screentime_features(sample_df)
        assert "roll:Work:7d_mean" in features.columns
        assert "roll:Work:7d_std" in features.columns


class TestTemporalFeatures:
    def test_creates_expected_columns(self, sample_df: pd.DataFrame):
        features = build_temporal_features(sample_df)
        assert "dow" in features.columns
        assert "is_weekend" in features.columns
        assert "dow_sin" in features.columns
        assert "dow_cos" in features.columns

    def test_weekend_flag_correct(self):
        # 2024-01-06 is a Saturday, 2024-01-07 is a Sunday
        dates = pd.date_range("2024-01-01", periods=14, freq="D")
        df = pd.DataFrame(index=dates.date)
        features = build_temporal_features(df)
        # Monday Jan 1: not weekend
        assert features.iloc[0]["is_weekend"] == 0
        # Saturday Jan 6: weekend
        assert features.iloc[5]["is_weekend"] == 1


class TestBuildFeatureFrame:
    def test_returns_aligned_x_y(self, sample_df: pd.DataFrame):
        X, y = build_feature_frame(sample_df, target_col="time:Work")
        assert len(X) == len(y)
        assert len(X) > 0
        # Target is shifted: last day of input should not be in output
        assert len(X) < len(sample_df)

    def test_no_nans_in_output(self, sample_df: pd.DataFrame):
        X, y = build_feature_frame(sample_df, target_col="time:Work")
        assert not X.isna().any().any()
        assert not y.isna().any()
