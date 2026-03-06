"""Tests for the FreeStyle Libre CGM data loader."""

from pathlib import Path

import pandas as pd
import pytest

from quantifiedme.load.freestyle_libre import (
    create_fake_glucose_df,
    load_glucose_daily_df,
    load_glucose_df,
)


# Minimal valid LibreView CSV export (EU mmol/L format)
# Abbott exports include 2 header rows before the column header
SAMPLE_CSV_MMOL = """\
FreeStyle LibreLink - John Doe - 2024-01-31 12:00 UTC.csv
Device Name,Device Serial Number,
LibreLink,ABC-12345-67890,
Device Timestamp,Record Type,Historic Glucose mmol/L,Scan Glucose mmol/L,Non-numeric Rapid-Acting Insulin,Rapid-Acting Insulin (units),Non-numeric Food,Carbohydrates (grams),Carbohydrates (servings),Non-numeric Long-Acting Insulin,Long-Acting Insulin (units),Notes,Strip Glucose mmol/L,Ketone mmol/L,Meal Insulin (units),Correction Insulin (units),User Change Insulin (units)
01-01-2024 00:00,0,5.2,,,,,,,,,,,,,,,
01-01-2024 00:15,0,5.1,,,,,,,,,,,,,,,
01-01-2024 01:00,1,,5.5,,,,,,,,,,,,,,
01-01-2024 08:30,0,7.8,,,,,,,,,,,,,,,
01-01-2024 08:45,0,8.1,,,,,,,,,,,,,,,
01-01-2024 09:00,0,7.5,,,,,,,,,,,,,,,
"""


@pytest.fixture
def sample_csv_mmol(tmp_path: Path) -> Path:
    p = tmp_path / "glucosedata.csv"
    p.write_text(SAMPLE_CSV_MMOL)
    return p


def test_load_glucose_df_mmol(sample_csv_mmol: Path) -> None:
    df = load_glucose_df(path=sample_csv_mmol)

    assert isinstance(df, pd.DataFrame)
    assert "glucose" in df.columns
    assert "record_type" in df.columns
    assert df.index.name == "timestamp"
    assert isinstance(df.index, pd.DatetimeIndex)
    assert str(df.index.tz) == "UTC"  # must be explicitly UTC, not just tz-aware
    # 5 historic + 1 scan = 6 readings
    assert len(df) == 6


def test_load_glucose_df_sorted(sample_csv_mmol: Path) -> None:
    df = load_glucose_df(path=sample_csv_mmol)
    assert df.index.is_monotonic_increasing


def test_load_glucose_df_record_types(sample_csv_mmol: Path) -> None:
    df = load_glucose_df(path=sample_csv_mmol)
    assert set(df["record_type"].unique()).issubset({"historic", "scan"})
    assert (df["record_type"] == "historic").sum() == 5
    assert (df["record_type"] == "scan").sum() == 1


def test_load_glucose_df_values_range(sample_csv_mmol: Path) -> None:
    df = load_glucose_df(path=sample_csv_mmol)
    # All values should be in physiologically plausible mmol/L range
    assert (df["glucose"] > 2.0).all()
    assert (df["glucose"] < 25.0).all()


def test_load_glucose_daily_df(sample_csv_mmol: Path) -> None:
    df = load_glucose_daily_df(path=sample_csv_mmol)

    assert isinstance(df, pd.DataFrame)
    assert "glucose_mean" in df.columns
    assert "glucose_min" in df.columns
    assert "glucose_max" in df.columns
    assert "time_in_range" in df.columns
    assert "n_readings" in df.columns
    # Only one day of data in sample
    assert len(df) == 1


def test_create_fake_glucose_df() -> None:
    df = create_fake_glucose_df(start="2024-01-01", end="2024-01-07")

    assert isinstance(df, pd.DataFrame)
    assert "glucose" in df.columns
    assert isinstance(df.index, pd.DatetimeIndex)
    assert str(df.index.tz) == "UTC"
    # Jan 1 00:00 to Jan 7 00:00 inclusive = 6*24*4 + 1 = 577 readings at 15min intervals
    assert len(df) == 577
    # Values should be physiologically plausible
    assert (df["glucose"] > 2.5).all()
    assert (df["glucose"] < 16.0).all()


def test_fake_glucose_df_reproducible() -> None:
    df1 = create_fake_glucose_df(seed=0)
    df2 = create_fake_glucose_df(seed=0)
    pd.testing.assert_frame_equal(df1, df2)


def test_load_glucose_mgdl_unit(sample_csv_mmol: Path) -> None:
    df_mmol = load_glucose_df(path=sample_csv_mmol, unit="mmol/L")
    df_mgdl = load_glucose_df(path=sample_csv_mmol, unit="mg/dL")

    # mg/dL values should be ~18x higher than mmol/L
    ratio = (df_mgdl["glucose"] / df_mmol["glucose"]).mean()
    assert abs(ratio - 18.018) < 0.1


def test_load_glucose_df_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "nonexistent.csv"
    with pytest.raises(FileNotFoundError, match="nonexistent.csv"):
        load_glucose_df(path=missing)


def test_load_glucose_df_invalid_unit(sample_csv_mmol: Path) -> None:
    with pytest.raises(ValueError, match="Invalid unit"):
        load_glucose_df(path=sample_csv_mmol, unit="kg/m2")
