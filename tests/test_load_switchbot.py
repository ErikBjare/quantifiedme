"""Tests for the SwitchBot environmental sensor data loader."""

from pathlib import Path

import pandas as pd
import pytest

from quantifiedme.load.switchbot import (
    create_fake_environment_df,
    load_environment_daily_df,
    load_environment_df,
    load_environment_devices_df,
)


# SwitchBot Meter export (temperature + humidity)
SAMPLE_CSV_METER = """\
Time,Temperature(°C),Humidity(%)
2024-01-15 00:00,20.5,52
2024-01-15 00:05,20.4,53
2024-01-15 06:00,19.8,55
2024-01-15 12:00,22.3,48
2024-01-15 18:00,21.7,50
2024-01-15 23:55,20.1,54
"""

# SwitchBot CO2 Sensor export (temperature + humidity + CO2)
SAMPLE_CSV_CO2 = """\
Time,Temperature(°C),Humidity(%),CO2(ppm)
2024-01-15 00:00,20.5,52,430
2024-01-15 00:05,20.4,53,425
2024-01-15 09:00,21.0,49,650
2024-01-15 12:00,22.3,48,800
2024-01-15 18:00,21.7,50,720
2024-01-15 23:55,20.1,54,445
"""


@pytest.fixture
def meter_csv(tmp_path: Path) -> Path:
    p = tmp_path / "bedroom_meter.csv"
    p.write_text(SAMPLE_CSV_METER)
    return p


@pytest.fixture
def co2_csv(tmp_path: Path) -> Path:
    p = tmp_path / "office_co2.csv"
    p.write_text(SAMPLE_CSV_CO2)
    return p


def test_load_meter_df(meter_csv: Path) -> None:
    df = load_environment_df(path=meter_csv)

    assert isinstance(df, pd.DataFrame)
    assert "temperature_c" in df.columns
    assert "humidity_pct" in df.columns
    assert "co2_ppm" not in df.columns  # meter doesn't have CO2
    assert df.index.name == "timestamp"
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.tz is not None
    assert len(df) == 6


def test_load_co2_df(co2_csv: Path) -> None:
    df = load_environment_df(path=co2_csv)

    assert "co2_ppm" in df.columns
    assert (df["co2_ppm"] > 400).all()
    assert len(df) == 6


def test_load_df_sorted(meter_csv: Path) -> None:
    df = load_environment_df(path=meter_csv)
    assert df.index.is_monotonic_increasing


def test_load_df_with_device_name(meter_csv: Path) -> None:
    df = load_environment_df(path=meter_csv, device_name="bedroom")
    assert "device" in df.columns
    assert (df["device"] == "bedroom").all()


def test_load_devices_df(meter_csv: Path, co2_csv: Path) -> None:
    df = load_environment_devices_df(
        devices={"bedroom": meter_csv, "office": co2_csv}
    )

    assert isinstance(df, pd.DataFrame)
    assert "device" in df.columns
    assert set(df["device"].unique()) == {"bedroom", "office"}
    # Combined: 6 bedroom + 6 office = 12 rows
    assert len(df) == 12


def test_load_environment_daily_df(meter_csv: Path) -> None:
    df = load_environment_daily_df(path=meter_csv)

    assert isinstance(df, pd.DataFrame)
    assert "temperature_c_mean" in df.columns
    assert "temperature_c_min" in df.columns
    assert "temperature_c_max" in df.columns
    assert "humidity_pct_mean" in df.columns
    # Single day of data
    assert len(df) == 1
    # Max temp should be >= mean >= min
    assert (df["temperature_c_max"] >= df["temperature_c_mean"]).all()
    assert (df["temperature_c_mean"] >= df["temperature_c_min"]).all()


def test_create_fake_environment_df_with_co2() -> None:
    df = create_fake_environment_df(
        start="2024-01-01", end="2024-01-07", include_co2=True
    )

    assert isinstance(df, pd.DataFrame)
    assert "temperature_c" in df.columns
    assert "humidity_pct" in df.columns
    assert "co2_ppm" in df.columns
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.tz is not None

    # Plausible ranges
    assert (df["temperature_c"] > 10).all()
    assert (df["temperature_c"] < 35).all()
    assert (df["humidity_pct"] > 10).all()
    assert (df["humidity_pct"] < 100).all()
    assert (df["co2_ppm"] > 300).all()
    assert (df["co2_ppm"] < 2500).all()


def test_create_fake_environment_df_no_co2() -> None:
    df = create_fake_environment_df(include_co2=False)
    assert "co2_ppm" not in df.columns


def test_fake_environment_df_reproducible() -> None:
    df1 = create_fake_environment_df(seed=0)
    df2 = create_fake_environment_df(seed=0)
    pd.testing.assert_frame_equal(df1, df2)
