"""Tests for the Home Assistant environmental sensor data loader."""

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from quantifiedme.load.home_assistant import (
    create_fake_sensor_df,
    load_sensor_df,
)


def _create_modern_db(path: Path) -> Path:
    """Create a minimal HA SQLite DB with modern schema (2023+)."""
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE states_meta (
            metadata_id INTEGER PRIMARY KEY,
            entity_id TEXT NOT NULL
        );
        CREATE TABLE states (
            state_id INTEGER PRIMARY KEY,
            metadata_id INTEGER NOT NULL,
            state TEXT,
            last_updated_ts REAL
        );
        INSERT INTO states_meta VALUES (1, 'sensor.temperature_bedroom');
        INSERT INTO states_meta VALUES (2, 'sensor.co2_office');
        INSERT INTO states VALUES (1, 1, '20.5', 1704067200.0);
        INSERT INTO states VALUES (2, 1, '21.0', 1704070800.0);
        INSERT INTO states VALUES (3, 2, '850', 1704067200.0);
        INSERT INTO states VALUES (4, 2, 'unavailable', 1704074400.0);
    """)
    con.commit()
    con.close()
    return path


def _create_legacy_db(path: Path) -> Path:
    """Create a minimal HA SQLite DB with legacy schema (pre-2023)."""
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE states (
            state_id INTEGER PRIMARY KEY,
            entity_id TEXT,
            state TEXT,
            last_updated TEXT
        );
        INSERT INTO states VALUES (1, 'sensor.temperature_bedroom', '19.8', '2024-01-01 00:00:00');
        INSERT INTO states VALUES (2, 'sensor.temperature_bedroom', '20.1', '2024-01-01 01:00:00');
        INSERT INTO states VALUES (3, 'sensor.co2_office', 'unavailable', '2024-01-01 00:00:00');
    """)
    con.commit()
    con.close()
    return path


@pytest.fixture
def modern_db(tmp_path: Path) -> Path:
    return _create_modern_db(tmp_path / "home-assistant_v2.db")


@pytest.fixture
def legacy_db(tmp_path: Path) -> Path:
    return _create_legacy_db(tmp_path / "home-assistant_v2.db")


def test_load_sensor_df_modern(modern_db: Path) -> None:
    df = load_sensor_df(path=modern_db)

    assert isinstance(df, pd.DataFrame)
    assert df.index.name == "timestamp"
    assert isinstance(df.index, pd.DatetimeIndex)
    assert str(df.index.tz) == "UTC"
    assert "entity_id" in df.columns
    assert "state" in df.columns
    assert "unit" in df.columns
    # 'unavailable' row should be dropped: 4 rows - 1 = 3
    assert len(df) == 3
    assert df["state"].notna().all()
    # Without units mapping, unit column should be None/NaN
    assert df["unit"].isna().all()


def test_load_sensor_df_with_units(modern_db: Path) -> None:
    units = {"sensor.temperature_bedroom": "°C", "sensor.co2_office": "ppm"}
    df = load_sensor_df(path=modern_db, units=units)

    assert "unit" in df.columns
    temp_rows = df[df["entity_id"] == "sensor.temperature_bedroom"]
    co2_rows = df[df["entity_id"] == "sensor.co2_office"]
    assert (temp_rows["unit"] == "°C").all()
    assert (co2_rows["unit"] == "ppm").all()


def test_load_sensor_df_modern_filter_entity(modern_db: Path) -> None:
    df = load_sensor_df(path=modern_db, entity_ids=["sensor.temperature_bedroom"])

    assert len(df) == 2
    assert (df["entity_id"] == "sensor.temperature_bedroom").all()


def test_load_sensor_df_modern_sorted(modern_db: Path) -> None:
    df = load_sensor_df(path=modern_db)
    assert df.index.is_monotonic_increasing


def test_load_sensor_df_legacy(legacy_db: Path) -> None:
    df = load_sensor_df(path=legacy_db)

    assert isinstance(df, pd.DataFrame)
    assert df.index.name == "timestamp"
    assert isinstance(df.index, pd.DatetimeIndex)
    assert str(df.index.tz) == "UTC"
    # 'unavailable' dropped: 3 rows - 1 = 2
    assert len(df) == 2
    assert df["state"].notna().all()


def test_load_sensor_df_legacy_filter_entity(legacy_db: Path) -> None:
    df = load_sensor_df(path=legacy_db, entity_ids=["sensor.temperature_bedroom"])

    assert len(df) == 2
    assert (df["entity_id"] == "sensor.temperature_bedroom").all()


def test_load_sensor_df_empty_entity_ids(modern_db: Path) -> None:
    """entity_ids=[] should return empty DataFrame, not all entities."""
    df = load_sensor_df(path=modern_db, entity_ids=[])

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert "entity_id" in df.columns
    assert "state" in df.columns
    assert "unit" in df.columns
    assert isinstance(df.index, pd.DatetimeIndex)
    assert str(df.index.tz) == "UTC"


def test_load_sensor_df_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Home Assistant database not found"):
        load_sensor_df(path=tmp_path / "nonexistent.db")


def test_create_fake_sensor_df() -> None:
    df = create_fake_sensor_df(start="2024-01-01", end="2024-01-07")

    assert isinstance(df, pd.DataFrame)
    assert df.index.name == "timestamp"
    assert isinstance(df.index, pd.DatetimeIndex)
    assert str(df.index.tz) == "UTC"
    assert "entity_id" in df.columns
    assert "state" in df.columns
    assert "unit" in df.columns
    assert len(df) > 0
    assert df["state"].notna().all()
    assert df["unit"].notna().all()
