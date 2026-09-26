"""Tests for the Home Assistant environmental sensor data loader."""

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from quantifiedme.load.home_assistant import (
    DailyFeature,
    aggregate_daily_features,
    create_fake_sensor_df,
    load_daily_df,
    load_sensor_df,
    load_sensor_df_api,
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


HA_API_RESPONSE = [
    [
        {
            "entity_id": "sensor.temperature_bedroom",
            "state": "20.5",
            "last_updated": "2024-01-01T00:00:00+00:00",
        },
        {
            "entity_id": "sensor.temperature_bedroom",
            "state": "21.0",
            "last_updated": "2024-01-01T01:00:00+00:00",
        },
        {
            "entity_id": "sensor.temperature_bedroom",
            "state": "unavailable",
            "last_updated": "2024-01-01T02:00:00+00:00",
        },
    ],
    [
        {
            "entity_id": "sensor.co2_office",
            "state": "850",
            "last_updated": "2024-01-01T00:00:00+00:00",
        },
    ],
]


def _make_api_mock(response_data: object) -> MagicMock:
    mock_response = MagicMock()
    mock_response.json.return_value = response_data
    mock_response.raise_for_status.return_value = None
    return mock_response


def test_load_sensor_df_api() -> None:
    with patch("requests.get", return_value=_make_api_mock(HA_API_RESPONSE)):
        df = load_sensor_df_api(
            url="http://homeassistant.local:8123",
            token="test-token",
            entity_ids=["sensor.temperature_bedroom", "sensor.co2_office"],
        )

    assert isinstance(df, pd.DataFrame)
    assert df.index.name == "timestamp"
    assert isinstance(df.index, pd.DatetimeIndex)
    assert str(df.index.tz) == "UTC"
    assert "entity_id" in df.columns
    assert "state" in df.columns
    assert "unit" in df.columns
    # 'unavailable' row dropped: 4 rows - 1 = 3
    assert len(df) == 3
    assert df["state"].notna().all()
    assert df["unit"].isna().all()


def test_load_sensor_df_api_with_units() -> None:
    units = {"sensor.temperature_bedroom": "°C", "sensor.co2_office": "ppm"}
    with patch("requests.get", return_value=_make_api_mock(HA_API_RESPONSE)):
        df = load_sensor_df_api(
            url="http://homeassistant.local:8123",
            token="test-token",
            entity_ids=["sensor.temperature_bedroom", "sensor.co2_office"],
            units=units,
        )

    temp_rows = df[df["entity_id"] == "sensor.temperature_bedroom"]
    co2_rows = df[df["entity_id"] == "sensor.co2_office"]
    assert (temp_rows["unit"] == "°C").all()
    assert (co2_rows["unit"] == "ppm").all()


def test_load_sensor_df_api_sorted() -> None:
    with patch("requests.get", return_value=_make_api_mock(HA_API_RESPONSE)):
        df = load_sensor_df_api(
            url="http://homeassistant.local:8123", token="test-token"
        )

    assert df.index.is_monotonic_increasing


def test_load_sensor_df_api_empty_entity_ids() -> None:
    df = load_sensor_df_api(
        url="http://homeassistant.local:8123",
        token="test-token",
        entity_ids=[],
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert "entity_id" in df.columns
    assert isinstance(df.index, pd.DatetimeIndex)
    assert str(df.index.tz) == "UTC"


def test_load_sensor_df_api_empty_response() -> None:
    with patch("requests.get", return_value=_make_api_mock([])):
        df = load_sensor_df_api(
            url="http://homeassistant.local:8123", token="test-token"
        )

    assert len(df) == 0
    assert isinstance(df.index, pd.DatetimeIndex)
    assert str(df.index.tz) == "UTC"


def test_load_sensor_df_api_http_error() -> None:
    import requests  # type: ignore[import-untyped]

    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = requests.HTTPError("401 Unauthorized")
    with patch("requests.get", return_value=mock_response):
        with pytest.raises(requests.HTTPError):
            load_sensor_df_api(url="http://homeassistant.local:8123", token="bad-token")


def _sauna_readings() -> pd.DataFrame:
    """Long-format readings: day 1 has a sauna session (max 82 °C), day 2 does not (max 22 °C)."""
    rows = [
        # 2024-01-01: idle then a hot session
        ("sensor.sauna_probe_temperature", 20.0, "2024-01-01T08:00:00+00:00"),
        ("sensor.sauna_probe_temperature", 82.0, "2024-01-01T18:00:00+00:00"),
        ("sensor.sauna_probe_temperature", 65.0, "2024-01-01T19:00:00+00:00"),
        # 2024-01-02: never heated
        ("sensor.sauna_probe_temperature", 19.0, "2024-01-02T08:00:00+00:00"),
        ("sensor.sauna_probe_temperature", 22.0, "2024-01-02T18:00:00+00:00"),
        # CO2 readings on day 1 only
        (
            "sensor.s1_pro_multi_sense_e8b4cc_scd40_co2_concentration",
            500.0,
            "2024-01-01T08:00:00+00:00",
        ),
        (
            "sensor.s1_pro_multi_sense_e8b4cc_scd40_co2_concentration",
            900.0,
            "2024-01-01T23:00:00+00:00",
        ),
    ]
    df = pd.DataFrame(rows, columns=["entity_id", "state", "ts"])
    df["timestamp"] = pd.to_datetime(df["ts"], utc=True)
    return df.drop(columns=["ts"]).set_index("timestamp")


def test_aggregate_daily_features_sauna_boolean() -> None:
    df = aggregate_daily_features(_sauna_readings())

    assert df.index.name == "date"
    assert "sauna" in df.columns
    assert "bedroom_co2" in df.columns
    # Day 1 crossed 60 °C → True; day 2 peaked at 22 °C → False
    assert bool(df.loc["2024-01-01", "sauna"]) is True
    assert bool(df.loc["2024-01-02", "sauna"]) is False
    # CO2 mean of 500 and 900 on day 1
    assert df.loc["2024-01-01", "bedroom_co2"] == pytest.approx(700.0)


def test_aggregate_daily_features_missing_day_is_nan_not_false() -> None:
    """A day with no readings must stay NaN (unknown), never coerced to False."""
    df = aggregate_daily_features(_sauna_readings())
    # Day 2 has no CO2 readings → NaN, not a fabricated value
    assert pd.isna(df.loc["2024-01-02", "bedroom_co2"])


def test_aggregate_daily_features_absent_sensor_all_nan() -> None:
    """A configured feature whose sensor never appears yields an all-NaN column."""
    features = [
        DailyFeature(name="never", entity_id="sensor.does_not_exist", agg="mean")
    ]
    df = aggregate_daily_features(_sauna_readings(), features=features)
    assert "never" in df.columns
    assert df["never"].isna().all()


def test_aggregate_daily_features_rejects_bad_agg() -> None:
    features = [
        DailyFeature(name="x", entity_id="sensor.sauna_probe_temperature", agg="median")
    ]
    with pytest.raises(ValueError, match="Unsupported agg"):
        aggregate_daily_features(_sauna_readings(), features=features)


def test_load_daily_df_from_sqlite(tmp_path: Path) -> None:
    """load_daily_df reads only the configured sensors and returns daily features."""
    db = tmp_path / "home-assistant_v2.db"
    con = sqlite3.connect(db)
    con.executescript("""
        CREATE TABLE states_meta (metadata_id INTEGER PRIMARY KEY, entity_id TEXT NOT NULL);
        CREATE TABLE states (
            state_id INTEGER PRIMARY KEY, metadata_id INTEGER NOT NULL,
            state TEXT, last_updated_ts REAL
        );
        INSERT INTO states_meta VALUES (1, 'sensor.sauna_probe_temperature');
        INSERT INTO states VALUES (1, 1, '20.0', 1704096000.0);
        INSERT INTO states VALUES (2, 1, '82.0', 1704132000.0);
    """)
    con.commit()
    con.close()

    features = [
        DailyFeature(
            name="sauna",
            entity_id="sensor.sauna_probe_temperature",
            agg="max",
            threshold=60.0,
        )
    ]
    df = load_daily_df(path=db, features=features)
    assert "sauna" in df.columns
    assert bool(df["sauna"].iloc[0]) is True


def test_aggregate_daily_features_index_union() -> None:
    """Second feature with longer date range must not lose its extra days (P1 fix)."""
    # sauna: 1 day; co2: 3 days — result must have 3 rows, not 1
    rows = [
        ("sensor.sauna_probe_temperature", 80.0, "2024-01-01T12:00:00+00:00"),
        ("sensor.s1_pro_multi_sense_e8b4cc_scd40_co2_concentration", 800.0, "2024-01-01T08:00:00+00:00"),
        ("sensor.s1_pro_multi_sense_e8b4cc_scd40_co2_concentration", 900.0, "2024-01-02T08:00:00+00:00"),
        ("sensor.s1_pro_multi_sense_e8b4cc_scd40_co2_concentration", 950.0, "2024-01-03T08:00:00+00:00"),
    ]
    df = pd.DataFrame(rows, columns=["entity_id", "state", "ts"])
    df["timestamp"] = pd.to_datetime(df["ts"], utc=True)
    df = df.drop(columns=["ts"]).set_index("timestamp")

    result = aggregate_daily_features(df)
    assert len(result) == 3, "CO2 days 2 and 3 must not be dropped when sauna ends on day 1"
    assert pd.isna(result.loc["2024-01-02", "sauna"])
    assert pd.isna(result.loc["2024-01-03", "sauna"])
    assert result.loc["2024-01-02", "bedroom_co2"] == pytest.approx(900.0)


def test_aggregate_daily_features_sum_threshold_empty_day_is_nan() -> None:
    """Empty day with agg='sum' must stay NaN (not coerced to False via zero-sum)."""
    rows = [
        ("sensor.steps", 8500.0, "2024-01-01T12:00:00+00:00"),
        # day 2: no readings at all
        ("sensor.steps", 12000.0, "2024-01-03T12:00:00+00:00"),
    ]
    df = pd.DataFrame(rows, columns=["entity_id", "state", "ts"])
    df["timestamp"] = pd.to_datetime(df["ts"], utc=True)
    df = df.drop(columns=["ts"]).set_index("timestamp")

    features = [DailyFeature(name="active", entity_id="sensor.steps", agg="sum", threshold=10000.0)]
    result = aggregate_daily_features(df, features)
    assert bool(result.loc["2024-01-01", "active"]) is False   # 8500 < 10000
    assert pd.isna(result.loc["2024-01-02", "active"]), "empty day must be NaN, not False"
    assert bool(result.loc["2024-01-03", "active"]) is True    # 12000 > 10000


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
