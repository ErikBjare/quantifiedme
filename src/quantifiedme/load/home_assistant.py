"""
Loads environmental sensor data from Home Assistant's SQLite database.

Home Assistant stores all sensor history in a local SQLite database at:
  ~/.homeassistant/home-assistant_v2.db

This loader reads the `states` table (with `states_meta` in newer HA versions 2023+)
to extract time-series data for environmental sensors like temperature, humidity, CO2.

Rather than per-device CSV parsers, this loader covers all sensors that Erik routes
through Home Assistant — current and future sensors automatically included.
"""

import sqlite3
from pathlib import Path

import pandas as pd

from ..config import load_config


def load_sensor_df(
    path: Path | None = None,
    entity_ids: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load environmental sensor data from Home Assistant's SQLite database.

    Args:
        path: Path to the Home Assistant SQLite database file.
              Defaults to config["data"]["home_assistant"].
        entity_ids: Optional list of entity IDs to filter (e.g.,
                    ["sensor.temperature_bedroom", "sensor.co2_office"]).
                    If None, all numeric sensor states are returned.

    Returns:
        DataFrame indexed by UTC timestamp with columns:
        - entity_id: HA entity ID
        - state: numeric sensor value

    Non-numeric states (e.g., 'unavailable', 'unknown') are dropped.
    """
    if path is None:
        config = load_config()
        path = Path(config["data"]["home_assistant"]).expanduser()
    else:
        path = Path(path).expanduser()

    if not path.exists():
        raise FileNotFoundError(f"Home Assistant database not found at {path}")

    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        tables = {
            r[0]
            for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        if "states_meta" in tables:
            df = _load_modern_schema(con, entity_ids)
        else:
            df = _load_legacy_schema(con, entity_ids)
    finally:
        con.close()

    return df


def _load_modern_schema(
    con: sqlite3.Connection, entity_ids: list[str] | None
) -> pd.DataFrame:
    """Load from HA 2023.x+ schema with states_meta table."""
    if entity_ids:
        placeholders = ",".join("?" * len(entity_ids))
        query = f"""
            SELECT
                sm.entity_id,
                s.state,
                datetime(s.last_updated_ts, 'unixepoch') AS last_updated
            FROM states s
            JOIN states_meta sm ON s.metadata_id = sm.metadata_id
            WHERE sm.entity_id IN ({placeholders})
            ORDER BY s.last_updated_ts
        """
        df = pd.read_sql_query(query, con, params=tuple(entity_ids))
    else:
        query = """
            SELECT
                sm.entity_id,
                s.state,
                datetime(s.last_updated_ts, 'unixepoch') AS last_updated
            FROM states s
            JOIN states_meta sm ON s.metadata_id = sm.metadata_id
            ORDER BY s.last_updated_ts
        """
        df = pd.read_sql_query(query, con)

    return _clean_df(df)


def _load_legacy_schema(
    con: sqlite3.Connection, entity_ids: list[str] | None
) -> pd.DataFrame:
    """Load from legacy HA schema (pre-2023) with entity_id in states table."""
    if entity_ids:
        placeholders = ",".join("?" * len(entity_ids))
        query = f"""
            SELECT entity_id, state, last_updated
            FROM states
            WHERE entity_id IN ({placeholders})
            ORDER BY last_updated
        """
        df = pd.read_sql_query(query, con, params=tuple(entity_ids))
    else:
        query = """
            SELECT entity_id, state, last_updated
            FROM states
            ORDER BY last_updated
        """
        df = pd.read_sql_query(query, con)

    return _clean_df(df)


def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize timestamps, coerce state to numeric, drop non-numeric rows."""
    df["timestamp"] = pd.to_datetime(df["last_updated"], utc=True)
    df = df.drop(columns=["last_updated"])
    df = df.set_index("timestamp")

    df["state"] = pd.to_numeric(df["state"], errors="coerce")
    df = df.dropna(subset=["state"])

    df = df.sort_index()
    df.index.name = "timestamp"
    return df


def create_fake_sensor_df(
    start: str = "2024-01-01",
    end: str = "2024-12-31",
    seed: int = 0,
) -> pd.DataFrame:
    """
    Create fake environmental sensor data for testing and notebook demos.

    Generates hourly readings for three representative sensor types:
    - sensor.temperature_bedroom (°C)
    - sensor.humidity_bedroom (%)
    - sensor.co2_office (ppm)
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, end=end, freq="h", tz="UTC")
    n = len(dates)

    temp = rng.normal(20, 3, n).clip(10, 35)
    humidity = rng.normal(50, 10, n).clip(20, 90)
    co2 = rng.normal(800, 200, n).clip(400, 2000)

    rows = []
    for i, ts in enumerate(dates):
        rows.extend(
            [
                {"timestamp": ts, "entity_id": "sensor.temperature_bedroom", "state": temp[i]},
                {"timestamp": ts, "entity_id": "sensor.humidity_bedroom", "state": humidity[i]},
                {"timestamp": ts, "entity_id": "sensor.co2_office", "state": co2[i]},
            ]
        )

    df = pd.DataFrame(rows).set_index("timestamp")
    df.index.name = "timestamp"
    return df
