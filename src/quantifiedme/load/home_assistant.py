"""
Loads environmental sensor data from Home Assistant.

Two access modes:
- **SQLite** (local): reads the HA SQLite database directly (``load_sensor_df``)
- **REST API** (remote): queries the HA history API over HTTP (``load_sensor_df_api``)

Home Assistant stores all sensor history in a local SQLite database at:
  ~/.homeassistant/home-assistant_v2.db

This loader reads the `states` table (with `states_meta` in newer HA versions 2023+)
to extract time-series data for environmental sensors like temperature, humidity, CO2.

Rather than per-device CSV parsers, this loader covers all sensors that Erik routes
through Home Assistant — current and future sensors automatically included.
"""

import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from ..config import load_config


def load_sensor_df(
    path: Path | None = None,
    entity_ids: list[str] | None = None,
    units: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Load environmental sensor data from Home Assistant's SQLite database.

    Args:
        path: Path to the Home Assistant SQLite database file.
              Defaults to config["data"]["home_assistant"].
        entity_ids: Optional list of entity IDs to filter (e.g.,
                    ["sensor.temperature_bedroom", "sensor.co2_office"]).
                    If None, all numeric sensor states are returned.
        units: Optional mapping from entity_id to unit string (e.g.,
               {"sensor.temperature_bedroom": "°C", "sensor.co2_office": "ppm"}).
               If provided, a ``unit`` column is populated; unknown entities get NaN.

    Returns:
        DataFrame indexed by UTC timestamp with columns:
        - entity_id: HA entity ID
        - state: numeric sensor value
        - unit: unit of measurement (NaN when units mapping not provided or entity unknown)

    Non-numeric states (e.g., 'unavailable', 'unknown') are dropped.
    """
    if path is None:
        config = load_config()
        path = Path(config["data"]["home_assistant"]).expanduser()
    else:
        path = Path(path).expanduser()

    if not path.exists():
        raise FileNotFoundError(f"Home Assistant database not found at {path}")

    if entity_ids is not None and len(entity_ids) == 0:
        return pd.DataFrame(columns=["entity_id", "state", "unit"]).set_index(
            pd.DatetimeIndex([], tz="UTC", name="timestamp")
        )

    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as con:
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

    df["unit"] = df["entity_id"].map(units) if units is not None else None
    return df


def _load_modern_schema(
    con: sqlite3.Connection, entity_ids: list[str] | None
) -> pd.DataFrame:
    """Load from HA 2023.x+ schema with states_meta table."""
    if entity_ids is not None:
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
    if entity_ids is not None:
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


def load_sensor_df_api(
    url: str,
    token: str,
    entity_ids: list[str] | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    units: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Load environmental sensor data from Home Assistant's REST API.

    Uses the ``/api/history/period`` endpoint, which is available on any
    networked HA instance and does not require local database access.

    Args:
        url: Base URL of the Home Assistant instance, e.g.
             ``"http://homeassistant.local:8123"``.
        token: Long-lived access token (Settings → Profile → Long-lived access tokens).
        entity_ids: Optional list of entity IDs to fetch. If None, HA returns
                    history for all entities (can be very large).
        start: Start of the history window (UTC). Defaults to 24 hours ago.
        end: End of the history window (UTC). Defaults to now.
        units: Optional mapping from entity_id to unit string. If provided,
               a ``unit`` column is populated; unknown entities get NaN.

    Returns:
        DataFrame indexed by UTC timestamp with columns:
        - entity_id: HA entity ID
        - state: numeric sensor value
        - unit: unit of measurement (NaN when units mapping not provided or entity unknown)

    Non-numeric states (e.g., 'unavailable', 'unknown') are dropped.

    Raises:
        ImportError: if the ``requests`` package is not installed.
        requests.HTTPError: on non-2xx responses from HA.
    """
    try:
        import requests
    except ImportError as e:
        raise ImportError("requests is required for REST API access: pip install requests") from e

    if start is None:
        start = datetime.now(tz=timezone.utc) - timedelta(days=1)
    if end is None:
        end = datetime.now(tz=timezone.utc)

    start_str = start.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    end_str = end.strftime("%Y-%m-%dT%H:%M:%S+00:00")

    api_url = f"{url.rstrip('/')}/api/history/period/{start_str}"
    params: dict[str, str] = {
        "end_time": end_str,
        "minimal_response": "true",
        "significant_changes_only": "false",
    }
    if entity_ids is not None:
        if len(entity_ids) == 0:
            return pd.DataFrame(columns=["entity_id", "state", "unit"]).set_index(
                pd.DatetimeIndex([], tz="UTC", name="timestamp")
            )
        params["filter_entity_id"] = ",".join(entity_ids)

    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(api_url, params=params, headers=headers, timeout=30)
    response.raise_for_status()

    # Response is a list of lists (one per entity) of state objects
    data = response.json()
    rows = []
    for entity_history in data:
        for state_obj in entity_history:
            rows.append(
                {
                    "entity_id": state_obj["entity_id"],
                    "state": state_obj["state"],
                    "last_updated": state_obj["last_updated"],
                }
            )

    if not rows:
        return pd.DataFrame(columns=["entity_id", "state", "unit"]).set_index(
            pd.DatetimeIndex([], tz="UTC", name="timestamp")
        )

    df = _clean_df(pd.DataFrame(rows))
    df["unit"] = df["entity_id"].map(units) if units is not None else None
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
                {"timestamp": ts, "entity_id": "sensor.temperature_bedroom", "state": temp[i], "unit": "°C"},
                {"timestamp": ts, "entity_id": "sensor.humidity_bedroom", "state": humidity[i], "unit": "%"},
                {"timestamp": ts, "entity_id": "sensor.co2_office", "state": co2[i], "unit": "ppm"},
            ]
        )

    df = pd.DataFrame(rows).set_index("timestamp")
    df.index.name = "timestamp"
    return df
