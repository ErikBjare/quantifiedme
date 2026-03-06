"""
Loads SwitchBot environmental sensor data.

SwitchBot Meter, MeterPlus, and CO2 Sensor devices log temperature,
humidity, and CO2. Data can be exported via the SwitchBot app:
  App -> Device -> History -> Export

The export produces a CSV file per device.

Typical columns:
  - Meter/MeterPlus: Time, Temperature(°C), Humidity(%)
  - CO2 Sensor:      Time, Temperature(°C), Humidity(%), CO2(ppm)

Multiple device exports (e.g., bedroom + office) can be loaded and
merged into a single DataFrame with a device label column.
"""

from pathlib import Path

import pandas as pd

from ..config import load_config


def load_environment_df(
    path: Path | str | None = None,
    device_name: str | None = None,
) -> pd.DataFrame:
    """
    Load SwitchBot environmental sensor export (single device).

    Returns a DataFrame indexed by timestamp with columns:
    - temperature_c: temperature in °C
    - humidity_pct: relative humidity (0–100)
    - co2_ppm: CO2 concentration in ppm (only if device supports it)

    Parameters
    ----------
    path:
        Path to the SwitchBot CSV export. Falls back to config if None.
        Config key: ``data.switchbot`` (single device) or
        ``data.switchbot_devices`` (dict of name -> path).
    device_name:
        Optional label to add as a column (useful when combining devices).
    """
    if path is None:
        config = load_config()
        cfg = config.get("data", {}).get("switchbot")
        if cfg is None:
            raise KeyError(
                "No switchbot data path in config. "
                "Add 'switchbot = \"~/path/to/export.csv\"' under [data]."
            )
        path = Path(cfg).expanduser()
    else:
        path = Path(path).expanduser()

    assert path.exists(), f"SwitchBot export not found at {path}"

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # Detect timestamp column (SwitchBot uses "Time" or "Date/Time")
    time_col = _find_col(df, ["Time", "Date/Time", "Timestamp", "Date"])
    if time_col is None:
        raise ValueError(
            f"Could not find timestamp column in {path}. "
            f"Available columns: {list(df.columns)}"
        )

    df["timestamp"] = pd.to_datetime(df[time_col], utc=True)
    df = df.drop(columns=[time_col])
    df = df.set_index("timestamp").sort_index()

    # Normalize column names
    col_map: dict[str, str] = {}
    for col in df.columns:
        lc = col.lower().strip()
        if "temperature" in lc or "temp" in lc:
            col_map[col] = "temperature_c"
        elif "humidity" in lc or "humid" in lc:
            col_map[col] = "humidity_pct"
        elif "co2" in lc or "carbon" in lc:
            col_map[col] = "co2_ppm"
        else:
            # Keep unrecognized columns but normalize name
            col_map[col] = lc.replace(" ", "_").replace("(", "").replace(")", "").replace("%", "pct").replace("/", "_")

    df = df.rename(columns=col_map)

    # Keep only numeric columns
    df = df.select_dtypes(include="number")

    if device_name is not None:
        df["device"] = device_name

    return df


def load_environment_devices_df(devices: dict[str, Path | str] | None = None) -> pd.DataFrame:
    """
    Load and combine multiple SwitchBot device exports.

    Parameters
    ----------
    devices:
        Dict mapping device name to CSV path. Falls back to
        ``data.switchbot_devices`` in config if None.

    Returns a single DataFrame with a ``device`` column to distinguish sources.
    """
    if devices is None:
        config = load_config()
        raw = config.get("data", {}).get("switchbot_devices", {})
        devices = {name: Path(p).expanduser() for name, p in raw.items()}

    if not devices:
        raise ValueError(
            "No devices provided. Pass a dict or configure "
            "'switchbot_devices' under [data] in config."
        )

    dfs = []
    for name, path in devices.items():
        df = load_environment_df(path=path, device_name=name)
        dfs.append(df)

    return pd.concat(dfs).sort_index()


def load_environment_daily_df(
    path: Path | str | None = None,
    device_name: str | None = None,
) -> pd.DataFrame:
    """
    Load SwitchBot data aggregated to daily statistics.

    Returns a DataFrame indexed by date with mean/min/max for each
    sensor channel (temperature, humidity, and CO2 if available).
    """
    df = load_environment_df(path=path, device_name=device_name)

    # Drop device column before aggregating if present
    numeric_df = df.select_dtypes(include="number")

    daily = numeric_df.groupby(numeric_df.index.date).agg(["mean", "min", "max"])
    daily.columns = ["_".join(c) for c in daily.columns]
    daily.index = pd.to_datetime(daily.index, utc=True)
    daily.index.name = "timestamp"
    return daily


def create_fake_environment_df(
    start: str = "2024-01-01",
    end: str = "2024-01-31",
    interval_minutes: int = 5,
    include_co2: bool = True,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Generate fake SwitchBot environmental data for testing and notebooks.

    Simulates realistic diurnal patterns:
    - Temperature: warmer in the afternoon (~21°C baseline)
    - Humidity: slightly higher at night
    - CO2 (if include_co2): higher during occupied hours, spikes around meals
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start=start, end=end, freq=f"{interval_minutes}min", tz="UTC")
    n = len(timestamps)
    hours = timestamps.hour.to_numpy() + timestamps.minute.to_numpy() / 60

    # Temperature: baseline 20°C, +2°C afternoon peak, small noise
    temp = 20.0 + 2.0 * np.sin((hours - 6) * np.pi / 12) + rng.normal(0, 0.3, n)
    temp = temp.clip(15.0, 28.0)

    # Humidity: baseline 50%, slightly higher overnight
    humid = 50.0 - 5.0 * np.sin((hours - 6) * np.pi / 12) + rng.normal(0, 2.0, n)
    humid = humid.clip(20.0, 85.0)

    data: dict[str, object] = {"temperature_c": temp, "humidity_pct": humid}

    if include_co2:
        # CO2: outdoor ~420 ppm baseline; elevated during occupied daytime hours
        occupied = ((hours >= 8) & (hours <= 22)).astype(float)
        co2 = 420.0 + 200.0 * occupied + rng.normal(0, 30.0, n)
        co2 = co2.clip(350.0, 2000.0)
        data["co2_ppm"] = co2

    df = pd.DataFrame(data, index=timestamps)
    df.index.name = "timestamp"
    return df


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None
