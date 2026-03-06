"""
Loads Abbott FreeStyle Libre continuous glucose monitor (CGM) data.

Export available via LibreView: https://www.libreview.com/
Go to Settings -> My Uploads -> Export Data

The exported CSV has a few header rows followed by glucose readings.
Two record types are present:
  - Historic Glucose (automatic readings every 15 minutes)
  - Scan Glucose (manual finger-stick scans)

Glucose values are in mmol/L in the EU export and mg/dL in the US export.
This loader normalizes everything to mmol/L (divide mg/dL by 18.018).
"""

from pathlib import Path

import pandas as pd

from ..config import load_config

# Abbott's CSV starts with metadata rows before the actual data header.
# The exact count varies by export version; we detect the header by
# looking for the "Device Timestamp" column.
_HEADER_COL = "Device Timestamp"

# Record type IDs in the Abbott export
_RECORD_HISTORIC = 0  # automatic 15-min reading
_RECORD_SCAN = 1      # manual scan


def load_glucose_df(
    path: Path | None = None,
    unit: str = "mmol/L",
) -> pd.DataFrame:
    """
    Load FreeStyle Libre glucose data.

    Returns a DataFrame with columns:
    - glucose: blood glucose value (mmol/L by default)
    - record_type: 'historic' (automatic) or 'scan' (manual)

    Parameters
    ----------
    path:
        Path to the LibreView CSV export. Falls back to config if None.
    unit:
        Output unit. Either 'mmol/L' (default) or 'mg/dL'.
    """
    if path is None:
        config = load_config()
        path = Path(config["data"]["freestyle_libre"]).expanduser()
    else:
        path = Path(path).expanduser()

    if not path.exists():
        raise FileNotFoundError(f"FreeStyle Libre export not found at {path}")

    valid_units = {"mmol/L", "mg/dL"}
    if unit not in valid_units:
        raise ValueError(f"Invalid unit {unit!r}. Must be one of: {valid_units}")

    # Skip the metadata header rows — find the line with the column headers
    with open(path, encoding="utf-8-sig") as f:
        lines = f.readlines()

    header_line = next(
        (i for i, line in enumerate(lines) if _HEADER_COL in line),
        None,
    )
    if header_line is None:
        raise ValueError(
            f"Could not find '{_HEADER_COL}' column in {path}. "
            "Is this a valid FreeStyle Libre export?"
        )

    # index_col=False: Abbott exports often have a trailing comma, creating N+1
    # fields per row. Without this, pandas auto-assigns the first column as the
    # row index, shifting all column assignments by one.
    df = pd.read_csv(path, skiprows=header_line, low_memory=False, index_col=False)

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]

    # Parse timestamps
    df["timestamp"] = pd.to_datetime(df["Device Timestamp"], dayfirst=True, utc=True)
    df = df.set_index("timestamp").sort_index()

    # Determine if historic or scan reading and extract glucose value
    # Historic readings: column "Historic Glucose mmol/L" or "Historic Glucose mg/dL"
    # Scan readings: column "Scan Glucose mmol/L" or "Scan Glucose mg/dL"
    glucose_col_mmol = _find_col(df, ["Historic Glucose mmol/L", "Scan Glucose mmol/L"])
    glucose_col_mgdl = _find_col(df, ["Historic Glucose mg/dL", "Scan Glucose mg/dL"])

    if glucose_col_mmol:
        # EU export — values already in mmol/L
        historic_col = next((c for c in df.columns if "Historic Glucose" in c and "mmol" in c), None)
        scan_col = next((c for c in df.columns if "Scan Glucose" in c and "mmol" in c), None)
        factor = 1.0
    elif glucose_col_mgdl:
        # US export — convert mg/dL to mmol/L
        historic_col = next((c for c in df.columns if "Historic Glucose" in c and "mg" in c), None)
        scan_col = next((c for c in df.columns if "Scan Glucose" in c and "mg" in c), None)
        factor = 1 / 18.018
    else:
        raise ValueError(
            "Could not find glucose columns in export. "
            f"Available columns: {list(df.columns)}"
        )

    records = []
    if historic_col and historic_col in df.columns:
        hist = df[[historic_col]].dropna(subset=[historic_col]).copy()
        hist["glucose"] = pd.to_numeric(hist[historic_col], errors="coerce") * factor
        hist["record_type"] = "historic"
        records.append(hist[["glucose", "record_type"]])

    if scan_col and scan_col in df.columns:
        scan = df[[scan_col]].dropna(subset=[scan_col]).copy()
        scan["glucose"] = pd.to_numeric(scan[scan_col], errors="coerce") * factor
        scan["record_type"] = "scan"
        records.append(scan[["glucose", "record_type"]])

    if not records:
        raise ValueError("No glucose readings found in export.")

    result = pd.concat(records).sort_index()
    result = result.dropna(subset=["glucose"])

    if unit == "mg/dL":
        result["glucose"] = result["glucose"] * 18.018

    return result


def load_glucose_daily_df(path: Path | None = None) -> pd.DataFrame:
    """
    Load FreeStyle Libre data aggregated to daily stats.

    Returns a DataFrame indexed by date with columns:
    - glucose_mean: daily mean glucose (mmol/L)
    - glucose_min: daily min
    - glucose_max: daily max
    - glucose_std: daily std dev (variability)
    - time_in_range: fraction of readings in 3.9-10.0 mmol/L (standard TIR range)
    - n_readings: number of readings that day
    """
    df = load_glucose_df(path=path)

    daily = df.groupby(df.index.date).agg(
        glucose_mean=("glucose", "mean"),
        glucose_min=("glucose", "min"),
        glucose_max=("glucose", "max"),
        glucose_std=("glucose", "std"),
        n_readings=("glucose", "count"),
    )
    # Time in range (3.9–10.0 mmol/L)
    in_range = df[(df["glucose"] >= 3.9) & (df["glucose"] <= 10.0)]
    tir = in_range.groupby(in_range.index.date).size() / df.groupby(df.index.date).size()
    daily["time_in_range"] = tir

    daily.index = pd.to_datetime(daily.index, utc=True)
    daily.index.name = "timestamp"
    return daily


def create_fake_glucose_df(
    start: str = "2024-01-01",
    end: str = "2024-01-31",
    interval_minutes: int = 15,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Generate fake CGM data for testing and notebook demos.

    Simulates realistic glucose fluctuations around a stable baseline
    with meal-time spikes and overnight stability.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start=start, end=end, freq=f"{interval_minutes}min", tz="UTC")
    n = len(timestamps)

    # Baseline ~5.5 mmol/L with diurnal variation and random walk noise
    hours = timestamps.hour.to_numpy() + timestamps.minute.to_numpy() / 60
    # Meal spikes at 8h, 13h, 19h
    meal_effect = (
        2.0 * np.exp(-0.5 * ((hours - 8) % 24) ** 2)
        + 2.5 * np.exp(-0.5 * ((hours - 13) % 24) ** 2)
        + 2.0 * np.exp(-0.5 * ((hours - 19) % 24) ** 2)
    )
    noise = rng.normal(0, 0.3, n)
    glucose = (5.5 + meal_effect * 0.4 + noise).clip(3.0, 15.0)

    df = pd.DataFrame(
        {"glucose": glucose, "record_type": "historic"},
        index=timestamps,
    )
    df.index.name = "timestamp"
    return df


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None
