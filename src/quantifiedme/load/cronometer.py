"""
Loads Cronometer nutrition data.

Cronometer exports are available at: https://cronometer.com/
Go to Settings -> Account -> Export Data

Two export formats are supported:
- Daily Summary: One row per day with macro/micronutrient totals
- Servings: One row per food item logged

The daily summary CSV (``servings.csv`` with date-grouped aggregation, or
the dedicated daily export) is what most users will use.
"""

from pathlib import Path

import pandas as pd

from ..config import load_config


def load_nutrition_df(path: Path | None = None) -> pd.DataFrame:
    """
    Load Cronometer daily nutrition summary.

    Returns a DataFrame indexed by date with columns for key macros:
    - energy_kcal
    - protein_g
    - carbs_g
    - fat_g
    - fiber_g (if available)
    - sugar_g (if available)

    The CSV export from Cronometer uses columns like:
    "Date","Energy (kcal)","Protein (g)","Net Carbs (g)","Carbs (g)","Fat (g)",...
    """
    if path is None:
        config = load_config()
        path = Path(config["data"]["cronometer"]).expanduser()
    else:
        path = Path(path).expanduser()

    if not path.exists():
        raise FileNotFoundError(f"Cronometer export not found at {path}")

    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.rename(columns={"Date": "date"})
    df = df.set_index(pd.DatetimeIndex(df["date"], tz="UTC"))
    df = df.drop(columns=["date"])

    # Normalize column names to snake_case with units
    col_map: dict[str, str] = {}
    for col in df.columns:
        normalized = col.lower()
        normalized = normalized.replace(" ", "_")
        normalized = normalized.replace("(", "").replace(")", "")
        col_map[col] = normalized

    df = df.rename(columns=col_map)

    # Keep only numeric columns (drop any leftover string cols)
    df = df.select_dtypes(include="number")

    # Sort chronologically
    df = df.sort_index()
    df.index.name = "timestamp"

    return df


def load_servings_df(path: Path | None = None) -> pd.DataFrame:
    """
    Load Cronometer servings log (individual food entries).

    Returns a DataFrame with each logged food item, indexed by date.
    Useful for meal-level analysis.
    """
    if path is None:
        config = load_config()
        path = Path(
            config["data"].get("cronometer_servings", config["data"]["cronometer"])
        ).expanduser()
    else:
        path = Path(path).expanduser()

    if not path.exists():
        raise FileNotFoundError(f"Cronometer servings export not found at {path}")

    df = pd.read_csv(path, parse_dates=["Day"])
    df = df.rename(columns={"Day": "date"})
    df["timestamp"] = pd.to_datetime(df["date"], utc=True)
    df = df.set_index("timestamp")

    return df


def create_fake_nutrition_df(
    start: str = "2024-01-01",
    end: str = "2024-12-31",
    seed: int = 0,
) -> pd.DataFrame:
    """
    Create fake nutrition data for testing and notebook demos.

    Generates plausible macro values with natural variation.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, end=end, freq="D", tz="UTC")

    n = len(dates)
    # Baseline values with realistic variation
    energy = rng.normal(2200, 300, n).clip(1200, 3500)
    protein = rng.normal(120, 25, n).clip(40, 250)
    fat = rng.normal(80, 20, n).clip(20, 180)
    carbs = rng.normal(250, 60, n).clip(50, 500)
    fiber = rng.normal(25, 8, n).clip(5, 60)
    sugar = rng.normal(50, 15, n).clip(5, 120)

    df = pd.DataFrame(
        {
            "energy_kcal": energy,
            "protein_g": protein,
            "fat_g": fat,
            "carbs_g": carbs,
            "fiber_g": fiber,
            "sugar_g": sugar,
        },
        index=dates,
    )
    df.index.name = "timestamp"
    return df
