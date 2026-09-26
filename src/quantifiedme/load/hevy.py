"""
Loads strength-training data from a Hevy (https://www.hevyapp.com) CSV export.

Export from the app: Profile → Settings → Export & Import Data → Export Workouts.
This produces ``workout_data.csv`` with one row per logged set::

    "title","start_time","end_time","description","exercise_title","superset_id",
    "exercise_notes","set_index","set_type","weight_kg","reps","distance_km",
    "duration_seconds","rpe"

Timestamps are naive local time formatted like ``"28 May 2026, 18:05"``.

Configure the path in ``config.toml``::

    [data]
    hevy = "~/Downloads/workout_data.csv"
"""

from pathlib import Path

import pandas as pd

from ..config import load_config

HEVY_DATE_FMT = "%d %b %Y, %H:%M"

# Hevy set types; warm-up sets are excluded from working-set/volume totals
WARMUP_SET_TYPES = {"warmup"}

DAILY_COLUMNS = ["sessions", "sets", "reps", "volume_kg", "minutes"]


def _parse_time(col: pd.Series) -> pd.Series:
    """Parse Hevy timestamps, falling back to pandas inference for other formats."""
    parsed = pd.to_datetime(col, format=HEVY_DATE_FMT, errors="coerce")
    if parsed.isna().any():
        fallback = pd.to_datetime(col[parsed.isna()], format="mixed", errors="coerce")
        parsed = parsed.fillna(fallback)
    return parsed


def load_sets_df(path: Path | str | None = None) -> pd.DataFrame:
    """Load the Hevy export as one row per set.

    Adds parsed ``start``/``end`` (naive local) and a ``workout_id`` column
    (a workout is identified by its title + start time, since the export has
    no id column).
    """
    if path is None:
        config = load_config()
        path = config["data"]["hevy"]
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Hevy export not found at {path}")

    df = pd.read_csv(path)
    df["start"] = _parse_time(df["start_time"])
    df["end"] = _parse_time(df["end_time"])
    df = df.dropna(subset=["start"])
    for col in ["weight_kg", "reps", "distance_km", "duration_seconds", "rpe"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["workout_id"] = df["title"].astype(str) + "@" + df["start"].astype(str)
    return df.reset_index(drop=True)


def sets_to_daily_df(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-set rows into one row per (local) day.

    Columns:

    - ``sessions``: number of workouts started that day
    - ``sets``: working sets (warm-up sets excluded)
    - ``reps``: reps across working sets
    - ``volume_kg``: sum of weight × reps across working sets (kg); bodyweight
      and cardio sets without a weight contribute 0
    - ``minutes``: total workout duration (end − start per workout)

    Only days with at least one workout are included.
    """
    if df.empty:
        out = pd.DataFrame(columns=DAILY_COLUMNS, dtype=float)
        out.index = pd.DatetimeIndex([], name="date")
        return out

    df = df.copy()
    df["date"] = df["start"].dt.normalize()
    working = df[~df["set_type"].astype(str).str.lower().isin(WARMUP_SET_TYPES)]
    working = working.assign(
        volume=working["weight_kg"].fillna(0) * working["reps"].fillna(0)
    )

    workouts = df.drop_duplicates("workout_id")
    workout_minutes = (workouts["end"] - workouts["start"]).dt.total_seconds() / 60

    daily = pd.DataFrame(
        {
            "sessions": workouts.groupby("date").size(),
            "sets": working.groupby("date").size(),
            "reps": working.groupby("date")["reps"].sum(),
            "volume_kg": working.groupby("date")["volume"].sum(),
            "minutes": workout_minutes.groupby(workouts["date"]).sum(),
        }
    )
    daily[["sets", "reps", "volume_kg"]] = daily[["sets", "reps", "volume_kg"]].fillna(
        0
    )
    daily[["sessions", "sets"]] = daily[["sessions", "sets"]].astype(int)
    daily.index = pd.DatetimeIndex(daily.index, name="date")
    return daily[DAILY_COLUMNS].sort_index()


def load_daily_df(path: Path | str | None = None) -> pd.DataFrame:
    """Load daily strength-training summary from the Hevy export."""
    return sets_to_daily_df(load_sets_df(path))
