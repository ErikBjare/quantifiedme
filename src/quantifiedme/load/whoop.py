"""
Loads Whoop data.

Supports two export formats:

- **Standard CSV export** (current Whoop dashboard export): flat directory with
  ``sleeps.csv``, ``workouts.csv``, ``physiological_cycles.csv``, and
  ``journal_entries.csv``. Daily-aggregate data. No granular per-minute HR.
- **GDPR full export** (legacy, from https://privacy.whoop.com/): ``Health/``
  subdirectory with ``sleeps.csv`` (different schema, with ``during`` JSON
  tuple), ``metrics*.csv`` granular per-minute HR/accel/skin temp.

Format is auto-detected from directory contents.
"""

from datetime import timedelta
from pathlib import Path
from typing import Literal

import pandas as pd

from ..config import load_config

WhoopFormat = Literal["standard", "gdpr"]


def _whoop_dir() -> Path:
    return Path(load_config()["data"]["whoop"]).expanduser()


def _detect_format(d: Path) -> WhoopFormat:
    """Return 'standard' for new CSV export, 'gdpr' for legacy full-export.

    Detection: ``physiological_cycles.csv`` only exists in the standard export;
    a ``Health/`` subdirectory is the GDPR-format signature.
    """
    if (d / "physiological_cycles.csv").exists():
        return "standard"
    if (d / "Health").is_dir():
        return "gdpr"
    raise FileNotFoundError(
        f"Whoop export not recognized at {d}. "
        f"Expected either 'physiological_cycles.csv' (standard) "
        f"or 'Health/' subdir (GDPR)."
    )


# ── Standard CSV export (current Whoop dashboard export) ──────────────────────


def _to_utc(timestamp_col: pd.Series, tz_col: pd.Series) -> pd.Series:
    """Convert naive local timestamps + per-row tz offset → UTC.

    Whoop CSVs ship the local timestamp in one column and the offset in
    another (e.g. ``"2026-05-09 16:52:30"`` + ``"UTC+02:00"``). Concatenate
    into an ISO-with-offset string and let pandas convert to UTC.
    """
    # "UTC+02:00" → "+02:00"; pandas understands ISO with offset directly.
    offset = tz_col.str.replace("UTC", "", regex=False)
    iso = timestamp_col.astype(str) + offset.astype(str)
    return pd.to_datetime(iso, utc=True, errors="coerce")


def _load_sleep_standard(d: Path) -> pd.DataFrame:
    """Load daily sleep summary from new-format ``sleeps.csv``.

    Index: date (waking day, derived from ``Wake onset`` local date).
    Excludes naps from the main per-day record.
    """
    path = d / "sleeps.csv"
    if not path.exists():
        raise FileNotFoundError(f"Whoop sleeps.csv not found at {path}")

    # sleeps.csv columns (standard CSV export, as of 2026-05): "Cycle start time",
    # "Cycle end time", "Cycle timezone", "Sleep onset", "Wake onset",
    # "Sleep performance %", "Respiratory rate (rpm)", "Asleep duration (min)",
    # "In bed duration (min)", "Light sleep duration (min)",
    # "Deep (SWS) duration (min)", "REM duration (min)", "Awake duration (min)",
    # "Sleep need (min)", "Sleep debt (min)", "Sleep efficiency %",
    # "Sleep consistency %", "Nap"
    df = pd.read_csv(path)
    df = df[df["Nap"].astype(str).str.lower() != "true"].copy()

    # Wake onset is naive local time; use its local date as the index.
    wake_local = pd.to_datetime(df["Wake onset"], errors="coerce")
    df = df[wake_local.notna()].copy()
    wake_local = wake_local[wake_local.notna()]

    out = pd.DataFrame(
        {
            "score": df["Sleep performance %"],
            "duration": pd.to_timedelta(df["Asleep duration (min)"], unit="m"),
            "time_in_bed": pd.to_timedelta(df["In bed duration (min)"], unit="m"),
            "efficiency": df["Sleep efficiency %"],
            "consistency": df["Sleep consistency %"],
            "debt": df["Sleep debt (min)"],
            "respiratory_rate": df["Respiratory rate (rpm)"],
            "rem": pd.to_timedelta(df["REM duration (min)"], unit="m"),
            "deep": pd.to_timedelta(df["Deep (SWS) duration (min)"], unit="m"),
            "light": pd.to_timedelta(df["Light sleep duration (min)"], unit="m"),
            "awake": pd.to_timedelta(df["Awake duration (min)"], unit="m"),
        }
    )
    out.index = pd.to_datetime(wake_local.dt.date, utc=True)
    out.index.name = "timestamp"
    return out.sort_index()


def _load_cycles_standard(d: Path) -> pd.DataFrame:
    """Load daily cycle metrics from ``physiological_cycles.csv``.

    This is the daily HR / HRV / recovery / strain row — replaces the
    granular ``metrics*.csv`` loader from the GDPR export at daily resolution.
    """
    path = d / "physiological_cycles.csv"
    if not path.exists():
        raise FileNotFoundError(f"Whoop physiological_cycles.csv not found at {path}")

    # physiological_cycles.csv columns (standard CSV export, as of 2026-05):
    # "Cycle start time", "Cycle end time", "Cycle timezone", "Recovery score %",
    # "Resting heart rate (bpm)", "Heart rate variability (ms)",
    # "Skin temp (celsius)", "Blood oxygen %", "Day Strain", "Energy burned (cal)",
    # "Max HR (bpm)", "Average HR (bpm)", "Sleep onset", "Wake onset",
    # "Sleep performance %", "Respiratory rate (rpm)", "Asleep duration (min)",
    # "In bed duration (min)", "Light sleep duration (min)",
    # "Deep (SWS) duration (min)", "REM duration (min)", "Awake duration (min)",
    # "Sleep need (min)", "Sleep debt (min)", "Sleep efficiency %",
    # "Sleep consistency %"
    df = pd.read_csv(path)

    # Use Wake onset for date assignment (same convention as sleep)
    wake_local = pd.to_datetime(df["Wake onset"], errors="coerce")
    df = df[wake_local.notna()].copy()
    wake_local = wake_local[wake_local.notna()]

    out = pd.DataFrame(
        {
            "recovery": df["Recovery score %"],
            "resting_hr": df["Resting heart rate (bpm)"],
            "hrv": df["Heart rate variability (ms)"],
            "skin_temp": df["Skin temp (celsius)"],
            "spo2": df["Blood oxygen %"],
            "strain": df["Day Strain"],
            "energy_kcal": df["Energy burned (cal)"],
        }
    )
    out.index = pd.to_datetime(wake_local.dt.date, utc=True)
    out.index.name = "timestamp"
    return out.sort_index()


def _load_workouts_standard(d: Path) -> pd.DataFrame:
    """Load workout events from ``workouts.csv`` (event-level, one row per workout)."""
    path = d / "workouts.csv"
    if not path.exists():
        raise FileNotFoundError(f"Whoop workouts.csv not found at {path}")

    # workouts.csv columns (standard CSV export, as of 2026-05):
    # "Cycle start time", "Cycle end time", "Cycle timezone",
    # "Workout start time", "Workout end time", "Duration (min)",
    # "Activity name", "Activity Strain", "Energy burned (cal)",
    # "Max HR (bpm)", "Average HR (bpm)",
    # "HR Zone 1 %", "HR Zone 2 %", "HR Zone 3 %", "HR Zone 4 %", "HR Zone 5 %",
    # "GPS enabled"
    df = pd.read_csv(path)
    # Workout timestamps are naive local; convert using each row's "Cycle timezone".
    df["start"] = _to_utc(df["Workout start time"], df["Cycle timezone"])
    df["end"] = _to_utc(df["Workout end time"], df["Cycle timezone"])
    df["duration"] = pd.to_timedelta(df["Duration (min)"], unit="m")
    return df.rename(
        columns={
            "Activity name": "activity",
            "Activity Strain": "strain",
            "Energy burned (cal)": "energy_kcal",
            "Max HR (bpm)": "max_hr",
            "Average HR (bpm)": "avg_hr",
        }
    )[
        [
            "start",
            "end",
            "duration",
            "activity",
            "strain",
            "energy_kcal",
            "max_hr",
            "avg_hr",
        ]
    ]


def _load_journal_standard(d: Path) -> pd.DataFrame:
    """Load journal entries from ``journal_entries.csv``.

    Event-level: one row per question-answer. Includes free-text Notes column
    when present — callers wanting privacy-aware aggregation should pivot via
    :func:`load_journal_daily_df` instead.
    """
    path = d / "journal_entries.csv"
    if not path.exists():
        raise FileNotFoundError(f"Whoop journal_entries.csv not found at {path}")

    # journal_entries.csv columns (standard CSV export, as of 2026-05):
    # "Cycle start time", "Cycle end time", "Cycle timezone",
    # "Question text", "Answered yes" (bool string), "Notes" (free text, often empty)
    # Question text is full natural-language ("Engaged in sexual activity?",
    # "Have any caffeine?", etc.) — Whoop changes the set of questions over time.
    df = pd.read_csv(path)
    # Use Cycle end time (= wake morning) to match the wake-date convention
    # used by _load_sleep_standard / _load_cycles_standard. Using Cycle start
    # time would offset journal rows one day behind sleep/cycles in joins.
    df["cycle_end"] = pd.to_datetime(df["Cycle end time"], errors="coerce", utc=True)
    if "Notes" not in df.columns:
        df["Notes"] = pd.NA
    df = df.rename(
        columns={
            "Question text": "question",
            "Answered yes": "answered_yes",
            "Notes": "notes",
        }
    )
    return df[["cycle_end", "question", "answered_yes", "notes"]]


def load_journal_daily_df(include_notes: bool = False) -> pd.DataFrame:
    """Pivot journal entries to one row per day, one column per question.

    Returns a DataFrame with date index and one boolean column per Whoop
    journal question (e.g. ``slept_well``, ``stressed``). Question text is
    normalized to a snake_case key.

    Parameters
    ----------
    include_notes
        If True, also include a 'journal_notes' column with concatenated
        free-text notes for the day. Default False for privacy (notes can
        contain sensitive personal context).
    """
    raw = _load_journal_standard(_whoop_dir())
    raw["date"] = raw["cycle_end"].dt.date
    raw["question_key"] = raw["question"].apply(_question_to_key)

    pivot = raw.pivot_table(
        index="date",
        columns="question_key",
        values="answered_yes",
        aggfunc=lambda s: any(str(v).lower() == "true" for v in s),
    )
    pivot.index = pd.to_datetime(pivot.index, utc=True)
    pivot.index.name = "timestamp"

    if include_notes:
        notes_per_day = (
            raw.dropna(subset=["notes"])
            .groupby("date")["notes"]
            .apply(lambda s: " | ".join(str(n) for n in s if str(n).strip()))
        )
        # notes_per_day is indexed by Python date; pivot.index is a UTC
        # DatetimeIndex. Bring them into the same shape before assignment,
        # otherwise pandas aligns by value and silently fills all NaN.
        notes_per_day.index = pd.to_datetime(notes_per_day.index, utc=True)
        pivot["notes"] = notes_per_day

    return pivot.sort_index()


def _question_to_key(question: str) -> str:
    """Normalize a Whoop journal question to a stable snake_case column key.

    Examples
    --------
    >>> _question_to_key("Engaged in sexual activity?")
    'engaged_in_sexual_activity'
    >>> _question_to_key("Have any caffeine?")
    'caffeine'
    >>> _question_to_key("Felt stressed today?")
    'stressed'
    """
    import re

    if not isinstance(question, str):
        return "unknown"
    # Strip leading verbs/auxiliaries to get to the noun phrase
    q = question.lower().strip().rstrip("?").strip()
    for prefix in (
        "have any ",
        "had any ",
        "any ",
        "felt ",
        "feel ",
        "did you ",
        "do you ",
        "are you ",
    ):
        if q.startswith(prefix):
            q = q[len(prefix) :]
            break
    # Remove trailing "today" / "yesterday"
    q = re.sub(r"\s+(today|yesterday)$", "", q)
    # Normalize to snake_case
    q = re.sub(r"[^a-z0-9]+", "_", q).strip("_")
    return q or "unknown"


# ── GDPR full export (legacy) ─────────────────────────────────────────────────


def _load_heartrate_gdpr(d: Path) -> pd.DataFrame:
    """Granular per-minute HR from the GDPR-export ``Health/metrics*.csv`` files."""
    health = d / "Health"
    if not health.is_dir():
        raise FileNotFoundError(f"Whoop GDPR Health/ dir not found at {health}")

    # Health/metrics-*.csv columns (GDPR full export): hr, accel_x, accel_y,
    # accel_z, skin_temp, ts — one row per minute. Multiple files split by
    # date range; concatenated here.
    dfs = []
    for file in health.iterdir():
        if file.name.startswith("metrics") and file.name.endswith(".csv"):
            df = pd.read_csv(file, parse_dates=True)
            dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No metrics*.csv files in {health}")

    df = pd.concat(dfs, ignore_index=True)
    df = df.set_index(pd.DatetimeIndex(df["ts"]))
    del df["ts"]
    # Only HR kept here; accel_*/skin_temp available in raw df if needed
    df = df[["hr"]].sort_index()
    df.index.name = "timestamp"
    return df


def _load_sleep_gdpr(d: Path) -> pd.DataFrame:
    """Legacy GDPR-format sleep load (uses ``during`` JSON-tuple column)."""
    path = d / "Health" / "sleeps.csv"
    if not path.exists():
        raise FileNotFoundError(f"Whoop GDPR sleeps.csv not found at {path}")

    # Health/sleeps.csv columns (GDPR full export): created_at, updated_at,
    # activity_id, score, quality_duration, latency, max_heart_rate,
    # average_heart_rate, debt_pre, debt_post, need_from_strain, sleep_need,
    # habitual_sleep_need, disturbances, time_in_bed, light_sleep_duration,
    # slow_wave_sleep_duration, rem_sleep_duration, cycles_count,
    # wake_duration, arousal_time, no_data_duration, in_sleep_efficiency,
    # credit_from_naps, hr_baseline, respiratory_rate, sleep_consistency,
    # algo_version, projected_score, projected_sleep, optimal_sleep_times,
    # kilojoules, user_id, during, timezone_offset, survey_response_id,
    # percent_recorded, auto_detected, state, responded, team_act_id, source,
    # is_significant, is_normal, is_nap
    # The "during" column is a Postgres tstzrange string with start/end
    # iso-format timestamps; we extract those as start/end columns.
    import ast

    df = pd.read_csv(path, parse_dates=True)

    def parse_during(x):
        # tstzrange ships as e.g. `["2022-01-01T00:00:00","2022-01-01T08:00:00")`.
        # Rewrite the closing paren to make it a Python-syntax list, then
        # literal_eval (safer than eval on file content).
        try:
            return ast.literal_eval(x.replace(")", "]"))
        except (ValueError, SyntaxError):
            raise ValueError(f"Could not parse `during` value: {x!r}") from None

    df["start"] = pd.to_datetime(df["during"].apply(lambda x: parse_during(x)[0]))
    df["end"] = pd.to_datetime(df["during"].apply(lambda x: parse_during(x)[1]))
    df["duration"] = df["end"] - df["start"]

    df = df[["start", "end", "duration", "score"]]

    # Hard-coded 8-hour offset retained from legacy code — works for Erik's
    # historical data (UTC+8 era). Re-evaluate if older data with different
    # timezones is loaded.
    offset = timedelta(hours=8)
    df = df.set_index(
        pd.to_datetime(pd.DatetimeIndex(df["start"] - offset).date, utc=True)  # type: ignore[arg-type]
    )
    df = df.sort_index()
    df.index.name = "timestamp"
    return df


# ── Public API (format-dispatching) ───────────────────────────────────────────


def load_heartrate_df() -> pd.DataFrame:
    """Load granular HR data. Only available for GDPR-format exports.

    Standard exports don't include per-minute HR — use :func:`load_cycles_df`
    for daily HR/HRV summaries instead.
    """
    d = _whoop_dir()
    fmt = _detect_format(d)
    if fmt == "gdpr":
        return _load_heartrate_gdpr(d)
    raise NotImplementedError(
        f"Granular heartrate data not available in '{fmt}' Whoop export format. "
        f"Use load_cycles_df() for daily HR/HRV summary, or request a GDPR "
        f"full export from https://privacy.whoop.com/."
    )


def load_sleep_df() -> pd.DataFrame:
    """Load daily sleep summary. Works for both export formats."""
    d = _whoop_dir()
    fmt = _detect_format(d)
    if fmt == "standard":
        return _load_sleep_standard(d)
    return _load_sleep_gdpr(d)


def load_cycles_df() -> pd.DataFrame:
    """Load daily physiological cycle summary (recovery, HRV, RHR, strain).

    Only available in the standard export format.
    """
    d = _whoop_dir()
    if _detect_format(d) != "standard":
        raise NotImplementedError(
            "Cycle data only available in standard Whoop export, not GDPR full export."
        )
    return _load_cycles_standard(d)


def load_workouts_df() -> pd.DataFrame:
    """Load workout events. Only available in the standard export format."""
    d = _whoop_dir()
    if _detect_format(d) != "standard":
        raise NotImplementedError(
            "Workout data only available in standard Whoop export, not GDPR full export."
        )
    return _load_workouts_standard(d)
