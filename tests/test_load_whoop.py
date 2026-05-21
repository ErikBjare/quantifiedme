"""Tests for the Whoop data loader.

Covers both the standard CSV export format (current Whoop dashboard) and the
GDPR full-export format (legacy). Standard-format paths are tested with fixture
CSVs that match the actual column schemas Whoop ships (verified against Erik's
2026-05-13 export).
"""

from pathlib import Path

import pandas as pd
import pytest

from quantifiedme.config import load_config
from quantifiedme.load import whoop
from quantifiedme.load.whoop import (
    _detect_format,
    _load_cycles_standard,
    _load_journal_standard,
    _load_sleep_standard,
    _load_workouts_standard,
    _question_to_key,
    load_cycles_df,
    load_heartrate_df,
    load_journal_daily_df,
    load_sleep_df,
    load_workouts_df,
)

has_whoop_config = load_config().get("data", {}).get("whoop", False)


# ── Fixture CSV content (matches Whoop standard export schemas, 2026-05) ──────


SLEEPS_CSV = """\
Cycle start time,Cycle end time,Cycle timezone,Sleep onset,Wake onset,Sleep performance %,Respiratory rate (rpm),Asleep duration (min),In bed duration (min),Light sleep duration (min),Deep (SWS) duration (min),REM duration (min),Awake duration (min),Sleep need (min),Sleep debt (min),Sleep efficiency %,Sleep consistency %,Nap
2026-05-09 23:00:00,2026-05-10 07:00:00,UTC+02:00,2026-05-09 23:15:00,2026-05-10 07:00:00,92,15.2,450,475,210,110,130,25,480,30,94,85,false
2026-05-10 23:30:00,2026-05-11 06:45:00,UTC+02:00,2026-05-10 23:45:00,2026-05-11 06:45:00,88,15.5,415,435,200,100,115,20,480,65,95,80,false
2026-05-11 13:00:00,2026-05-11 13:45:00,UTC+02:00,2026-05-11 13:05:00,2026-05-11 13:40:00,50,15.0,35,40,20,5,10,5,0,0,87,0,true
"""

CYCLES_CSV = """\
Cycle start time,Cycle end time,Cycle timezone,Recovery score %,Resting heart rate (bpm),Heart rate variability (ms),Skin temp (celsius),Blood oxygen %,Day Strain,Energy burned (cal),Max HR (bpm),Average HR (bpm),Sleep onset,Wake onset,Sleep performance %,Respiratory rate (rpm),Asleep duration (min),In bed duration (min),Light sleep duration (min),Deep (SWS) duration (min),REM duration (min),Awake duration (min),Sleep need (min),Sleep debt (min),Sleep efficiency %,Sleep consistency %
2026-05-09 23:00:00,2026-05-10 23:00:00,UTC+02:00,72,52,58,33.1,96,12.5,2600,165,75,2026-05-09 23:15:00,2026-05-10 07:00:00,92,15.2,450,475,210,110,130,25,480,30,94,85
2026-05-10 23:30:00,2026-05-11 23:30:00,UTC+02:00,65,55,52,33.4,95,14.2,2800,172,78,2026-05-10 23:45:00,2026-05-11 06:45:00,88,15.5,415,435,200,100,115,20,480,65,95,80
"""

WORKOUTS_CSV = """\
Cycle start time,Cycle end time,Cycle timezone,Workout start time,Workout end time,Duration (min),Activity name,Activity Strain,Energy burned (cal),Max HR (bpm),Average HR (bpm),HR Zone 1 %,HR Zone 2 %,HR Zone 3 %,HR Zone 4 %,HR Zone 5 %,GPS enabled
2026-05-10 23:00:00,2026-05-11 23:00:00,UTC+02:00,2026-05-10 18:00:00,2026-05-10 19:00:00,60,Running,9.5,520,175,142,5,15,45,30,5,true
"""

JOURNAL_CSV = """\
Cycle start time,Cycle end time,Cycle timezone,Question text,Answered yes,Notes
2026-05-09 23:00:00,2026-05-10 23:00:00,UTC+02:00,Have any caffeine?,true,
2026-05-09 23:00:00,2026-05-10 23:00:00,UTC+02:00,Felt stressed today?,false,
2026-05-09 23:00:00,2026-05-10 23:00:00,UTC+02:00,Engaged in sexual activity?,true,a private note
2026-05-10 23:30:00,2026-05-11 23:30:00,UTC+02:00,Have any caffeine?,false,
2026-05-10 23:30:00,2026-05-11 23:30:00,UTC+02:00,Felt stressed today?,true,big deadline
"""


@pytest.fixture
def standard_export(tmp_path: Path) -> Path:
    """Create a complete standard-format Whoop export directory."""
    (tmp_path / "sleeps.csv").write_text(SLEEPS_CSV)
    (tmp_path / "physiological_cycles.csv").write_text(CYCLES_CSV)
    (tmp_path / "workouts.csv").write_text(WORKOUTS_CSV)
    (tmp_path / "journal_entries.csv").write_text(JOURNAL_CSV)
    return tmp_path


@pytest.fixture
def gdpr_export(tmp_path: Path) -> Path:
    """Create a minimal GDPR-format Whoop export (Health/ subdir only)."""
    (tmp_path / "Health").mkdir()
    return tmp_path


@pytest.fixture
def patched_whoop_dir(monkeypatch: pytest.MonkeyPatch, standard_export: Path) -> Path:
    """Point the public API at a fixture directory by patching _whoop_dir."""
    monkeypatch.setattr(whoop, "_whoop_dir", lambda: standard_export)
    return standard_export


# ── Format detection ──────────────────────────────────────────────────────────


def test_detect_format_standard(standard_export: Path) -> None:
    assert _detect_format(standard_export) == "standard"


def test_detect_format_gdpr(gdpr_export: Path) -> None:
    assert _detect_format(gdpr_export) == "gdpr"


def test_detect_format_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Whoop export not recognized"):
        _detect_format(tmp_path)


# ── Standard-format loaders ───────────────────────────────────────────────────


def test_load_sleep_standard(standard_export: Path) -> None:
    df = _load_sleep_standard(standard_export)

    # Nap row excluded → 2 rows
    assert len(df) == 2
    assert df.index.name == "timestamp"
    assert isinstance(df.index, pd.DatetimeIndex)
    assert str(df.index.tz) == "UTC"
    assert df.index.is_monotonic_increasing

    # Column presence
    for col in (
        "score",
        "duration",
        "time_in_bed",
        "efficiency",
        "consistency",
        "debt",
        "respiratory_rate",
        "rem",
        "deep",
        "light",
        "awake",
    ):
        assert col in df.columns, f"missing column: {col}"

    # First row sanity (2026-05-10, score=92, 450 min asleep)
    assert df.iloc[0]["score"] == 92
    assert df.iloc[0]["duration"] == pd.Timedelta(minutes=450)


def test_load_cycles_standard(standard_export: Path) -> None:
    df = _load_cycles_standard(standard_export)

    assert len(df) == 2
    assert df.index.name == "timestamp"
    assert isinstance(df.index, pd.DatetimeIndex)
    assert str(df.index.tz) == "UTC"

    for col in (
        "recovery",
        "resting_hr",
        "hrv",
        "skin_temp",
        "spo2",
        "strain",
        "energy_kcal",
    ):
        assert col in df.columns

    assert df.iloc[0]["recovery"] == 72
    assert df.iloc[0]["resting_hr"] == 52
    assert df.iloc[0]["hrv"] == 58


def test_load_workouts_standard(standard_export: Path) -> None:
    df = _load_workouts_standard(standard_export)

    assert len(df) == 1
    assert list(df.columns) == [
        "start",
        "end",
        "duration",
        "activity",
        "strain",
        "energy_kcal",
        "max_hr",
        "avg_hr",
    ]
    # Timezone conversion: 2026-05-10 18:00 UTC+02:00 → 16:00 UTC
    assert df.iloc[0]["start"] == pd.Timestamp("2026-05-10 16:00:00", tz="UTC")
    assert df.iloc[0]["end"] == pd.Timestamp("2026-05-10 17:00:00", tz="UTC")
    assert df.iloc[0]["duration"] == pd.Timedelta(minutes=60)
    assert df.iloc[0]["activity"] == "Running"


def test_load_journal_standard_raw(standard_export: Path) -> None:
    df = _load_journal_standard(standard_export)

    assert len(df) == 5
    assert list(df.columns) == ["cycle_end", "question", "answered_yes", "notes"]
    assert str(df["cycle_end"].dt.tz) == "UTC"


def test_journal_date_index_aligns_with_sleep(patched_whoop_dir: Path) -> None:
    """Journal entries must be indexed by the same wake-date convention as sleep
    and cycles loaders — otherwise all_df.py joins drop every row.

    Greptile P1 (2026-05-19): cycle_start indexes journal one day behind sleep.
    """
    sleep = load_sleep_df()
    journal = load_journal_daily_df()

    # Both journal rows should match a sleep row by date. (Sleep has 3 rows
    # because of the nap; journal has 2 rows from the 2 distinct cycle ends.)
    common = sleep.index.intersection(journal.index)
    assert len(common) == len(journal), (
        f"Journal dates {list(journal.index)} should align with sleep dates "
        f"{list(sleep.index)} — got {len(common)} aligned of {len(journal)}"
    )


def test_load_journal_daily_df_pivot(patched_whoop_dir: Path) -> None:
    df = load_journal_daily_df()

    # Two days of pivoted data
    assert len(df) == 2
    assert df.index.name == "timestamp"
    assert isinstance(df.index, pd.DatetimeIndex)
    assert str(df.index.tz) == "UTC"

    # Normalized column keys present
    assert "caffeine" in df.columns
    assert "stressed" in df.columns
    assert "engaged_in_sexual_activity" in df.columns

    # Day 1: caffeine=True, stressed=False, sex=True
    assert df.iloc[0]["caffeine"] is True or df.iloc[0]["caffeine"] == True  # noqa: E712
    assert df.iloc[0]["stressed"] is False or df.iloc[0]["stressed"] == False  # noqa: E712
    # Day 2: caffeine=False, stressed=True
    assert df.iloc[1]["caffeine"] is False or df.iloc[1]["caffeine"] == False  # noqa: E712
    assert df.iloc[1]["stressed"] is True or df.iloc[1]["stressed"] == True  # noqa: E712

    # Privacy default: notes excluded
    assert "notes" not in df.columns


def test_load_journal_daily_df_include_notes(patched_whoop_dir: Path) -> None:
    df = load_journal_daily_df(include_notes=True)

    assert "notes" in df.columns
    # Day 1 had one non-empty note
    day1_notes = df.iloc[0]["notes"]
    assert isinstance(day1_notes, str)
    assert "private note" in day1_notes


# ── Public API dispatch ───────────────────────────────────────────────────────


def test_load_sleep_df_dispatches_to_standard(patched_whoop_dir: Path) -> None:
    df = load_sleep_df()
    # Standard format returns 'score' + 'duration' columns (not start/end)
    assert "score" in df.columns
    assert "duration" in df.columns
    assert "start" not in df.columns


def test_load_cycles_df_dispatches(patched_whoop_dir: Path) -> None:
    df = load_cycles_df()
    assert "recovery" in df.columns
    assert "hrv" in df.columns


def test_load_workouts_df_dispatches(patched_whoop_dir: Path) -> None:
    df = load_workouts_df()
    assert "activity" in df.columns


def test_load_heartrate_raises_on_standard_format(patched_whoop_dir: Path) -> None:
    with pytest.raises(NotImplementedError, match="GDPR full export"):
        load_heartrate_df()


def test_load_cycles_raises_on_gdpr_format(
    monkeypatch: pytest.MonkeyPatch, gdpr_export: Path
) -> None:
    monkeypatch.setattr(whoop, "_whoop_dir", lambda: gdpr_export)
    with pytest.raises(NotImplementedError, match="standard Whoop export"):
        load_cycles_df()


# ── File-missing error paths ──────────────────────────────────────────────────


def test_load_sleep_standard_missing_file(tmp_path: Path) -> None:
    (tmp_path / "physiological_cycles.csv").write_text(
        ""
    )  # makes detect see "standard"
    with pytest.raises(FileNotFoundError, match="sleeps.csv"):
        _load_sleep_standard(tmp_path)


def test_load_cycles_standard_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="physiological_cycles.csv"):
        _load_cycles_standard(tmp_path)


def test_load_workouts_standard_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="workouts.csv"):
        _load_workouts_standard(tmp_path)


def test_load_journal_standard_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="journal_entries.csv"):
        _load_journal_standard(tmp_path)


# ── Question text normalization ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "question,expected",
    [
        ("Have any caffeine?", "caffeine"),
        ("Had any alcohol?", "alcohol"),
        ("Felt stressed today?", "stressed"),
        ("Feel energized today?", "energized"),
        ("Engaged in sexual activity?", "engaged_in_sexual_activity"),
        ("Did you read?", "read"),
        ("Are you sick?", "sick"),
        ("Any caffeine?", "caffeine"),
        ("Slept in own bed yesterday?", "slept_in_own_bed"),
        ("", "unknown"),
    ],
)
def test_question_to_key(question: str, expected: str) -> None:
    assert _question_to_key(question) == expected


def test_question_to_key_non_string() -> None:
    assert _question_to_key(None) == "unknown"  # type: ignore[arg-type]
    assert _question_to_key(float("nan")) == "unknown"  # type: ignore[arg-type]


# ── Live config integration (skipped without real data) ───────────────────────


@pytest.mark.skipif(not has_whoop_config, reason="no whoop config")
def test_load_whoop_sleep() -> None:
    df = load_sleep_df()
    assert len(df) > 0


@pytest.mark.skipif(not has_whoop_config, reason="no whoop config")
def test_load_whoop_heartrate_or_cycles() -> None:
    # Whichever format the live config points at, one of these should succeed
    try:
        load_heartrate_df()
    except NotImplementedError:
        # Standard format — try cycles instead
        df = load_cycles_df()
        assert len(df) > 0
