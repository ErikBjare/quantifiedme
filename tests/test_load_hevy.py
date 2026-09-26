"""Tests for the Hevy CSV export loader (synthetic fixture, no personal data)."""

from pathlib import Path

import pandas as pd
import pytest

from quantifiedme.load.hevy import load_daily_df, load_sets_df, sets_to_daily_df

HEADER = (
    '"title","start_time","end_time","description","exercise_title","superset_id",'
    '"exercise_notes","set_index","set_type","weight_kg","reps","distance_km",'
    '"duration_seconds","rpe"\n'
)

ROWS = [
    # Day 1: one session, 1 warm-up + 2 working sets of squats + 1 bodyweight set
    '"Legs","3 Mar 2026, 18:00","3 Mar 2026, 19:15","","Squat (Barbell)",,"",0,"warmup",40,10,,,',
    '"Legs","3 Mar 2026, 18:00","3 Mar 2026, 19:15","","Squat (Barbell)",,"",1,"normal",100,5,,,8',
    '"Legs","3 Mar 2026, 18:00","3 Mar 2026, 19:15","","Squat (Barbell)",,"",2,"failure",100,4,,,10',
    '"Legs","3 Mar 2026, 18:00","3 Mar 2026, 19:15","","Pull Up",,"",0,"normal",,8,,,',
    # Day 2: two sessions
    '"Push","5 Mar 2026, 07:00","5 Mar 2026, 07:45","","Bench Press (Barbell)",,"",0,"normal",60,10,,,',
    '"Cardio","5 Mar 2026, 17:00","5 Mar 2026, 17:30","","Rowing Machine",,"",0,"normal",,,5.0,1800,',
]


@pytest.fixture
def hevy_csv(tmp_path: Path) -> Path:
    path = tmp_path / "workout_data.csv"
    path.write_text(HEADER + "\n".join(ROWS) + "\n")
    return path


def test_load_sets_df(hevy_csv: Path) -> None:
    df = load_sets_df(hevy_csv)
    assert len(df) == len(ROWS)
    assert df["start"].iloc[0] == pd.Timestamp("2026-03-03 18:00")
    assert df["end"].iloc[0] == pd.Timestamp("2026-03-03 19:15")
    assert df["workout_id"].nunique() == 3


def test_daily_df(hevy_csv: Path) -> None:
    daily = load_daily_df(hevy_csv)

    assert list(daily.index) == [pd.Timestamp("2026-03-03"), pd.Timestamp("2026-03-05")]
    assert daily.index.name == "date"
    assert list(daily.columns) == ["sessions", "sets", "reps", "volume_kg", "minutes"]

    d1 = daily.loc["2026-03-03"]
    assert d1["sessions"] == 1
    assert d1["sets"] == 3  # warm-up excluded
    assert d1["reps"] == 5 + 4 + 8
    assert d1["volume_kg"] == 100 * 5 + 100 * 4  # bodyweight set adds no volume
    assert d1["minutes"] == 75

    d2 = daily.loc["2026-03-05"]
    assert d2["sessions"] == 2
    assert d2["sets"] == 2
    assert d2["volume_kg"] == 600
    assert d2["minutes"] == 45 + 30


def test_iso_timestamps_fallback(tmp_path: Path) -> None:
    """Tolerate exports with ISO timestamps instead of Hevy's default format."""
    path = tmp_path / "workout_data.csv"
    path.write_text(
        HEADER
        + '"Legs","2026-03-03 18:00:00","2026-03-03 19:00:00","","Squat (Barbell)",,"",0,"normal",100,5,,,\n'
    )
    daily = load_daily_df(path)
    assert daily.loc["2026-03-03", "minutes"] == 60


def test_empty_export(tmp_path: Path) -> None:
    path = tmp_path / "workout_data.csv"
    path.write_text(HEADER)
    daily = sets_to_daily_df(load_sets_df(path))
    assert daily.empty
    assert "volume_kg" in daily.columns


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_sets_df(tmp_path / "nope.csv")
