"""Tests for the OpenElfbar database loader (synthetic fixture, no personal data)."""

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from quantifiedme.load.openelfbar import (
    hourly_to_daily_df,
    load_daily_df,
    load_hourly_df,
)

DEVICE = "aabbccddeeff"


def _write_db(
    path: Path,
    syncs: dict[str, dict[tuple[str, int], tuple[float, float]]],
    device: str = DEVICE,
):
    """syncs: started_at (UTC ISO) -> {(day_label, hour): (puffs, duration)}"""
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS sync_runs (sync_run_id TEXT PRIMARY KEY, started_at TEXT NOT NULL,
                                status TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS hourly_aggregates (device_identifier_hex TEXT, sync_run_id TEXT,
                                        metric TEXT, day_label TEXT, hour_index INTEGER,
                                        value REAL);
        """
    )
    for i, (started_at, values) in enumerate(syncs.items()):
        run = f"{device}-s{i}"
        conn.execute(
            "INSERT INTO sync_runs VALUES (?, ?, 'completed')", (run, started_at)
        )
        # the device always reports all 72 hours, zero-filled
        for label in ["two_days_ago", "yesterday", "today"]:
            for hour in range(24):
                puffs, duration = values.get((label, hour), (0, 0))
                for metric, value in [
                    ("puff_count", puffs),
                    ("duration_summary", duration),
                ]:
                    conn.execute(
                        "INSERT INTO hourly_aggregates VALUES (?, ?, ?, ?, ?, ?)",
                        (device, run, metric, label, hour, value),
                    )
    conn.commit()
    conn.close()


@pytest.fixture
def db(tmp_path: Path) -> Path:
    path = tmp_path / "openelfbar.db"
    _write_db(
        path,
        {
            # 2026-03-03 12:30 local (CET, UTC+1)
            "2026-03-03T11:30:00.000Z": {
                ("two_days_ago", 20): (5, 8),
                ("yesterday", 10): (3, 4),
                ("today", 9): (2, 3),
            },
            # next day, overlapping window; hour 9 on 03-03 got more puffs later
            "2026-03-04T20:10:00.000Z": {
                ("two_days_ago", 10): (3, 4),
                ("yesterday", 9): (4, 6),
                ("today", 20): (7, 9),
            },
        },
    )
    return path


def test_load_hourly_df_anchors_and_dedupes(db: Path) -> None:
    hourly = load_hourly_df(db)
    puffs = hourly["puffs"]
    assert puffs[pd.Timestamp("2026-03-01 20:00")] == 5
    assert puffs[pd.Timestamp("2026-03-02 10:00")] == 3
    # latest sync wins for an hour both syncs covered
    assert puffs[pd.Timestamp("2026-03-03 09:00")] == 4
    assert puffs[pd.Timestamp("2026-03-04 20:00")] == 7
    # the sync's own hour (21:00) was still in progress
    assert hourly.index.max() == pd.Timestamp("2026-03-04 20:00")


def test_daily_only_counts_fully_covered_days(db: Path) -> None:
    daily = hourly_to_daily_df(load_hourly_df(db))
    assert daily.loc["2026-03-02", "puffs"] == 3
    assert daily.loc["2026-03-03", "puffs"] == 4
    assert daily.loc["2026-03-03", "duration"] == 6
    assert daily.loc["2026-03-03", "covered_hours"] == 24
    # 2026-03-04 only covered up to 21:00, so the total is unknown
    assert daily.loc["2026-03-04", "covered_hours"] == 21
    assert pd.isna(daily.loc["2026-03-04", "puffs"])


def test_daily_date_offset(db: Path) -> None:
    daily = load_daily_df(db, date_offset_hours=5)
    # 03-01 00:00-04:00 belong to the (uncovered) day before
    assert daily.loc["2026-02-28", "covered_hours"] == 5
    assert pd.isna(daily.loc["2026-02-28", "puffs"])
    # 03-01 05:00 → 03-02 05:00 is fully covered
    assert daily.loc["2026-03-01", "puffs"] == 5
    assert daily.loc["2026-03-03", "puffs"] == 4


def test_missing_db(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_daily_df(tmp_path / "nope.db", date_offset_hours=0)


def test_sync_in_last_hour_leaves_day_incomplete(tmp_path: Path) -> None:
    path = tmp_path / "openelfbar.db"
    # 23:10 local: hour 23 is still in progress
    _write_db(path, {"2026-03-03T22:10:00.000Z": {("today", 23): (2, 3)}})
    daily = load_daily_df(path, date_offset_hours=0)
    assert daily.loc["2026-03-03", "covered_hours"] == 23
    assert pd.isna(daily.loc["2026-03-03", "puffs"])


def test_partial_device_makes_day_incomplete(tmp_path: Path) -> None:
    path = tmp_path / "openelfbar.db"
    _write_db(path, {"2026-03-04T11:30:00.000Z": {("yesterday", 9): (4, 6)}})
    # a second device only synced at noon on 03-03
    _write_db(
        path, {"2026-03-03T11:30:00.000Z": {("today", 9): (1, 1)}}, device="other"
    )
    daily = load_daily_df(path, date_offset_hours=0)
    assert daily.loc["2026-03-03", "covered_hours"] == 12
    assert pd.isna(daily.loc["2026-03-03", "puffs"])
    # 03-02 is fully covered by both devices, so it sums
    assert daily.loc["2026-03-02", "covered_hours"] == 24


def test_latest_sync_wins_across_dst_fallback(tmp_path: Path) -> None:
    path = tmp_path / "openelfbar.db"
    _write_db(
        path,
        {
            # 02:50 CEST, before the 2026-10-25 fall-back
            "2026-10-25T00:50:00.000Z": {("yesterday", 9): (1, 1)},
            # 02:10 CET, after it: later in UTC, earlier on the wall clock
            "2026-10-25T01:10:00.000Z": {("yesterday", 9): (5, 5)},
        },
    )
    hourly = load_hourly_df(path)
    assert hourly["puffs"][pd.Timestamp("2026-10-24 09:00")] == 5


def test_phone_timezone(db: Path) -> None:
    # 2026-03-04 20:10 UTC is already 03-05 in Tokyo, shifting every day label
    daily = load_daily_df(db, date_offset_hours=0, tz="Asia/Tokyo")
    assert daily.loc["2026-03-04", "puffs"] == 4
