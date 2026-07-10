"""Tests for the WHOOP API v2 loader.

Record fixtures match the v2 response schemas from the official docs
(https://developer.whoop.com/docs/developing/user-data/). No network calls —
HTTP is mocked at the requests/module boundary.
"""

import json
import time
from pathlib import Path

import pandas as pd
import pytest

from quantifiedme.load import whoop, whoop_api
from quantifiedme.load.whoop_api import (
    _cycles_to_df,
    _local_date,
    _paginate,
    _sleeps_to_df,
    _workouts_to_df,
    fetch_collection,
)

# ── Fixture records (v2 schemas) ──────────────────────────────────────────────

SLEEP_SCORED = {
    "id": "ecfc6a15-4661-442f-a9a4-f160dd7afae8",
    "cycle_id": 93845,
    "user_id": 10129,
    "start": "2026-05-09T21:15:00.000Z",
    "end": "2026-05-10T05:00:00.000Z",  # 07:00 local at +02:00
    "timezone_offset": "+02:00",
    "nap": False,
    "score_state": "SCORED",
    "score": {
        "stage_summary": {
            "total_in_bed_time_milli": 28_500_000,  # 475 min
            "total_awake_time_milli": 1_500_000,  # 25 min
            "total_no_data_time_milli": 0,
            "total_light_sleep_time_milli": 12_600_000,  # 210 min
            "total_slow_wave_sleep_time_milli": 6_600_000,  # 110 min
            "total_rem_sleep_time_milli": 7_800_000,  # 130 min
            "sleep_cycle_count": 4,
            "disturbance_count": 8,
        },
        "sleep_needed": {
            "baseline_milli": 28_800_000,
            "need_from_sleep_debt_milli": 1_800_000,  # 30 min
            "need_from_recent_strain_milli": 0,
            "need_from_recent_nap_milli": 0,
        },
        "respiratory_rate": 15.2,
        "sleep_performance_percentage": 92,
        "sleep_consistency_percentage": 85,
        "sleep_efficiency_percentage": 94,
    },
}

SLEEP_NAP = {
    **SLEEP_SCORED,
    "id": "nap-id",
    "nap": True,
    "start": "2026-05-10T11:00:00.000Z",
    "end": "2026-05-10T11:45:00.000Z",
}

SLEEP_UNSCORED = {
    "id": "pending-id",
    "cycle_id": 93846,
    "start": "2026-05-10T21:30:00.000Z",
    "end": "2026-05-11T04:45:00.000Z",
    "timezone_offset": "+02:00",
    "nap": False,
    "score_state": "PENDING_SCORE",
}

CYCLE = {
    "id": 93845,
    "user_id": 10129,
    "start": "2026-05-09T21:00:00.000Z",
    "end": "2026-05-10T21:00:00.000Z",
    "timezone_offset": "+02:00",
    "score_state": "SCORED",
    "score": {
        "strain": 12.5,
        "kilojoule": 10878.4,  # = 2600 kcal
        "average_heart_rate": 75,
        "max_heart_rate": 165,
    },
}

RECOVERY = {
    "cycle_id": 93845,
    "sleep_id": "ecfc6a15-4661-442f-a9a4-f160dd7afae8",
    "user_id": 10129,
    "created_at": "2026-05-10T05:05:00.000Z",
    "score_state": "SCORED",
    "score": {
        "user_calibrating": False,
        "recovery_score": 72,
        "resting_heart_rate": 52,
        "hrv_rmssd_milli": 58.0,
        "spo2_percentage": 96.0,
        "skin_temp_celsius": 33.1,
    },
}

WORKOUT = {
    "id": "workout-uuid",
    "user_id": 10129,
    "start": "2026-05-10T16:00:00.000Z",
    "end": "2026-05-10T17:00:00.000Z",
    "timezone_offset": "+02:00",
    "sport_name": "running",
    "score_state": "SCORED",
    "score": {
        "strain": 9.5,
        "average_heart_rate": 142,
        "max_heart_rate": 175,
        "kilojoule": 2175.68,  # = 520 kcal
        "percent_recorded": 100,
        "distance_meter": 10000.0,
        "zone_durations": {},
    },
}


# ── Date localization ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "iso,offset,expected",
    [
        ("2026-05-10T05:00:00.000Z", "+02:00", "2026-05-10"),
        ("2026-05-10T23:30:00.000Z", "+02:00", "2026-05-11"),  # crosses midnight
        ("2026-05-10T03:00:00.000Z", "-05:00", "2026-05-09"),  # negative offset
        ("2026-05-10T05:00:00.000Z", "Z", "2026-05-10"),
        ("2026-05-10T19:00:00.000Z", "+05:30", "2026-05-11"),  # half-hour offset
    ],
)
def test_local_date(iso: str, offset: str, expected: str) -> None:
    assert _local_date(iso, offset) == pd.Timestamp(expected, tz="UTC")


# ── Record → DataFrame mapping ────────────────────────────────────────────────


def test_sleeps_to_df() -> None:
    df = _sleeps_to_df([SLEEP_SCORED, SLEEP_NAP, SLEEP_UNSCORED])

    # Nap and unscored rows excluded
    assert len(df) == 1
    assert df.index.name == "timestamp"
    assert str(df.index.tz) == "UTC"
    # Wake date: 05:00 UTC + 02:00 = 07:00 local on 2026-05-10
    assert df.index[0] == pd.Timestamp("2026-05-10", tz="UTC")

    row = df.iloc[0]
    assert row["score"] == 92
    # asleep = light + deep + rem = 210 + 110 + 130 = 450 min
    assert row["duration"] == pd.Timedelta(minutes=450)
    assert row["time_in_bed"] == pd.Timedelta(minutes=475)
    assert row["efficiency"] == 94
    assert row["consistency"] == 85
    assert row["debt"] == 30
    assert row["respiratory_rate"] == 15.2
    assert row["rem"] == pd.Timedelta(minutes=130)
    assert row["deep"] == pd.Timedelta(minutes=110)
    assert row["light"] == pd.Timedelta(minutes=210)
    assert row["awake"] == pd.Timedelta(minutes=25)

    # Same columns as the CSV loader — derived modules must not care about source
    assert set(df.columns) == {
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
    }


def test_cycles_to_df() -> None:
    df = _cycles_to_df([CYCLE], [RECOVERY], [SLEEP_SCORED])

    assert len(df) == 1
    # Date from the linked sleep's wake time, not the cycle bounds
    assert df.index[0] == pd.Timestamp("2026-05-10", tz="UTC")

    row = df.iloc[0]
    assert row["recovery"] == 72
    assert row["resting_hr"] == 52
    assert row["hrv"] == 58.0
    assert row["skin_temp"] == 33.1
    assert row["spo2"] == 96.0
    assert row["strain"] == 12.5
    assert row["energy_kcal"] == pytest.approx(2600, abs=1)

    assert set(df.columns) == {
        "recovery",
        "resting_hr",
        "hrv",
        "skin_temp",
        "spo2",
        "strain",
        "energy_kcal",
    }


def test_cycles_to_df_falls_back_to_created_at() -> None:
    """Without the linked sleep record, wake date comes from recovery created_at."""
    df = _cycles_to_df([CYCLE], [RECOVERY], [])
    assert df.index[0] == pd.Timestamp("2026-05-10", tz="UTC")


def test_cycles_to_df_skips_unscored() -> None:
    unscored = {**RECOVERY, "score_state": "PENDING_SCORE"}
    df = _cycles_to_df([CYCLE], [unscored], [SLEEP_SCORED])
    assert df.empty


def test_workouts_to_df() -> None:
    df = _workouts_to_df([WORKOUT])

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
    row = df.iloc[0]
    assert row["start"] == pd.Timestamp("2026-05-10 16:00:00", tz="UTC")
    assert row["duration"] == pd.Timedelta(minutes=60)
    assert row["activity"] == "running"
    assert row["strain"] == 9.5
    assert row["energy_kcal"] == pytest.approx(520, abs=1)


def test_workouts_to_df_empty() -> None:
    df = _workouts_to_df([])
    assert df.empty
    assert "activity" in df.columns


# ── Token persistence & refresh rotation ──────────────────────────────────────


@pytest.fixture
def token_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    path = tmp_path / "whoop_token.json"
    monkeypatch.setattr(whoop_api, "_token_path", lambda: path)
    return path


def test_save_token_sets_expiry_and_permissions(token_file: Path) -> None:
    whoop_api._save_token(
        {"access_token": "a", "refresh_token": "r", "expires_in": 3600}
    )

    saved = json.loads(token_file.read_text())
    assert saved["expires_at"] == pytest.approx(time.time() + 3600, abs=5)
    assert oct(token_file.stat().st_mode)[-3:] == "600"


def test_has_auth(token_file: Path) -> None:
    assert not whoop_api.has_auth()
    whoop_api._save_token(
        {"access_token": "a", "refresh_token": "r", "expires_in": 3600}
    )
    assert whoop_api.has_auth()


def test_access_token_refreshes_and_persists_rotation(
    token_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """WHOOP rotates refresh tokens — the rotated token must hit disk
    immediately, or the next refresh uses a dead token and auth is bricked."""
    whoop_api._save_token(
        {"access_token": "old", "refresh_token": "old-refresh", "expires_at": 0}
    )
    monkeypatch.setattr(
        whoop_api, "_load_credentials", lambda: ("client-id", "client-secret")
    )

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "access_token": "new",
                "refresh_token": "new-refresh",
                "expires_in": 3600,
            }

    posted = {}

    def fake_post(url, data=None, timeout=None):
        posted.update(data)
        return FakeResponse()

    monkeypatch.setattr(whoop_api.requests, "post", fake_post)

    assert whoop_api._access_token() == "new"
    assert posted["grant_type"] == "refresh_token"
    assert posted["refresh_token"] == "old-refresh"
    # Rotated refresh token persisted
    assert json.loads(token_file.read_text())["refresh_token"] == "new-refresh"


def test_access_token_valid_no_refresh(token_file: Path) -> None:
    whoop_api._save_token(
        {"access_token": "ok", "refresh_token": "r", "expires_at": time.time() + 3600}
    )
    assert whoop_api._access_token() == "ok"


# ── Pagination & caching ──────────────────────────────────────────────────────


def test_paginate_follows_next_token(monkeypatch: pytest.MonkeyPatch) -> None:
    pages = [
        {"records": [{"id": 1}, {"id": 2}], "next_token": "tok-2"},
        {"records": [{"id": 3}], "next_token": None},
    ]
    calls = []

    def fake_get(path, params=None):
        calls.append(dict(params))
        return pages[len(calls) - 1]

    monkeypatch.setattr(whoop_api, "_api_get", fake_get)

    records = _paginate("/cycle")
    assert [r["id"] for r in records] == [1, 2, 3]
    # Second request must pass the camelCase query param
    assert calls[1]["nextToken"] == "tok-2"


def test_fetch_collection_incremental_upsert(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        whoop_api, "_cache_path", lambda name: tmp_path / f"{name}.json"
    )

    first = [
        {"id": "a", "start": "2026-05-01T00:00:00.000Z", "score": 1},
        {"id": "b", "start": "2026-05-10T00:00:00.000Z", "score": 1},
    ]
    second = [
        {"id": "b", "start": "2026-05-10T00:00:00.000Z", "score": 2},  # revised
        {"id": "c", "start": "2026-05-20T00:00:00.000Z", "score": 1},  # new
    ]
    starts = []

    def fake_paginate(path, start=None):
        starts.append(start)
        return first if len(starts) == 1 else second

    monkeypatch.setattr(whoop_api, "_paginate", fake_paginate)

    records = fetch_collection("sleeps")
    assert len(records) == 2
    assert starts[0] is None  # cold cache → full fetch

    records = fetch_collection("sleeps")
    assert starts[1] is not None  # warm cache → incremental with overlap
    assert starts[1] < pd.Timestamp("2026-05-10T00:00:00.000Z")
    assert [r["id"] for r in records] == ["a", "b", "c"]
    # Revised record replaced, not duplicated
    assert next(r for r in records if r["id"] == "b")["score"] == 2


# ── Dispatch from whoop.py ────────────────────────────────────────────────────


def test_whoop_dispatches_to_api_when_authorized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = pd.DataFrame({"recovery": [50]})
    monkeypatch.setattr(whoop_api, "has_auth", lambda: True)
    monkeypatch.setattr(whoop_api, "load_cycles_df", lambda: sentinel)
    # The autouse no_api fixture in test_load_whoop.py doesn't apply here;
    # restore the real _use_api in case of cross-module patching.
    assert whoop.load_cycles_df() is sentinel
