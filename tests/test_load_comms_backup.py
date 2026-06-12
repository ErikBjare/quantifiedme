"""Tests for the shared comms-backup reader (quantifiedme.load.comms_backup).

Uses synthetic XML fixtures written to tmp_path, so it needs no real backup and
runs anywhere (CI, alice-vm, Bob's VM). This pins the shared reader's behavior so
both Alice's comms_summary.py (aggregates) and erbot's extract_erik_sms.py
(sent-text corpus) can refactor against it.
"""

from datetime import timezone
from pathlib import Path

import pytest

from quantifiedme.load import comms_backup as cbl

# 2026-06-01T12:00:00Z in epoch millis.
JUN1_NOON_UTC_MS = "1780315200000"


def _write_sms(tmp_path: Path, name: str, rows: list[dict[str, str]]) -> Path:
    parts = [f'<?xml version="1.0"?>\n<smses count="{len(rows)}">']
    for r in rows:
        attrs = " ".join(f'{k}="{v}"' for k, v in r.items())
        parts.append(f"  <sms {attrs} />")
    parts.append("</smses>")
    path = tmp_path / name
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def _write_calls(tmp_path: Path, name: str, rows: list[dict[str, str]]) -> Path:
    parts = [f'<?xml version="1.0"?>\n<calls count="{len(rows)}">']
    for r in rows:
        attrs = " ".join(f'{k}="{v}"' for k, v in r.items())
        parts.append(f"  <call {attrs} />")
    parts.append("</calls>")
    path = tmp_path / name
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


# --- iter_records --------------------------------------------------------------


def test_iter_records_yields_attr_dicts(tmp_path: Path) -> None:
    path = _write_sms(
        tmp_path,
        "sms-1.xml",
        [
            {"type": "2", "address": "+46700000001", "date": JUN1_NOON_UTC_MS},
            {"type": "1", "address": "+46700000002", "date": JUN1_NOON_UTC_MS},
        ],
    )
    records = list(cbl.iter_records(path, "sms"))
    assert len(records) == 2
    assert records[0]["address"] == "+46700000001"
    assert all(isinstance(r, dict) for r in records)


def test_iter_records_ignores_other_tags(tmp_path: Path) -> None:
    path = _write_calls(
        tmp_path,
        "calls-1.xml",
        [{"type": "1", "number": "+4670", "date": JUN1_NOON_UTC_MS}],
    )
    # Asking for "sms" against a calls file yields nothing.
    assert list(cbl.iter_records(path, "sms")) == []
    assert len(list(cbl.iter_records(path, "call"))) == 1


# --- iter_sent_sms -------------------------------------------------------------


def test_iter_sent_sms_filters_to_type_2(tmp_path: Path) -> None:
    path = _write_sms(
        tmp_path,
        "sms-1.xml",
        [
            {"type": "2", "body": "sent one", "date": JUN1_NOON_UTC_MS},
            {"type": "1", "body": "received", "date": JUN1_NOON_UTC_MS},
            {"type": "2", "body": "sent two", "date": JUN1_NOON_UTC_MS},
            {"type": "3", "body": "draft", "date": JUN1_NOON_UTC_MS},
        ],
    )
    sent = list(cbl.iter_sent_sms(path))
    assert [r["body"] for r in sent] == ["sent one", "sent two"]


# --- normalize_number ----------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("+46 70 123 45 67", "+46701234567"),
        ("070-123 45 67", "0701234567"),
        ("(070) 123-4567", "0701234567"),
        ("+46701234567", "+46701234567"),
        (None, ""),
        ("", ""),
        ("no digits here", ""),
        # Only a single leading + is kept; embedded + from malformed input stripped.
        ("+1+800+555+1234", "+18005551234"),
    ],
)
def test_normalize_number(raw: str | None, expected: str) -> None:
    assert cbl.normalize_number(raw) == expected


def test_normalize_number_same_input_for_both_hashers() -> None:
    # The whole point of sharing normalize_number: two formats of the same number
    # collapse to one canonical string, so both consumers' hashers agree.
    assert cbl.normalize_number("+46 70 123 45 67") == cbl.normalize_number(
        "+46701234567"
    )


# --- epoch conversions ---------------------------------------------------------


def test_epoch_ms_to_day_utc() -> None:
    assert cbl.epoch_ms_to_day(JUN1_NOON_UTC_MS, tz=timezone.utc) == "2026-06-01"


def test_epoch_ms_to_iso_utc() -> None:
    iso = cbl.epoch_ms_to_iso(JUN1_NOON_UTC_MS, tz=timezone.utc)
    assert iso == "2026-06-01T12:00:00+00:00"


@pytest.mark.parametrize("bad", [None, "", "not-a-number", "12.5x"])
def test_epoch_conversions_reject_garbage(bad: str | None) -> None:
    assert cbl.epoch_ms_to_day(bad, tz=timezone.utc) is None
    assert cbl.epoch_ms_to_iso(bad, tz=timezone.utc) is None


# --- latest_file ---------------------------------------------------------------


def test_latest_file_picks_newest_by_name(tmp_path: Path) -> None:
    _write_sms(tmp_path, "sms-20260601000000.xml", [])
    newest = _write_sms(tmp_path, "sms-20260603000000.xml", [])
    _write_sms(tmp_path, "sms-20260602000000.xml", [])
    assert cbl.latest_file(tmp_path, "sms-") == newest


def test_latest_file_respects_prefix(tmp_path: Path) -> None:
    _write_calls(tmp_path, "calls-20260603000000.xml", [])
    sms = _write_sms(tmp_path, "sms-20260601000000.xml", [])
    assert cbl.latest_file(tmp_path, "sms-") == sms
    calls = cbl.latest_file(tmp_path, "calls-")
    assert calls is not None
    assert calls.name == "calls-20260603000000.xml"


def test_latest_file_none_when_absent(tmp_path: Path) -> None:
    assert cbl.latest_file(tmp_path, "sms-") is None
