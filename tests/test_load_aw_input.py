"""Tests for the aw-watcher-input loader (quantifiedme.load.aw_input).

Uses synthetic aw_core Events, so it needs no running ActivityWatch server and
runs anywhere. Pins the daily aggregation contract that
``derived.all_df.load_all_df`` relies on (the ``input:`` prefixed columns).
"""

from datetime import datetime, timedelta, timezone

from aw_core import Event

from quantifiedme.load import aw_input


def _ev(ts: datetime, **counters) -> Event:
    return Event(timestamp=ts, duration=timedelta(seconds=10), data=counters)


def test_aggregate_daily_sums_per_day() -> None:
    d0 = datetime(2026, 1, 1, 10, tzinfo=timezone.utc)
    d1 = datetime(2026, 1, 2, 11, tzinfo=timezone.utc)
    events = [
        _ev(d0, presses=100, clicks=5, deltaX=50, deltaY=-30, scrollX=0, scrollY=-8),
        _ev(d0 + timedelta(minutes=5), presses=50, clicks=2, deltaX=-10, deltaY=20, scrollX=3, scrollY=4),
        _ev(d1, presses=10, clicks=1, deltaX=5, deltaY=5, scrollX=0, scrollY=0),
    ]
    df = aw_input.aggregate_daily(events)

    assert len(df) == 2
    row0 = df.loc["2026-01-01"]
    assert row0["presses"] == 150
    assert row0["clicks"] == 7
    assert row0["deltaX"] == 40  # 50 + (-10)
    assert row0["deltaY"] == -10  # -30 + 20
    # sign-agnostic magnitudes: per-event abs summed, not sum-then-abs
    assert row0["mouse_move"] == 110  # (|50|+|-30|) + (|-10|+|20|) = 80+30
    assert row0["scroll"] == 15  # (|0|+|-8|) + (|3|+|4|) = 8+7


def test_aggregate_daily_empty_has_expected_columns() -> None:
    df = aw_input.aggregate_daily([])
    assert df.empty
    for col in [*aw_input.COUNTER_KEYS, "mouse_move", "scroll"]:
        assert col in df.columns


def test_aggregate_daily_missing_counter_defaults_zero() -> None:
    # Event carrying only key presses (no mouse counters) must not crash.
    d0 = datetime(2026, 3, 1, 9, tzinfo=timezone.utc)
    df = aw_input.aggregate_daily([_ev(d0, presses=42)])
    assert df.loc["2026-03-01"]["presses"] == 42
    assert df.loc["2026-03-01"]["clicks"] == 0
    assert df.loc["2026-03-01"]["mouse_move"] == 0
