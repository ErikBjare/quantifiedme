"""
Loads nicotine-vape usage from an OpenElfbar app database.

OpenElfbar is an Android/iOS app that syncs puff data from ELFBAR vapes over
BLE and stores it in SQLite (``files/SQLite/openelfbar.db`` in the app
sandbox). Copy it off the phone with ``scripts/pull-openelfbar.sh``.

Devices like the ELFX Ultra only expose **hourly aggregates**: on every sync
the device reports a rolling 3-day window (``today``/``yesterday``/
``two_days_ago`` × 24 hours) of puff counts and a duration summary. Day labels
are relative to the device clock, which the app sets to phone local time on
each sync, so rows are anchored to the sync's local date.

Hours that no sync covered are unknown, not zero: if you go more than ~3 days
without syncing, the device drops the oldest hours. Daily totals are therefore
NaN unless every hour of the day was covered.

Per-puff rows (``puffs`` table, Elfa Master V2 and similar) are not handled
yet.

Configure the path in ``config.toml``::

    [data]
    openelfbar = "~/annex/Logs/OpenElfbar/openelfbar-latest.db"
"""

import sqlite3
from datetime import timedelta
from pathlib import Path

import pandas as pd

from ..config import load_config

DAY_OFFSETS = {"today": 0, "yesterday": 1, "two_days_ago": 2}
METRICS = {"puff_count": "puffs", "duration_summary": "duration"}
DAILY_COLUMNS = ["puffs", "duration", "covered_hours"]
DEFAULT_TIMEZONE = "Europe/Stockholm"


def _resolve_path(path: Path | str | None) -> Path:
    if path is None:
        path = load_config()["data"]["openelfbar"]
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"OpenElfbar database not found at {path}")
    return path


def _load_raw_aggregates(path: Path) -> pd.DataFrame:
    # read-only so loading never touches the backup copy
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        return pd.read_sql_query(
            """
            SELECT h.device_identifier_hex AS device, h.metric, h.day_label,
                   h.hour_index, h.value, s.started_at
            FROM hourly_aggregates h
            JOIN sync_runs s USING (sync_run_id)
            WHERE s.status = 'completed'
            """,
            conn,
        )


def raw_to_hourly_df(raw: pd.DataFrame, tz: str = DEFAULT_TIMEZONE) -> pd.DataFrame:
    """Anchor raw aggregate rows to absolute local hours and dedupe across syncs.

    Returns one row per (device, local hour) that some sync covered, indexed
    by naive local ``timestamp`` (hour start), with ``puffs`` and ``duration``
    columns. When several syncs cover an hour, the latest sync wins. Hours at
    or after a sync's own time are dropped, since they hadn't happened yet.
    """
    if raw.empty:
        out = pd.DataFrame(columns=["device", "puffs", "duration"])
        out.index = pd.DatetimeIndex([], name="timestamp")
        return out

    raw = raw[raw["day_label"].isin(DAY_OFFSETS) & raw["metric"].isin(METRICS)]
    raw = raw[raw["hour_index"].between(0, 23)].copy()
    synced = pd.to_datetime(raw["started_at"], utc=True, format="ISO8601")
    synced_local = synced.dt.tz_convert(tz).dt.tz_localize(None)
    day = synced_local.dt.normalize() - pd.to_timedelta(
        raw["day_label"].map(DAY_OFFSETS), unit="D"
    )
    raw["timestamp"] = day + pd.to_timedelta(raw["hour_index"], unit="h")
    raw["synced"] = synced_local
    raw = raw[raw["timestamp"] <= raw["synced"].dt.floor("h")]

    # latest sync wins per (device, hour, metric)
    raw = raw.sort_values("synced").drop_duplicates(
        ["device", "timestamp", "metric"], keep="last"
    )
    hourly = raw.pivot_table(
        index=["device", "timestamp"], columns="metric", values="value", aggfunc="last"
    ).rename(columns=METRICS)
    hourly = hourly.reindex(columns=list(METRICS.values()))
    hourly.columns.name = None
    return hourly.reset_index().set_index("timestamp").sort_index()


def hourly_to_daily_df(
    hourly: pd.DataFrame, date_offset_hours: float = 0
) -> pd.DataFrame:
    """Sum covered hours into days (shifted by ``date_offset_hours``).

    ``covered_hours`` counts hours with data for the day; ``puffs`` and
    ``duration`` are NaN unless all 24 hours were covered, so gaps between
    syncs never read as abstinence. Totals are summed across devices.
    """
    if hourly.empty:
        out = pd.DataFrame(columns=DAILY_COLUMNS, dtype=float)
        out.index = pd.DatetimeIndex([], name="date")
        return out

    # merge devices first so an hour counts once toward coverage
    per_hour = hourly.groupby(level="timestamp")[["puffs", "duration"]].sum()
    dates = (
        pd.DatetimeIndex(per_hour.index) - timedelta(hours=date_offset_hours)
    ).normalize()
    daily = per_hour.groupby(dates).sum()
    daily["covered_hours"] = per_hour.groupby(dates).size()
    daily.loc[daily["covered_hours"] < 24, ["puffs", "duration"]] = float("nan")
    daily.index = pd.DatetimeIndex(daily.index, name="date")
    return daily[DAILY_COLUMNS].sort_index()


def load_hourly_df(
    path: Path | str | None = None, tz: str = DEFAULT_TIMEZONE
) -> pd.DataFrame:
    """Load covered hours (naive local time) from an OpenElfbar database."""
    return raw_to_hourly_df(_load_raw_aggregates(_resolve_path(path)), tz=tz)


def load_daily_df(
    path: Path | str | None = None, date_offset_hours: float | None = None
) -> pd.DataFrame:
    """Load daily vape usage; days without full sync coverage are NaN."""
    if date_offset_hours is None:
        date_offset_hours = load_config().get("me", {}).get("date_offset_hours", 0)
    return hourly_to_daily_df(load_hourly_df(path), date_offset_hours)
