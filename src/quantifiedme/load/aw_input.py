"""
Loader for aw-watcher-input buckets (type ``os.hid.input``).

aw-watcher-input records low-level keyboard/mouse activity in short
(~5-15s) events whose ``data`` holds cumulative counters for the interval:
``presses`` (key presses), ``clicks`` (mouse button clicks),
``deltaX``/``deltaY`` (mouse movement, pixels) and
``scrollX``/``scrollY`` (scroll deltas).

Unlike the window/afk buckets consumed via ``screentime``, this is a raw
input-intensity signal: useful as a productivity/engagement proxy that is
independent of which app was focused, and as an input-based "active time"
measure that complements AFK.

The daily dataframe (``load_daily_df``) sums each counter per day across all
configured hostnames, so it slots into ``derived.all_df.load_all_df`` with an
``input:`` prefix.
"""

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
from aw_client import ActivityWatchClient
from aw_core import Event

from ..cache import memory
from ..config import load_config

logger = logging.getLogger(__name__)

# Counter keys emitted by aw-watcher-input events (os.hid.input).
COUNTER_KEYS = ["presses", "clicks", "deltaX", "deltaY", "scrollX", "scrollY"]


def _get_aw_client(testing: bool = False) -> ActivityWatchClient:
    config = load_config(use_example=testing)
    sec_aw = config["data"].get("activitywatch", {})
    port = sec_aw.get("port", 5600 if not testing else 5666)
    return ActivityWatchClient(port=port, testing=testing)


def _hostnames(config) -> list[str]:
    sec_aw = config["data"].get("activitywatch", {})
    return sec_aw.get("hostnames", [])


@memory.cache(ignore=["awc"])
def load_events(
    awc: ActivityWatchClient,
    hostname: str,
    since: datetime,
    end: datetime,
) -> list[Event]:
    """Load raw aw-watcher-input events for a single hostname."""
    bucket_id = f"aw-watcher-input_{hostname}"
    events = awc.get_events(bucket_id, start=since, end=end)
    # Keep only events that actually carry counter data.
    events = [e for e in events if e.data]
    return events


def load_input_events(
    since: datetime | None = None,
    end: datetime | None = None,
    hostnames: list[str] | None = None,
    awc: ActivityWatchClient | None = None,
    testing: bool = False,
) -> list[Event]:
    """Load raw aw-watcher-input events across all configured hostnames."""
    now = datetime.now(tz=timezone.utc)
    if since is None:
        since = now - timedelta(days=365)
    if end is None:
        end = now
    assert since.tzinfo and end.tzinfo, "since/end must be timezone-aware"

    config = load_config(use_example=testing)
    if hostnames is None:
        hostnames = _hostnames(config)

    if awc is None:
        awc = _get_aw_client(testing=testing)

    all_events: list[Event] = []
    with awc:
        for hostname in hostnames:
            try:
                all_events += load_events(awc, hostname, since, end)
            except Exception as e:
                # A missing input bucket on a host is expected (not every
                # machine runs aw-watcher-input); warn and continue.
                logger.warning(f"Skipping aw-watcher-input for {hostname}: {e}")
    return all_events


def aggregate_daily(events: list[Event]) -> pd.DataFrame:
    """Sum input counters per day (local date of the event timestamp).

    Columns: ``presses``, ``clicks``, ``deltaX``, ``deltaY``, ``scrollX``,
    ``scrollY``, plus derived ``mouse_move`` (|deltaX|+|deltaY|) and ``scroll``
    (|scrollX|+|scrollY|). Pure over ``events`` so it is trivially testable.
    """
    if not events:
        return pd.DataFrame(
            columns=[*COUNTER_KEYS, "mouse_move", "scroll"],
            index=pd.DatetimeIndex([], name="date"),
        )

    rows = [{k: float(e.data.get(k, 0) or 0) for k in COUNTER_KEYS} for e in events]
    df = pd.DataFrame(rows)
    df["date"] = pd.DatetimeIndex([e.timestamp for e in events]).date

    # Per-event abs before groupby — opposing movements must not cancel before summing.
    df["mouse_move"] = df["deltaX"].abs() + df["deltaY"].abs()
    df["scroll"] = df["scrollX"].abs() + df["scrollY"].abs()

    daily = df.groupby("date").sum()
    daily.index = pd.DatetimeIndex(daily.index, name="date")
    return daily


def load_daily_df(
    since: datetime | None = None,
    end: datetime | None = None,
    hostnames: list[str] | None = None,
    awc: ActivityWatchClient | None = None,
    testing: bool = False,
) -> pd.DataFrame:
    """Return a daily dataframe of summed input counters, across all hostnames."""
    events = load_input_events(
        since=since, end=end, hostnames=hostnames, awc=awc, testing=testing
    )
    return aggregate_daily(events)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = load_daily_df()
    pd.set_option("display.max_columns", None)
    print(df.describe())
    print(f"\nTotal days: {len(df)}")
    if len(df):
        print(f"Range: {df.index.min().date()} / {df.index.max().date()}")
