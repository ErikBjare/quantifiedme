import logging
import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import aw_research.classify
import click
import numpy as np
import pandas as pd
from aw_client import ActivityWatchClient
from aw_core import Event
from aw_research.util import categorytime_per_day, split_into_weeks, verify_no_overlap
from aw_transform.union_no_overlap import union_no_overlap

from ..cache import cache_dir, memory
from ..config import _get_config_path, load_config
from ..load.activitywatch import load_events as load_events_activitywatch
from ..load.activitywatch_fake import create_fake_events
from ..load.smartertime import load_events as load_events_smartertime

logger = logging.getLogger(__name__)


def _cache_file(fast: bool) -> Path:
    return cache_dir / ("events_fast.pickle" if fast else "events.pickle")


def _get_aw_client(testing: bool) -> ActivityWatchClient:
    config = load_config(use_example=testing)
    sec_aw = config["data"].get("activitywatch", {})
    port = sec_aw.get("port", 5600 if not testing else 5666)
    return ActivityWatchClient(port=port, testing=testing)


DatasourceType = Literal["activitywatch", "smartertime_buckets", "fake", "toggl"]


def load_screentime(
    since: datetime | None = None,
    datasources: list[DatasourceType] | None = None,
    hostnames: list[str] | None = None,
    personal: bool = True,
    cache: bool = True,
    awc: ActivityWatchClient | None = None,
) -> list[Event]:
    config = load_config(use_example=not personal)

    now = datetime.now(tz=timezone.utc)
    if since is None:
        since = now - timedelta(days=365)
    else:
        assert since.tzinfo

    # The below code does caching using joblib, setting cache=False clears the cache.
    if not cache:
        memory.clear()

    # Auto-detect datasources from config if not specified
    if datasources is None:
        datasources = []
        if "activitywatch" in config["data"]:
            datasources.append("activitywatch")
        if "smartertime_buckets" in config["data"]:
            datasources.append("smartertime_buckets")

    # Check for invalid sources
    for source in datasources:
        assert source in [
            "activitywatch",
            "smartertime_buckets",
            "fake",
            "toggl",
        ], f"Invalid source: {source}"

    # Load hostnames from config if not specified
    hostnames_config = config["data"]["activitywatch"].get("hostnames", [])
    hostnames = hostnames or hostnames_config

    events: list[Event] = []

    if "activitywatch" in datasources:
        if awc is None:
            awc = _get_aw_client(not personal)
        for hostname in hostnames or []:
            logger.info(f"Getting events for {hostname}...")
            # Split up into previous days and today, to take advantage of caching
            # TODO: Split up into whole days
            # TODO: Use `aw_client.queries.canonicalQuery` instead
            events_aw: list[Event] = []
            for dtstart, dtend in split_into_weeks(since, now):
                events_aw += load_events_activitywatch(
                    awc, hostname, since=dtstart, end=dtend
                )
                logger.debug(f"{len(events_aw)} events retreived")
            for e in events_aw:
                e.data["$hostname"] = hostname
                e.data["$source"] = "activitywatch"
            events = _join_events(events, events_aw, f"activitywatch {hostname}")

    if "smartertime_buckets" in datasources:
        events_smartertime = load_events_smartertime(since)
        events = _join_events(events, events_smartertime, "smartertime")

    # if "toggl" in datasources:
    #    events_toggl = load_toggl(since, now)
    #    events = _join_events(events, events_toggl, "toggl")

    if "fake" in datasources:
        events_fake = list(create_fake_events(start=since, end=now))
        events = _join_events(events, events_fake, "fake")

    # Verify that no events are older than `since`
    print(f"Query start: {since}")
    print(f"Events start: {events[0].timestamp}")
    assert all([since <= e.timestamp for e in events])

    # Verify that no events take place in the future
    # FIXME: Doesn't work with fake data, atm
    if "fake" not in datasources:
        assert all([e.timestamp + e.duration <= now for e in events])

    # Verify that no events overlap
    verify_no_overlap(events)

    # Categorize
    events = classify(events, personal)

    return events


def load_screentime_cached(
    since: datetime | None = None, fast=False, **kwargs
) -> list[Event]:
    # returns screentime from picked cache produced by Dashboard.ipynb (or here)
    # if older than 1 day, it will be regenerated
    path = _cache_file(fast)
    cutoff = datetime.now() - timedelta(days=1)
    if path.exists() and datetime.fromtimestamp(path.stat().st_mtime) > cutoff:
        print(f"Loading from cache: {path}")
        with open(path, "rb") as f:
            events = pickle.load(f)
        # if fast didn't get us enough data to satisfy the query, we need to load the rest
        if fast and since and events[-1].timestamp < since:
            print("Fast couldn't satisfy since, trying again without fast")
            events = load_screentime_cached(since=since, fast=False, **kwargs)
        # trim according to since
        if since:
            events = [e for e in events if e.timestamp >= since]
        return events
    else:
        events = load_screentime(since=since, **kwargs)
        with open(path, "wb") as f:
            pickle.dump(events, f)
        return events


def _join_events(
    old_events: list[Event], new_events: list[Event], source: str
) -> list[Event]:
    if not new_events:
        logger.info(f"No events found from {source}, continuing...")
        return old_events

    logger.info(f"Fetch from {source} complete, joining with the rest...")

    event_first = min(new_events, key=lambda e: e.timestamp)
    event_last = max(new_events, key=lambda e: e.timestamp)
    logger.info(f"  Count: {len(new_events)}")
    logger.info(f"  Start: {event_first.timestamp}")
    logger.info(f"  End:   {event_last.timestamp}")
    verify_no_overlap(new_events)
    events = union_no_overlap(old_events, new_events)
    verify_no_overlap(events)
    return events


def classify(events: list[Event], personal: bool) -> list[Event]:
    # Now load the classes from within the notebook, or from a CSV file.
    config = load_config(use_example=not personal)
    categories_path = Path(config["data"]["categories"]).expanduser()
    # if categories_path is relative, it's relative to the config file
    if not categories_path.is_absolute():
        categories_path = (
            _get_config_path(use_example=not personal).parent / categories_path
        )

    aw_research.classify._init_classes(filename=str(categories_path))
    events = aw_research.classify.classify(events)

    return events


def load_category_df(events: list[Event]) -> pd.DataFrame:
    tss = {}
    all_categories = list({t for e in events for t in e.data["$tags"]})
    for cat in all_categories:
        try:
            tss[cat] = categorytime_per_day(events, cat)
        except Exception as e:
            if "No events to calculate on" not in str(e):
                raise
    df = pd.DataFrame(tss)
    df = df.replace(np.nan, 0)
    df["All_cols"] = df.sum(axis=1)
    df["All_events"] = sum([e.duration / 60 / 60 for e in events], timedelta(0))
    return df


@click.command()
@click.option("--csv", is_flag=True, help="Print as CSV")
def screentime(csv: bool):
    """Load all screentime and print total duration"""
    hostnames = load_config()["data"]["activitywatch"]["hostnames"]
    events = load_screentime(
        since=datetime.now(tz=timezone.utc) - timedelta(days=90),
        datasources=["activitywatch"],
        hostnames=hostnames,
        personal=True,
    )
    logger.info(f"Total duration: {sum((e.duration for e in events), timedelta(0))}")

    df = load_category_df(events)
    if csv:
        print(df.to_csv())
    else:
        print(df)


if __name__ == "__main__":
    screentime()
