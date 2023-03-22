import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

import click
import pandas as pd
import numpy as np
from joblib import Memory

from aw_core import Event
from aw_client import ActivityWatchClient
from aw_transform.union_no_overlap import union_no_overlap

import aw_research.classify
from aw_research.util import verify_no_overlap, split_into_weeks, categorytime_per_day

from ..load.activitywatch import load_events as load_events_activitywatch
from ..load.smartertime import load_events as load_events_smartertime

from ..config import load_config
from ..cache import cache_dir

from ..load.activitywatch_fake import create_fake_events

memory = Memory(cache_dir, verbose=0)
logger = logging.getLogger(__name__)


def _get_aw_client(testing: bool) -> ActivityWatchClient:
    config = load_config()
    sec_aw = config["data"].get("activitywatch", {})
    port = sec_aw.get("port", 5600 if not testing else 5666)
    return ActivityWatchClient(port=port, testing=testing)


def load_screentime(
    since: datetime | None,
    datasources: list[str],
    hostnames: list[str],
    personal: bool,
    testing: bool = False,
    cache: bool = True,
    awc: ActivityWatchClient | None = None,
) -> list[Event]:
    now = datetime.now(tz=timezone.utc)
    if since is None:
        since = now - timedelta(days=365)

    # The below code does caching using joblib, setting cache=False clears the cache.
    if not cache:
        memory.clear()

    assert since.tzinfo
    assert now.tzinfo

    # Check for invalid sources
    for source in datasources:
        assert source in ["activitywatch", "smartertime", "fake", "toggl"]

    events: list[Event] = []

    if "activitywatch" in datasources:
        if awc is None:
            awc = _get_aw_client(testing)
        for hostname in hostnames:
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

    if "smartertime" in datasources:
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
    use_example = not personal
    if use_example:
        logger.info("Using example categories")
        categories_path = Path(__file__).parent / "categories.example.toml"
    else:
        categories_path = Path(load_config()["data"]["categories"]).expanduser()

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
    return df


@click.command()
def screentime():
    """Load all screentime and print total duration"""
    # TODO: Get awc parameters from config
    hostnames = load_config()["data"]["activitywatch"]["hostnames"]
    events = load_screentime(
        since=datetime.now(tz=timezone.utc) - timedelta(days=90),
        datasources=["activitywatch"],
        hostnames=hostnames,
        personal=True,
    )
    print(f"Total duration: {sum((e.duration for e in events), timedelta(0))}")

    print(load_category_df(events))


if __name__ == "__main__":
    screentime()
