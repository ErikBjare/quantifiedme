import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
import random
from typing import Iterable

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

memory = Memory(cache_dir, verbose=0)
logger = logging.getLogger(__name__)


def load_screentime(
    since: datetime,
    datasources: list[str],
    hostnames: list[str],
    personal: bool,
    testing: bool = False,
    cache: bool = True,
    awc: ActivityWatchClient | None = None,
) -> list[Event]:
    now = datetime.now(tz=timezone.utc)

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
            awc = ActivityWatchClient(testing=testing)
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
    # TODO: Move to example config.toml
    example_classes = [
        # (Social) Media
        (r"Facebook|facebook.com", "Social Media", "Media"),
        (r"Reddit|reddit.com", "Social Media", "Media"),
        (r"Spotify|spotify.com", "Music", "Media"),
        (r"subcategory without matching", "Video", "Media"),
        (r"YouTube|youtube.com", "YouTube", "Video"),
        (r"Plex|plex.tv", "Plex", "Video"),
        (r"Fallout 4", "Games", "Media"),
        # Work
        (r"github.com|stackoverflow.com", "Programming", "Work"),
        (r"[Aa]ctivity[Ww]atch|aw-.*", "ActivityWatch", "Programming"),
        (r"[Qq]uantified[Mm]e", "QuantifiedMe", "Programming"),
        (r"[Tt]hankful", "Thankful", "Programming"),
        # School
        (r"subcategory without matching", "School", "Work"),
        (r"Duolingo|Brilliant|Khan Academy", "Self-directed", "School"),
        (r"Analysis in One Variable", "Maths", "School"),
        (r"Applied Machine Learning", "CS", "School"),
        (r"Advanced Web Security", "CS", "School"),
    ]

    # Now load the classes from within the notebook, or from a CSV file.
    load_from_file = True if personal else False
    if load_from_file:
        # TODO: Move categories into config.toml itself
        categories_path = Path(load_config()["data"]["categories"]).expanduser()
        aw_research.classify._init_classes(filename=str(categories_path))
    else:
        logger.info("Using example categories")
        # gives non-sensical type error on check:
        #   Argument "new_classes" to "_init_classes" has incompatible type "List[Tuple[str, str, str]]"; expected "Optional[List[Tuple[str, str, Optional[str]]]]"
        aw_research.classify._init_classes(new_classes=example_classes)  # type: ignore

    events = aw_research.classify.classify(events)

    return events


fakedata_weights = [
    (100, None),
    (2, {"title": "Uncategorized"}),
    (5, {"title": "ActivityWatch"}),
    (4, {"title": "Thankful"}),
    (3, {"title": "QuantifiedMe"}),
    (3, {"title": "FMAA01 - Analysis in One Variable"}),
    (3, {"title": "EDAN95 - Applied Machine Learning"}),
    (2, {"title": "Stack Overflow"}),
    (2, {"title": "phone: Brilliant"}),
    (2, {"url": "youtube.com", "title": "YouTube"}),
    (1, {"url": "reddit.com"}),
    (1, {"url": "facebook.com"}),
    (1, {"title": "Plex"}),
    (1, {"title": "Spotify"}),
    (1, {"title": "Fallout 4"}),
]


# TODO: Merge/replace with aw-fakedata
def create_fake_events(start: datetime, end: datetime) -> Iterable[Event]:
    assert start.tzinfo
    assert end.tzinfo

    # First set RNG seeds to make the notebook reproducible
    random.seed(0)
    np.random.seed(0)

    # Ensures events don't start a few ms in the past
    start += timedelta(seconds=1)

    pareto_alpha = 0.5
    pareto_mode = 5
    time_passed = timedelta()
    while start + time_passed < end:
        duration = timedelta(seconds=np.random.pareto(pareto_alpha) * pareto_mode)
        duration = min([timedelta(hours=1), duration])
        timestamp = start + time_passed
        time_passed += duration
        if start + time_passed > end:
            break
        data = random.choices(
            [d[1] for d in fakedata_weights], [d[0] for d in fakedata_weights]
        )[0]
        if data:
            yield Event(timestamp=timestamp, duration=duration, data=data)


def load_category_df(events: list[Event]):
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
    # TODO: Get awc/hostname parameters from config
    events = load_screentime(
        since=datetime.now(tz=timezone.utc) - timedelta(days=90),
        datasources=["activitywatch"],
        hostnames=["erb-main2-arch", "erb-m2.localdomain"],
        personal=True,
        awc=ActivityWatchClient("quantifiedme-screentime", port=5667),
    )
    print(f"Total duration: {sum((e.duration for e in events), timedelta(0))}")

    print(load_category_df(events))


if __name__ == "__main__":
    screentime()
