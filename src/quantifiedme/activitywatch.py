"""
Much of this taken from the QuantifiedMe notebook
"""

from typing import List, Iterable, Dict
from datetime import datetime, timedelta, timezone
import logging
import random

import pandas as pd
import numpy as np
import click

from aw_core import Event
from aw_client import ActivityWatchClient
from aw_transform.union_no_overlap import union_no_overlap

import aw_research
import aw_research.classify
from aw_research import verify_no_overlap, split_into_weeks
from aw_research import categorytime_per_day

from .load_toggl import load_toggl

from .config import load_config

logger = logging.getLogger(__name__)

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


def _load_smartertime_devices(since: datetime) -> Dict[str, List]:
    result = {}
    for hostname, smartertime_awbucket_path in load_config()["data"][
        "smartertime_buckets"
    ].items():
        events = aw_research.classify._get_events_smartertime(
            since, filepath=smartertime_awbucket_path
        )
        for e in events:
            e.data["$source"] = "smartertime"
            e.data["$hostname"] = hostname
        result[hostname] = events
    return result


def load_smartertime(since: datetime) -> List[Event]:
    events_smartertime: List[Event] = []
    for hostname, events in _load_smartertime_devices(since).items():
        events_smartertime = union_no_overlap(events_smartertime, events)
    return events_smartertime


def _join_events(
    old_events: List[Event], new_events: List[Event], source: str
) -> List[Event]:
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


def load_complete_timeline(
    since: datetime,
    datasources: List[str],
    hostnames: List[str],
    personal: bool,
    testing: bool = False,
    cache: bool = True,
    awc: ActivityWatchClient = None,
):
    now = datetime.now(tz=timezone.utc)

    # The below code does caching using joblib, setting cache=False clears the cache.
    if not cache:
        aw_research.classify.memory.clear()

    assert since.tzinfo
    assert now.tzinfo

    # TODO: Load available datasources from config file
    if not datasources:
        datasources = ["activitywatch", "smartertime"]

    # Check for invalid sources
    for source in datasources:
        assert source in ["activitywatch", "smartertime", "fake", "toggl"]

    events: List[Event] = []

    if "activitywatch" in datasources:
        if awc is None:
            awc = ActivityWatchClient(testing=testing)
        for hostname in hostnames:
            logger.info(f"Getting events for {hostname}...")
            # Split up into previous days and today, to take advantage of caching
            # TODO: Split up into whole days
            # TODO: Use `aw_client.queries.canonicalQuery` instead
            events_aw: List[Event] = []
            for dtstart, dtend in split_into_weeks(since, now):
                events_aw += aw_research.classify.get_events(
                    awc,
                    hostname,
                    since=dtstart,
                    end=dtend,
                    include_smartertime=False,
                    include_toggl=False,
                )
                logger.debug(f"{len(events_aw)} events retreived")
            for e in events_aw:
                e.data["$hostname"] = hostname
                e.data["$source"] = "activitywatch"
            events = _join_events(events, events_aw, f"activitywatch {hostname}")

    if "smartertime" in datasources:
        events_smartertime = load_smartertime(since)
        events = _join_events(events, events_smartertime, "smartertime")

    if "toggl" in datasources:
        events_toggl = load_toggl(since, now)
        events = _join_events(events, events_toggl, "toggl")

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


def classify(events: List[Event], personal: bool) -> List[Event]:
    # TODO: Move to example config.toml
    classes = [
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
        aw_research.classify._init_classes(filename=load_config()["data"]["categories"])
    else:
        logger.info("Using example categories")
        # gives non-sensical type error on check:
        #   Argument "new_classes" to "_init_classes" has incompatible type "List[Tuple[str, str, str]]"; expected "Optional[List[Tuple[str, str, Optional[str]]]]"
        aw_research.classify._init_classes(new_classes=classes)  # type: ignore

    events = aw_research.classify.classify(events)

    return events


def load_category_df(events: List[Event]):
    tss = {}
    all_categories = list(set(t for e in events for t in e.data["$tags"]))
    for cat in all_categories:
        try:
            tss[cat] = categorytime_per_day(events, cat)
        except Exception as e:
            if "No events to calculate on" not in str(e):
                raise
    df = pd.DataFrame(tss)
    df.replace(np.nan, 0)
    return df


@click.command()
def activitywatch():
    events = load_complete_timeline(datetime.now(tz=timezone.utc) - timedelta(days=90))
    print(f"Total duration: {sum((e.duration for e in events), timedelta(0))}")

    print(load_category_df(events))


if __name__ == "__main__":
    activitywatch()
