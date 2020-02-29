"""
Much of this taken from the QuantifiedMe notebook
"""

from typing import List, Iterable
from datetime import datetime, timedelta, timezone
import random

import pandas as pd
import numpy as np
import click
from aw_core import Event
import aw_research
from aw_research.classify import _union_no_overlap
from aw_research import verify_no_overlap, split_into_weeks, split_into_days
from aw_research import split_event_on_hour, categorytime_per_day, categorytime_during_day, start_of_day, end_of_day
from aw_research import load_toggl

from .config import load_config

fakedata_weights = [
    (100, None),
    (2, {'title': 'Uncategorized'}),
    (5, {'title': 'ActivityWatch'}),
    (4, {'title': 'Thankful'}),
    (3, {'title': 'QuantifiedMe'}),
    (3, {'title': 'FMAA01 - Analysis in One Variable'}),
    (3, {'title': 'EDAN95 - Applied Machine Learning'}),
    (2, {'title': 'Stack Overflow'}),
    (2, {'title': 'phone: Brilliant'}),
    (2, {'url': 'youtube.com', 'title': 'YouTube'}),
    (1, {'url': 'reddit.com'}),
    (1, {'url': 'facebook.com'}),
    (1, {'title': 'Plex'}),
    (1, {'title': 'Spotify'}),
    (1, {'title': 'Fallout 4'}),
]


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
        data = random.choices([d[1] for d in fakedata_weights], [d[0] for d in fakedata_weights])[0]
        if data:
            yield Event(timestamp=timestamp, duration=duration, data=data)


def load_smartertime(since) -> List[Event]:
    events_smartertime: List[Event] = []
    # TODO: Use paths from config file
    for smartertime_awbucket_path in load_config()["data"]["smartertime_buckets"]:
        new_events = aw_research.classify._get_events_smartertime(since, filepath=smartertime_awbucket_path)
        events_smartertime = _union_no_overlap(events_smartertime, new_events)
    for e in events_smartertime:
        e.data['$source'] = 'smartertime'
    return events_smartertime


# TODO: Make it possible to select which hostname to get a timeline for
def load_complete_timeline(since: datetime, datasources: List[str] = None, hostnames: List[str] = ["erb-main2", "erb-main2-arch", "erb-laptop2-arch", "SHADOW-DEADGSK6"]):
    now = datetime.now(tz=timezone.utc)

    assert since.tzinfo
    assert now.tzinfo

    # TODO: Load available datasources from config file
    if not datasources:
        datasources = ["activitywatch", "smartertime"]

    # Check for invalid sources
    for source in datasources:
        assert source in ["activitywatch", "smartertime", "fake", "toggl"]

    events: List[Event] = []

    # TODO: Load from multiple devices
    if 'activitywatch' in datasources:
        for hostname in hostnames:
            # Split up into previous days and today, to take advantage of caching
            # TODO: Split up into whole days
            events_aw: List[Event] = []
            for dtstart, dtend in split_into_weeks(since, now):
                events_aw += aw_research.classify.get_events(hostname, since=dtstart, end=dtend, include_smartertime=False, include_toggl=False)
                print(len(events_aw))
            for e in events_aw:
                e.data['$source'] = 'activitywatch'

            events = _union_no_overlap(events, events_aw)
            verify_no_overlap(events)

    # The above code does caching using joblib, use the following if you want to clear the cache:
    #aw_research.classify.memory.clear()

    if 'smartertime' in datasources:
        events_smartertime = load_smartertime(since)
        verify_no_overlap(events_smartertime)
        events = _union_no_overlap(events, events_smartertime)
        verify_no_overlap(events)

    if 'toggl' in datasources:
        events_toggl = load_toggl(since, now)
        verify_no_overlap(events_toggl)
        print(f"Oldest: {min(events_toggl, key=lambda e: e.timestamp).timestamp}")
        events = _union_no_overlap(events, events_toggl)
        verify_no_overlap(events)

    if 'fake' in datasources:
        fake_events = list(create_fake_events(start=since, end=now))
        events += fake_events

    # Verify that no events are older than `since`
    print(since)
    print(events[0].timestamp)
    assert all([since <= e.timestamp for e in events])

    # Verify that no events take place in the future
    # FIXME: Doesn't work with fake data, atm
    if 'fake' not in datasources:
        assert all([e.timestamp + e.duration <= now for e in events])

    # Verify that no events overlap
    verify_no_overlap(events)

    # Categorize
    events = classify(events)

    return events


def classify(events: List[Event]) -> List[Event]:
    # TODO: Move to example config.toml
    classes = [
        # (Social) Media
        (r'Facebook|facebook.com', 'Social Media', 'Media'),
        (r'Reddit|reddit.com', 'Social Media', 'Media'),
        (r'Spotify|spotify.com', 'Music', 'Media'),
        (r'subcategory without matching', 'Video', 'Media'),
        (r'YouTube|youtube.com', 'YouTube', 'Video'),
        (r'Plex|plex.tv', 'Plex', 'Video'),
        (r'Fallout 4', 'Games', 'Media'),

        # Work
        (r'github.com|stackoverflow.com', 'Programming', 'Work'),
        (r'[Aa]ctivity[Ww]atch|aw-.*', 'ActivityWatch', 'Programming'),
        (r'[Qq]uantified[Mm]e', 'QuantifiedMe', 'Programming'),
        (r'[Tt]hankful', 'Thankful', 'Programming'),

        # School
        (r'subcategory without matching', 'School', 'Work'),
        (r'Duolingo|Brilliant|Khan Academy', 'Self-directed', 'School'),
        (r'Analysis in One Variable', 'Maths', 'School'),
        (r'Applied Machine Learning', 'CS', 'School'),
        (r'Advanced Web Security', 'CS', 'School'),
    ]

    personal = True

    # Now load the classes from within the notebook, or from a CSV file.
    load_from_file = True if personal else False
    if load_from_file:
        # TODO: Move categories into config.toml itself
        aw_research.classify._init_classes(filename=load_config()["data"]["categories"])
    else:
        aw_research.classify._init_classes(new_classes=classes)

    events = aw_research.classify.classify(events)

    return events


def load_category_df(events: List[Event]):
    tss = {}
    all_categories = list(set(t for e in events for t in e.data['$tags']))
    for cat in all_categories:
        try:
            tss[cat] = categorytime_per_day(events, cat)
        except Exception as e:
            if "No events to calculate on" not in str(e):
                raise
    return pd.DataFrame(tss)


@click.command()
def activitywatch():
    events = load_complete_timeline(datetime.now(tz=timezone.utc) - timedelta(days=90))
    print(f"Total duration: {sum((e.duration for e in events), timedelta(0))}")

    print(load_category_df(events))


if __name__ == "__main__":
    activitywatch()
