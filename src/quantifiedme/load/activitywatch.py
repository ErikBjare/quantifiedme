"""
This was originally part of aw-research, which in turn was based on/refactored out of the QuantifiedMe notebook.
"""

import logging
from datetime import datetime, timezone
from urllib.parse import urlparse

import aw_research
import aw_research.classify
from aw_client import ActivityWatchClient
from aw_core import Event

from ..cache import memory

logger = logging.getLogger(__name__)


@memory.cache(ignore=["awc"])
def load_events(
    awc: ActivityWatchClient,
    hostname: str,
    since: datetime,
    end: datetime,
) -> list[Event]:
    query = aw_research.classify.build_query(hostname)
    logger.debug(f"Query:\n{query}")

    result = awc.query(query, timeperiods=[(since, end)])
    events = [Event(**e) for e in result[0]]

    # Filter by time
    events = [
        e
        for e in events
        if since.astimezone(timezone.utc) < e.timestamp
        and e.timestamp + e.duration < end.astimezone(timezone.utc)
    ]
    assert all(since.astimezone(timezone.utc) < e.timestamp for e in events)
    assert all(e.timestamp + e.duration < end.astimezone(timezone.utc) for e in events)

    # Filter out events without data (which sometimes happens for whatever reason)
    events = [e for e in events if e.data]

    for event in events:
        if "app" not in event.data:
            if "url" in event.data:
                event.data["app"] = urlparse(event.data["url"]).netloc
            else:
                print("Unexpected event: ", event)

    events = [e for e in events if e.data]
    return events


def test_load_events():
    awc = ActivityWatchClient("testloadevents", port=5667, testing=True)
    hostname = "erb-main2-arch"
    since = datetime(2020, 1, 1, tzinfo=timezone.utc)
    end = datetime(2020, 1, 2, tzinfo=timezone.utc)
    events = load_events(awc, hostname, since, end)
    assert events
    print(len(events))


if __name__ == "__main__":
    test_load_events()
