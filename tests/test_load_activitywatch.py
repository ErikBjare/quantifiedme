import os
from datetime import datetime, timezone

from aw_client import ActivityWatchClient
from quantifiedme.load.activitywatch import load_events

hostname = os.uname().nodename


def test_load_events():
    awc = ActivityWatchClient("testloadevents", port=5600, testing=False)
    hostname = os.uname().nodename
    now = datetime.now(tz=timezone.utc)
    today = datetime.combine(now, datetime.min.time(), tzinfo=timezone.utc)
    since = today
    end = now
    events = load_events(awc, hostname, since, end)
    assert events
    print(len(events))


if __name__ == "__main__":
    test_load_events()
