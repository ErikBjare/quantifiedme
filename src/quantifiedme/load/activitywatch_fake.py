import random
from collections.abc import Iterable
from datetime import datetime, timedelta

import numpy as np

from aw_core import Event

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
