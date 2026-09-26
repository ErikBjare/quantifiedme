import os
from datetime import datetime, timezone

import pytest
from aw_client import ActivityWatchClient
from quantifiedme.load.activitywatch import load_events

hostname = os.uname().nodename


@pytest.mark.xfail(
    reason="aw_research query incompatible with aw-server v0.12.3b14 (400/500 errors)",
    strict=False,
)
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


BUCKETS = {
    "aw-watcher-window_laptop": {
        "type": "currentwindow",
        "hostname": "laptop",
        "last_updated": "2026-09-01",
    },
    "aw-watcher-afk_laptop": {
        "type": "afkstatus",
        "hostname": "laptop",
        "last_updated": "2026-09-01",
    },
    "aw-watcher-window_desktop-synced-from-desktop": {
        "type": "currentwindow",
        "hostname": "desktop",
        "last_updated": "2025-01-01",
    },
    "aw-watcher-afk_desktop-synced-from-desktop": {
        "type": "afkstatus",
        "hostname": "desktop",
        "last_updated": "2025-01-01",
    },
    "aw-watcher-window_windowonly": {
        "type": "currentwindow",
        "hostname": "windowonly",
        "last_updated": "2026-09-01",
    },
    "aw-watcher-android-synced-from-phone": {
        "type": "currentwindow",
        "hostname": "phone",
        "last_updated": "2026-09-10",
    },
}


@pytest.fixture(params=[False, True], ids=["fallback", "multidevice"])
def has_multidevice(request, monkeypatch):
    import aw_client.queries
    import quantifiedme.load.activitywatch as mod

    if request.param and not hasattr(aw_client.queries, "canonicalMultideviceEvents"):
        pytest.skip("aw-client without multidevice support")
    monkeypatch.setattr(mod, "HAS_MULTIDEVICE", request.param)
    return request.param


def test_discover_hosts(has_multidevice):
    from quantifiedme.load.activitywatch import discover_hosts

    hosts = discover_hosts(BUCKETS)
    names = [h[1] for h in hosts]
    # Desktop hosts first (most recent first), synced hosts included
    assert names[:2] == ["laptop", "desktop"]
    assert hosts[1][2] == "aw-watcher-window_desktop-synced-from-desktop"
    assert "windowonly" not in names
    # Mobile hosts need aw-client with multidevice support
    assert ("phone" in names) == has_multidevice

    assert "phone" not in [h[1] for h in discover_hosts(BUCKETS, include_mobile=False)]
    assert [h[1] for h in discover_hosts(BUCKETS, exclude=["laptop"])][0] == "desktop"
    # Explicit hostnames select and order hosts
    hosts = discover_hosts(BUCKETS, hostnames=["desktop", "missing", "laptop"])
    assert [h[1] for h in hosts] == ["desktop", "laptop"]


class _FakeClient:
    """Answers each per-host query with events from the bucket it queries."""

    def __init__(self, buckets, events_by_bucket):
        self.buckets = buckets
        self.events_by_bucket = events_by_bucket

    def get_buckets(self):
        return self.buckets

    def query(self, query, timeperiods):
        for bid, events in self.events_by_bucket.items():
            if f'query_bucket("{bid}")' in query:
                return [events]
        return [[]]


def test_load_activitywatch_combines_hosts(monkeypatch, has_multidevice):
    from datetime import timedelta

    import quantifiedme.derived.screentime as screentime
    import quantifiedme.load.activitywatch as mod

    if not has_multidevice:
        pytest.skip("fallback path uses the aw-research query")
    # Bypass the joblib cache
    monkeypatch.setattr(screentime, "load_events_host", mod.load_events_host.func)

    t0 = datetime(2026, 1, 5, 12, tzinfo=timezone.utc)

    def ev(minutes, duration, app):
        return {
            "timestamp": (t0 + timedelta(minutes=minutes)).isoformat(),
            "duration": duration * 60,
            "data": {"app": app},
        }

    awc = _FakeClient(
        BUCKETS,
        {
            "aw-watcher-window_laptop": [ev(0, 30, "code")],
            "aw-watcher-android-synced-from-phone": [ev(20, 20, "Chat")],
        },
    )
    config: dict = {"data": {"activitywatch": {}}}
    events = screentime._load_activitywatch(
        awc,  # type: ignore[arg-type]
        config,
        t0 - timedelta(days=1),
        t0 + timedelta(days=1),
    )
    by_host = {
        h: sum((e.duration for e in events if e.data["$hostname"] == h), timedelta())
        for h in ("laptop", "phone")
    }
    # laptop has priority; the phone only fills the 10 minutes after it
    assert by_host == {"laptop": timedelta(minutes=30), "phone": timedelta(minutes=10)}
    assert all(e.data["$source"] == "activitywatch" for e in events)
