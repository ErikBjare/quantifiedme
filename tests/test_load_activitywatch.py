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
