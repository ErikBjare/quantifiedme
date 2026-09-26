"""
This was originally part of aw-research, which in turn was based on/refactored out of the QuantifiedMe notebook.
"""

import logging
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import aw_client.queries
import aw_research
import aw_research.classify
from aw_client import ActivityWatchClient
from aw_core import Event

from ..cache import memory

logger = logging.getLogger(__name__)

# aw-client > 0.5.15 ships multidevice query helpers (ActivityWatch/aw-client#121).
# Until that is released, fall back to one aw-research query per desktop host.
HAS_MULTIDEVICE = hasattr(aw_client.queries, "canonicalMultideviceEvents")
# Typed as Any so type checking also passes against aw-client without the helpers
_queries: Any = aw_client.queries

# A host's buckets, as a hashable spec (also used as part of the cache key):
#   ("desktop", hostname, bid_window, bid_afk, bid_browsers)
#   ("android", hostname, bid_android)
HostSpec = tuple


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
    return _postprocess([Event(**e) for e in result[0]], since, end)


def _postprocess(events: list[Event], since: datetime, end: datetime) -> list[Event]:
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


def discover_hosts(
    buckets: dict[str, dict[str, Any]],
    hostnames: Sequence[str] | None = None,
    exclude: Sequence[str] = (),
    include_mobile: bool = True,
) -> list[HostSpec]:
    """Discover hosts to query from bucket metadata, in priority order.

    If ``hostnames`` is given, only those hosts are used, in that order.
    Otherwise all hosts with usable buckets are used: desktop hosts (window +
    afk bucket, including ones synced with aw-sync) first, then mobile hosts,
    each ordered by most recent activity. Mobile hosts need aw-client with
    multidevice support and are skipped otherwise.
    """
    if HAS_MULTIDEVICE:
        params = _queries.multideviceHostParams(buckets, hosts=hostnames)
        specs: list[HostSpec] = []
        for p in params:
            if isinstance(p, aw_client.queries.DesktopQueryParams):
                hostname = _hostname(p.bid_window, buckets[p.bid_window])
                specs.append(
                    (
                        "desktop",
                        hostname,
                        p.bid_window,
                        p.bid_afk,
                        tuple(p.bid_browsers),
                    )
                )
            elif include_mobile:
                hostname = _hostname(p.bid_android, buckets[p.bid_android])
                specs.append(("android", hostname, p.bid_android))
    else:
        specs = _discover_desktop_hosts(buckets, hostnames)
    return [s for s in specs if s[1] not in exclude]


def _hostname(bid: str, bucket: dict[str, Any]) -> str | None:
    for hostname in (
        bucket.get("hostname"),
        (bucket.get("data") or {}).get("hostname"),
        bid.rsplit("-synced-from-", 1)[1] if "-synced-from-" in bid else None,
    ):
        if hostname and hostname != "unknown":
            return hostname
    return None


def _discover_desktop_hosts(
    buckets: dict[str, dict[str, Any]], hostnames: Sequence[str] | None
) -> list[HostSpec]:
    """Fallback discovery for aw-client without multidevice support."""
    by_host: dict[str, dict[str, str]] = {}
    last_updated: dict[str, str] = {}
    for bid, bucket in buckets.items():
        hostname = _hostname(bid, bucket)
        if hostname is None:
            continue
        role = {"currentwindow": "window", "afkstatus": "afk"}.get(
            bucket.get("type") or ""
        )
        if role is None or bid.startswith("aw-watcher-android"):
            continue
        by_host.setdefault(hostname, {})[role] = bid
        last_updated[hostname] = max(
            last_updated.get(hostname, ""), bucket.get("last_updated") or ""
        )
    usable = [h for h, roles in by_host.items() if {"window", "afk"} <= roles.keys()]
    if hostnames is None:
        hostnames = sorted(usable, key=lambda h: last_updated[h], reverse=True)
    return [
        ("desktop", h, by_host[h]["window"], by_host[h]["afk"], ())
        for h in hostnames
        if h in usable
    ]


def _host_params(spec: HostSpec):
    # Events are classified client-side (see derived.screentime.classify), so
    # pass fixed classes rather than letting aw-client fetch them from a server.
    from aw_client.classes import default_classes

    if spec[0] == "desktop":
        _, _, bid_window, bid_afk, bid_browsers = spec
        return aw_client.queries.DesktopQueryParams(
            bid_window=bid_window,
            bid_afk=bid_afk,
            bid_browsers=list(bid_browsers),
            classes=default_classes,
        )
    return aw_client.queries.AndroidQueryParams(
        bid_android=spec[2], classes=default_classes
    )


@memory.cache(ignore=["awc"])
def load_events_host(
    awc: ActivityWatchClient,
    host: HostSpec,
    since: datetime,
    end: datetime,
) -> list[Event]:
    """Load canonical events for one discovered host (desktop or Android).

    Uses the same per-host query as ``canonicalMultideviceEvents`` in aw-client
    (exact bucket IDs, Android events not merged by app). Hosts are combined
    client-side (see ``derived.screentime``) so each event keeps its
    ``$hostname``. ``host`` (from :func:`discover_hosts`) is part of the cache
    key, so the cache is invalidated when a host's buckets change.

    Results for a time range are cached indefinitely, so events synced after a
    range was first loaded (e.g. a device that syncs late) only show up once
    the cache is cleared (``load_screentime(cache=False)``).
    """
    query = (
        _queries.canonicalEvents(
            _host_params(host), exact_bucket_ids=True, merge_android=False
        )
        + "\nRETURN = sort_by_timestamp(events);"
    )
    logger.debug(f"Query:\n{query}")

    result = awc.query(query, timeperiods=[(since, end)])
    return _postprocess([Event(**e) for e in result[0]], since, end)
