import hashlib
import json
import logging
import pickle
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import aw_research.classify
import click
import numpy as np
import pandas as pd
from aw_client import ActivityWatchClient
from aw_core import Event
from aw_research.util import categorytime_per_day, split_into_weeks, verify_no_overlap
from aw_transform.union_no_overlap import union_no_overlap

from ..cache import cache_dir, memory
from ..config import _get_config_path, load_config
from ..load import activitywatch as aw_load
from ..load.activitywatch import discover_hosts, load_events_host
from ..load.activitywatch import load_events as load_events_activitywatch
from ..load.activitywatch_fake import create_fake_events
from ..load.smartertime import load_events as load_events_smartertime

logger = logging.getLogger(__name__)


# Category rules, as given to aw_research.classify._init_classes: either
# {"new_classes": [(regex, tag, parent_tag), ...]} or {"filename": path}.
CategoryRules = dict


# Bump when the loader changes what ends up in the screentime event cache, so
# existing caches are not reused.
CACHE_VERSION = 1

# How long cached screentime events are reused before being fetched again.
CACHE_TTL = timedelta(days=1)

# Cache file names used before the cache was keyed.
_LEGACY_CACHE_FILES = ("events.pickle", "events_fast.pickle")


def _cache_file(fast: bool, key: str) -> Path:
    return cache_dir / f"events-{'fast' if fast else 'full'}-{key}.pickle"


def _get_aw_client(testing: bool) -> ActivityWatchClient:
    config = load_config(use_example=testing)
    sec_aw = config["data"].get("activitywatch", {})
    port = sec_aw.get("port", 5600 if not testing else 5666)
    return ActivityWatchClient(port=port, testing=testing)


DatasourceType = Literal["activitywatch", "smartertime_buckets", "fake", "toggl"]


def load_screentime(
    since: datetime | None = None,
    datasources: list[DatasourceType] | None = None,
    hostnames: list[str] | None = None,
    personal: bool = True,
    cache: bool = True,
    awc: ActivityWatchClient | None = None,
) -> list[Event]:
    """Load screentime events from all datasources, categorized."""
    events = _load_screentime_raw(since, datasources, hostnames, personal, cache, awc)
    return classify(events, personal)


def _resolve_datasources(
    config, datasources: list[DatasourceType] | None
) -> list[DatasourceType]:
    """Auto-detect datasources from config if not specified, and validate them."""
    if datasources is None:
        datasources = []
        if "activitywatch" in config["data"]:
            datasources.append("activitywatch")
        if "smartertime_buckets" in config["data"]:
            datasources.append("smartertime_buckets")

    for source in datasources:
        assert source in [
            "activitywatch",
            "smartertime_buckets",
            "fake",
            "toggl",
        ], f"Invalid source: {source}"
    return datasources


def _load_screentime_raw(
    since: datetime | None = None,
    datasources: list[DatasourceType] | None = None,
    hostnames: list[str] | None = None,
    personal: bool = True,
    cache: bool = True,
    awc: ActivityWatchClient | None = None,
) -> list[Event]:
    """Load screentime events from all datasources, not yet categorized."""
    config = load_config(use_example=not personal)

    now = datetime.now(tz=timezone.utc)
    if since is None:
        since = now - timedelta(days=365)
    else:
        assert since.tzinfo

    # The below code does caching using joblib, setting cache=False clears the cache.
    if not cache:
        memory.clear()

    datasources = _resolve_datasources(config, datasources)

    events: list[Event] = []

    if "activitywatch" in datasources:
        if awc is None:
            awc = _get_aw_client(not personal)
        events = _join_events(
            events,
            _load_activitywatch(awc, config, since, now, hostnames),
            "activitywatch",
        )

    if "smartertime_buckets" in datasources:
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
    assert all(since <= e.timestamp for e in events)

    # Verify that no events take place in the future
    # FIXME: Doesn't work with fake data, atm
    if "fake" not in datasources:
        assert all(e.timestamp + e.duration <= now for e in events)

    # Verify that no events overlap
    verify_no_overlap(events)

    return events


def _discover_aw_hosts(
    awc: ActivityWatchClient, config, hostnames: list[str] | None = None
) -> list:
    """The ActivityWatch hosts to load, per the ``[data.activitywatch]`` settings."""
    sec_aw = config["data"].get("activitywatch", {})
    return discover_hosts(
        awc.get_buckets(),
        hostnames=hostnames or sec_aw.get("hostnames") or None,
        exclude=sec_aw.get("exclude_hostnames", []),
        include_mobile=sec_aw.get("include_mobile", True),
    )


def _load_activitywatch(
    awc: ActivityWatchClient,
    config,
    since: datetime,
    now: datetime,
    hostnames: list[str] | None = None,
) -> list[Event]:
    """Load events from all ActivityWatch hosts, combined without overlap.

    Hosts are discovered from bucket metadata (including buckets synced with
    aw-sync), so new devices are picked up automatically. Config options in
    ``[data.activitywatch]``:

    - ``hostnames``: only use these hosts, in this priority order (optional)
    - ``exclude_hostnames``: hosts to skip (optional)
    - ``include_mobile``: include Android devices (default: true, needs
      aw-client with multidevice support)

    Where hosts overlap in time, the earlier (higher priority) host wins.
    """
    hosts = _discover_aw_hosts(awc, config, hostnames)
    logger.info(f"ActivityWatch hosts (in priority order): {[h[1] for h in hosts]}")

    # Split up into weeks, to take advantage of caching
    # TODO: Split up into whole days
    # One query per host, combined here in priority order (first host wins
    # where hosts overlap), so each event keeps its $hostname.
    events: list[Event] = []
    for host in hosts:
        hostname = host[1]
        logger.info(f"Getting events for {hostname}...")
        events_aw: list[Event] = []
        for dtstart, dtend in split_into_weeks(since, now):
            if aw_load.HAS_MULTIDEVICE:
                events_aw += load_events_host(awc, host, since=dtstart, end=dtend)
            else:
                # aw-client without multidevice support: aw-research query
                events_aw += load_events_activitywatch(
                    awc, hostname, since=dtstart, end=dtend
                )
            logger.debug(f"{len(events_aw)} events retreived")
        for e in events_aw:
            e.data["$hostname"] = hostname
            e.data["$source"] = "activitywatch"
        events = _join_events(events, events_aw, f"activitywatch {hostname}")
    return events


def screentime_cache_key(
    datasources: list[DatasourceType] | None = None,
    hostnames: list[str] | None = None,
    personal: bool = True,
    awc: ActivityWatchClient | None = None,
    rules: CategoryRules | None = None,
) -> str:
    """Key for the screentime event cache.

    Covers everything that changes the cached (categorized) events: the category
    rules, the datasources, the ActivityWatch host settings and the hosts they
    resolve to (so a newly discovered device invalidates the cache), and
    :data:`CACHE_VERSION`.
    """
    config = load_config(use_example=not personal)
    datasources = _resolve_datasources(config, datasources)
    sec_aw = config["data"].get("activitywatch", {})
    if rules is None:
        rules = load_category_rules(personal)
    parts: dict = {
        "version": CACHE_VERSION,
        "categories": _category_rules_fingerprint(rules),
        "personal": personal,
        "datasources": sorted(datasources),
        "hostnames": hostnames,
        "activitywatch": {
            k: sec_aw.get(k)
            for k in ("port", "hostnames", "exclude_hostnames", "include_mobile")
        },
    }
    if "activitywatch" in datasources:
        try:
            awc = awc or _get_aw_client(not personal)
            hosts = _discover_aw_hosts(awc, config, hostnames)
            # sorted: discovery orders hosts by most recent activity, which
            # changes whenever another device is used
            parts["hosts"] = sorted(json.dumps(h) for h in hosts)
        except Exception as e:
            # can't reach aw-server: key on the host settings alone
            logger.warning(f"Failed to discover ActivityWatch hosts: {e}")
    if "smartertime_buckets" in datasources:
        parts["smartertime_buckets"] = config["data"]["smartertime_buckets"]
    blob = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _read_cache(path: Path, since: datetime) -> list[Event] | None:
    """Return the cached raw events if the cache is fresh and covers `since`."""
    if not path.exists():
        return None
    if datetime.now() - datetime.fromtimestamp(path.stat().st_mtime) > CACHE_TTL:
        return None
    with open(path, "rb") as f:
        cached = pickle.load(f)
    # The query start moves with the clock, so allow for it within the TTL.
    if cached["since"] > since + CACHE_TTL:
        return None
    print(f"Loading from cache: {path}")
    return cached["events"]


def _cleanup_cache(fast: bool, keep: Path) -> None:
    """Remove cache files written under other keys (and pre-key legacy files)."""
    mode = "fast" if fast else "full"
    stale = [p for p in cache_dir.glob(f"events-{mode}-*.pickle") if p != keep]
    stale += [cache_dir / name for name in _LEGACY_CACHE_FILES]
    for p in stale:
        if p.exists():
            logger.info(f"Removing stale screentime cache: {p}")
            p.unlink()


def load_screentime_cached(
    since: datetime | None = None, fast=False, **kwargs
) -> list[Event]:
    """Like :func:`load_screentime`, but reuses events cached within the last day.

    The cache holds categorized events, keyed by :func:`screentime_cache_key`, so
    changing the category rules, datasources or hosts loads fresh events rather
    than reusing stale ones. (Categorizing is too slow to redo on every load:
    O(events x rules). Re-fetching is mostly served by the per-week joblib cache
    in ``load.activitywatch``.) ``fast`` loads (a short range, used
    by :func:`load_all_df`) get their own cache file so they don't evict the full
    one, and can be served from a full cache that covers their range.
    """
    personal = kwargs.get("personal", True)
    if since is None:
        since = datetime.now(tz=timezone.utc) - timedelta(days=365)
    rules = load_category_rules(personal)
    key = screentime_cache_key(
        kwargs.get("datasources"),
        kwargs.get("hostnames"),
        personal,
        kwargs.get("awc"),
        rules,
    )
    path = _cache_file(fast, key)
    candidates = [path, _cache_file(False, key)] if fast else [path]
    for candidate in candidates:
        events = _read_cache(candidate, since)
        if events is not None:
            return [e for e in events if e.timestamp >= since]

    events = classify(_load_screentime_raw(since=since, **kwargs), personal, rules)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump({"since": since, "events": events}, f)
    tmp.replace(path)
    _cleanup_cache(fast, keep=path)
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


# Config value for `[data] categories` that selects the aw-server category rules.
SERVER_CATEGORIES = "server"

# Regex that never matches, for categories whose rule type is "none" (pure parents).
_NEVER_MATCH = "(?!)"


def _category_tags(names: list[list[str]]) -> dict[tuple[str, ...], str]:
    """Pick a unique tag for each category path.

    aw_research identifies categories by a single name, while aw-server categories
    are paths (e.g. ["Media", "Games"] and ["P", "Games"]). The tag is the leaf name
    when that is unambiguous, otherwise the shortest path suffix that is unique,
    joined with ">" (e.g. "Media>Games"). Surrounding double quotes are stripped
    (the web UI allows names like '"Social"').
    """
    paths = [tuple(n.strip('"') for n in name) for name in names]
    tags: dict[tuple[str, ...], str] = {}
    used: set[str] = set()
    # sorted, so tags don't depend on the order of rules in the server settings
    for orig, path in sorted(zip(names, paths, strict=True), key=lambda x: x[1]):
        tag = None
        for n in range(1, len(path) + 1):
            # compare the joined strings, since names may themselves contain ">"
            cand = ">".join(path[-n:])
            if cand not in used and sum(">".join(p[-n:]) == cand for p in paths) == 1:
                tag = cand
                break
        if tag is None:
            tag = ">".join(path)
            i = 2
            while f"{tag}#{i}" in used or tag in used:
                tag = f"{'>'.join(path)}#{i}"
                i += 1
        used.add(tag)
        tags[tuple(orig)] = tag
    return tags


def server_classes_to_aw_research(
    classes: list[tuple[list[str], dict]],
) -> list[tuple[str, str, str | None]]:
    """Convert aw-server categories to aw_research's (regex, tag, parent_tag) tuples.

    Missing parents are added, `ignore_case` becomes an inline `(?i)` flag, and
    categories without a regex get a never-matching one so they still register as
    parents.
    """
    rules = {tuple(name): rule for name, rule in classes}
    for name in list(rules):
        for n in range(1, len(name)):
            rules.setdefault(name[:n], {"type": "none"})

    tags = _category_tags([list(name) for name in rules])
    result: list[tuple[str, str, str | None]] = []
    for name, rule in rules.items():
        if name == ("Uncategorized",):
            continue
        regex = rule.get("regex") if rule.get("type") == "regex" else None
        if regex and rule.get("ignore_case"):
            regex = f"(?i){regex}"
        parent = tags[name[:-1]] if len(name) > 1 else None
        result.append((regex or _NEVER_MATCH, tags[name], parent))
    return result


def load_server_classes(testing: bool = False) -> list[tuple[list[str], dict]]:
    """Fetch the category rules configured in aw-server (the same ones the web UI uses)."""
    awc = _get_aw_client(testing)
    try:
        classes = awc.get_setting("classes")
    except Exception as e:
        classes = None
        logger.warning(f"Failed to get categories from aw-server: {e}")
    if not classes:
        from aw_client.classes import default_classes

        logger.warning("No categories set in aw-server, using the default categories")
        return default_classes
    return [(c["name"], c["rule"]) for c in classes]


def load_category_rules(personal: bool = True) -> CategoryRules:
    """Load the category rules selected by `[data] categories` in the config.

    "server" (or unset) uses the categories configured in aw-server, anything
    else is a path to a categories file in aw_research's TOML or CSV format
    (relative paths are relative to the config file).
    """
    config = load_config(use_example=not personal)
    categories = config["data"].get("categories", SERVER_CATEGORIES)
    if categories == SERVER_CATEGORIES:
        classes = server_classes_to_aw_research(load_server_classes(not personal))
        return {"new_classes": classes}
    categories_path = Path(categories).expanduser()
    if not categories_path.is_absolute():
        categories_path = (
            _get_config_path(use_example=not personal).parent / categories_path
        )
    return {"filename": str(categories_path)}


def _category_rules_fingerprint(rules: CategoryRules) -> str:
    """Hash of the effective category rules (a file's contents, not just its path)."""
    if "filename" in rules:
        path = Path(rules["filename"])
        content = path.read_bytes() if path.exists() else b""
        blob = rules["filename"].encode() + b"\0" + content
    else:
        blob = json.dumps(rules["new_classes"]).encode()
    return hashlib.sha256(blob).hexdigest()


def classify(
    events: list[Event], personal: bool, rules: CategoryRules | None = None
) -> list[Event]:
    """Categorize events, using the aw-server category rules by default.

    See :func:`load_category_rules` for how the rules are selected.
    """
    if rules is None:
        rules = load_category_rules(personal)
    aw_research.classify._init_classes(**rules)
    return aw_research.classify.classify(events)


def load_category_df(events: list[Event]) -> pd.DataFrame:
    tss = {}
    all_categories = list({t for e in events for t in e.data["$tags"]})
    events_by_date = defaultdict(list)
    for e in events:
        events_by_date[e.timestamp.date()].append(e)
    for cat in all_categories:
        try:
            tss[cat] = categorytime_per_day(events, cat)
        except Exception as e:
            if "No events to calculate on" not in str(e):
                raise
    df = pd.DataFrame(tss)
    df = df.replace(np.nan, 0)
    df["All_cols"] = df.sum(axis=1)
    # df.index is a DatetimeIndex (from resample), so convert each Timestamp to a
    # date for the lookup: a pd.Timestamp never equals a datetime.date key.
    df["All_events"] = [
        sum((e.duration for e in events_by_date.get(d.date(), [])), start=timedelta(0))
        for d in df.index
    ]
    return df


@click.command()
@click.option("--csv", is_flag=True, help="Print as CSV")
def screentime(csv: bool):
    """Loads screentime data, and prints total duration."""
    events = load_screentime(
        since=datetime.now(tz=timezone.utc) - timedelta(days=90),
        datasources=["activitywatch"],
        personal=True,
    )
    logger.info(f"Total duration: {sum((e.duration for e in events), timedelta(0))}")

    df = load_category_df(events)
    if csv:
        print(df.to_csv())
    else:
        print(df)


if __name__ == "__main__":
    screentime()
