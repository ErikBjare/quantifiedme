"""Tests for the screentime event cache (keying and reuse)."""

import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from aw_core import Event

from quantifiedme.derived import screentime


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Isolated cache dir, config, host discovery, loader and classifier."""
    state: dict[str, Any] = {
        "config": {
            "data": {
                "activitywatch": {
                    "port": 5600,
                    "hostnames": None,
                    "exclude_hostnames": [],
                    "include_mobile": True,
                }
            }
        },
        "hosts": [
            ("desktop", "laptop", "w-laptop", "a-laptop", ()),
            ("android", "phone", "a-phone"),
        ],
        "rules": {"new_classes": [("GitHub", "Programming", None)]},
        "loads": 0,
    }

    def load_config(use_example=False):
        return state["config"]

    def load_raw(since=None, **kwargs):
        state["loads"] += 1
        now = datetime.now(tz=timezone.utc)
        return [
            Event(
                timestamp=now - timedelta(days=d, hours=1),
                duration=timedelta(minutes=1),
                data={"title": "GitHub"},
            )
            for d in range(60)
            if now - timedelta(days=d, hours=1) >= since
        ]

    def classify(events, personal, rules=None):
        tags = {
            regex: tag for regex, tag, _ in (rules or state["rules"])["new_classes"]
        }
        for e in events:
            e.data["$tags"] = {tags.get(e.data["title"], "Uncategorized")}
        return events

    monkeypatch.setattr(screentime, "cache_dir", tmp_path)
    monkeypatch.setattr(screentime, "load_config", load_config)
    monkeypatch.setattr(screentime, "_get_aw_client", lambda testing: object())
    monkeypatch.setattr(
        screentime, "_discover_aw_hosts", lambda awc, config, hn: state["hosts"]
    )
    monkeypatch.setattr(screentime, "_load_screentime_raw", load_raw)
    monkeypatch.setattr(screentime, "classify", classify)
    monkeypatch.setattr(screentime, "load_category_rules", lambda p: state["rules"])
    state["dir"] = tmp_path
    return state


def _since(days: int) -> datetime:
    return datetime.now(tz=timezone.utc) - timedelta(days=days)


def test_unchanged_config_reuses_cache(env):
    key = screentime.screentime_cache_key()
    assert screentime.screentime_cache_key() == key
    screentime.load_screentime_cached(since=_since(30))
    screentime.load_screentime_cached(since=_since(30))
    assert env["loads"] == 1


@pytest.mark.parametrize(
    "setting,value",
    [
        ("hostnames", ["laptop"]),
        ("exclude_hostnames", ["phone"]),
        ("include_mobile", False),
        ("port", 5666),
    ],
)
def test_host_settings_change_key(env, setting, value):
    key = screentime.screentime_cache_key()
    env["config"]["data"]["activitywatch"][setting] = value
    assert screentime.screentime_cache_key() != key


def test_hostnames_argument_changes_key(env):
    assert screentime.screentime_cache_key() != screentime.screentime_cache_key(
        hostnames=["laptop"]
    )


def test_new_discovered_host_reloads_but_host_order_does_not(env):
    screentime.load_screentime_cached(since=_since(30))
    env["hosts"] = list(reversed(env["hosts"]))
    screentime.load_screentime_cached(since=_since(30))
    assert env["loads"] == 1
    env["hosts"] = env["hosts"] + [("desktop", "new", "w-new", "a-new", ())]
    screentime.load_screentime_cached(since=_since(30))
    assert env["loads"] == 2


def test_fresh_cache_used_when_server_unreachable(env, monkeypatch):
    screentime.load_screentime_cached(since=_since(30))

    def unreachable(awc, config, hn):
        raise ConnectionError

    monkeypatch.setattr(screentime, "_discover_aw_hosts", unreachable)
    screentime.load_screentime_cached(since=_since(30))
    assert env["loads"] == 1


def test_cache_written_without_hosts_is_not_reused_once_known(env, monkeypatch):
    def unreachable(awc, config, hn):
        raise ConnectionError

    with monkeypatch.context() as m:
        m.setattr(screentime, "_discover_aw_hosts", unreachable)
        screentime.load_screentime_cached(since=_since(30))
    screentime.load_screentime_cached(since=_since(30))
    assert env["loads"] == 2


def test_cache_version_changes_key(env, monkeypatch):
    key = screentime.screentime_cache_key()
    monkeypatch.setattr(screentime, "CACHE_VERSION", screentime.CACHE_VERSION + 1)
    assert screentime.screentime_cache_key() != key


def test_ruleset_change_changes_key_and_recategorizes(env):
    key = screentime.screentime_cache_key()
    events = screentime.load_screentime_cached(since=_since(30))
    assert events[0].data["$tags"] == {"Programming"}
    env["rules"] = {"new_classes": [("GitHub", "Work", None)]}
    assert screentime.screentime_cache_key() != key
    events = screentime.load_screentime_cached(since=_since(30))
    assert events[0].data["$tags"] == {"Work"}
    assert env["loads"] == 2


def test_rules_file_fingerprint_tracks_contents(tmp_path):
    path = tmp_path / "categories.toml"
    path.write_text('[categories]\nWork = ["GitHub"]\n')
    rules = {"filename": str(path)}
    fp = screentime._category_rules_fingerprint(rules)
    assert screentime._category_rules_fingerprint(rules) == fp
    path.write_text('[categories]\nWork = ["GitLab"]\n')
    assert screentime._category_rules_fingerprint(rules) != fp
    assert screentime._category_rules_fingerprint({"new_classes": []}) != fp


def test_host_change_reloads_and_removes_stale_files(env):
    screentime.load_screentime_cached(since=_since(30))
    (env["dir"] / "events.pickle").write_bytes(b"legacy")
    env["config"]["data"]["activitywatch"]["include_mobile"] = False
    screentime.load_screentime_cached(since=_since(30))
    assert env["loads"] == 2
    files = sorted(p.name for p in env["dir"].iterdir())
    assert files == [f"events-full-{screentime.screentime_cache_key()}.pickle"]


def test_since_not_covered_reloads(env):
    screentime.load_screentime_cached(since=_since(10))
    events = screentime.load_screentime_cached(since=_since(10))
    assert env["loads"] == 1
    assert all(e.timestamp >= _since(10) for e in events)
    # even slightly earlier than the cached start is not covered
    screentime.load_screentime_cached(since=_since(10) - timedelta(hours=1))
    assert env["loads"] == 2


def test_fast_is_served_from_full_cache(env):
    screentime.load_screentime_cached(since=_since(30))
    events = screentime.load_screentime_cached(since=_since(10), fast=True)
    assert env["loads"] == 1
    assert events and all(e.timestamp >= _since(10) for e in events)


def test_expired_cache_reloads(env):
    screentime.load_screentime_cached(since=_since(30))
    path = screentime._cache_file(False, screentime.screentime_cache_key())
    old = time.time() - 2 * 24 * 3600
    os.utime(path, (old, old))
    screentime.load_screentime_cached(since=_since(30))
    assert env["loads"] == 2
