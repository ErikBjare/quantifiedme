# Code originally from now deprecated repo: https://github.com/ActivityWatch/aw-importer-smartertime

import csv
from datetime import datetime, timedelta, timezone
import secrets
import json
from pathlib import Path

from aw_core.models import Event
import aw_client
from aw_transform.union_no_overlap import union_no_overlap

from ..config import load_config


def load_events(since: datetime) -> list[Event]:
    # TODO: allow loading directly from export, so we don't need to manually convert to aw-bucket json
    events_smartertime: list[Event] = []
    # TODO: underspecified hostname priority
    for hostname, events in _load_smartertime_devices(since).items():
        events_smartertime = union_no_overlap(events_smartertime, events)
    return events_smartertime


def _load_smartertime_devices(since: datetime) -> dict[str, list]:
    """Loads smartertime data from all devices/files specified in config"""

    result = {}
    for hostname, smartertime_awbucket_path in load_config()["data"][
        "smartertime_buckets"
    ].items():
        events = _load_smartertime_events(
            since, filepath=str(Path(smartertime_awbucket_path).expanduser())
        )
        for e in events:
            e.data["$source"] = "smartertime"
            e.data["$hostname"] = hostname
        result[hostname] = events
    return result


def _load_smartertime_events(since: datetime, filepath) -> list[Event]:
    """Loads smartertime data from a single json file (generated below)"""

    print(f"Loading smartertime data from {filepath}")
    with open(filepath) as f:
        data = json.load(f)
        events = [Event(**e) for e in data["events"]]

    # Filter out events before `since`
    events = [e for e in events if since.astimezone(timezone.utc) < e.timestamp]

    # Filter out no-events and non-phone events
    events = [
        e for e in events if any(s in e.data["activity"] for s in ["phone:", "call:"])
    ]

    # Normalize to window-bucket data schema
    for e in events:
        e.data["app"] = e.data["activity"]
        e.data["title"] = e.data["app"]

    return events


def read_csv_to_events(filepath):
    events = []
    with open(filepath, "r") as f:
        c = csv.DictReader(f)
        for r in c:
            # print(r)
            dt = datetime.fromtimestamp(float(r["Timestamp UTC ms"]) / 1000)
            tz_h, tz_m = map(int, r["Time"].split("GMT+")[1].split()[0].split(":"))
            dt = dt.replace(tzinfo=timezone(timedelta(hours=tz_h, minutes=tz_m)))
            td = timedelta(milliseconds=float(r["Duration ms"]))
            e = Event(
                timestamp=dt,
                duration=td,
                data={
                    "activity": r["Activity"],
                    "device": r["Device"],
                    "place": r["Place"],
                    "room": r["Room"],
                },
            )
            events.append(e)
    return events


def read_csv_to_bucket(filepath):
    events = read_csv_to_events(filepath)
    end = max(e.timestamp + e.duration for e in events)
    bucket = {
        "id": f"smartertime_export_{end.date()}_{secrets.token_hex(4)}",
        "created": datetime.now(),
        "event_type": "smartertime.v0",
        "client": "",
        "hostname": "",
        "data": {"readonly": True},
        "events": events,
    }
    return bucket


def save_bucket(bucket):
    filename = bucket["id"] + ".awbucket.json"
    with open(filename, "w") as f:
        json.dump(bucket, f, indent=True, default=default)
    print(f"Saved as {filename}")


def import_to_awserver(bucket):
    awc = aw_client.ActivityWatchClient("smartertime2activitywatch", testing=True)
    buckets = json.loads(json.dumps({"buckets": [bucket]}, default=default))
    awc._post("import", buckets)


def default(o):
    if hasattr(o, "isoformat"):
        return o.isoformat()
    elif hasattr(o, "total_seconds"):
        return o.total_seconds()
    else:
        raise NotImplementedError


def convert_csv_to_awbucket(filepath):
    bucket = read_csv_to_bucket(filepath)
    save_bucket(bucket)
    # import_to_awserver(bucket)


def test_load_smartertime_events():
    events = _load_smartertime_events(
        datetime(2020, 1, 1),
        filepath=Path(
            "~/Programming/quantifiedme/data/smartertime/smartertime_export_erb-f3_2022-02-01_efa36e6a.awbucket.json"
        ).expanduser(),
    )
    assert len(events) > 0


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "convert":
        assert len(sys.argv) > 2
        filename = sys.argv.pop()
        convert_csv_to_awbucket(filename)
    else:
        test_load_smartertime_events()
