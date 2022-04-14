"""
Loads ActivityWatch data into InfluxDB, for visualization with Grafana and such.
"""

import os

from aw_core import Event
from aw_client import ActivityWatchClient

import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS

org = os.environ.get("INFLUX_ORG", "Personal")
assert "INFLUX_TOKEN" in os.environ
token = os.environ["INFLUX_TOKEN"]

# Store the URL of your InfluxDB instance
url = "http://localhost:8086"
client = influxdb_client.InfluxDBClient(url=url, token=token, org=org)

hostname = "erb-main2-arch"  # TODO: Make an option
bucket_name_influx = os.environ.get("INFLUX_BUCKET", "activitywatch")


def copy_bucket_to_influxdb(aw: ActivityWatchClient, bucket_id: str):
    print(f"Copying bucket {bucket_id} to InfluxDB")
    print("Getting events...")
    events = aw.get_events(bucket_id)
    print("Sending events...")
    send_events_to_influxdb(events, bucket_id)


def init_influxdb():
    print("Creating bucket")
    bucket_api = client.buckets_api()
    if existing_bucket := bucket_api.find_bucket_by_name(bucket_name_influx):
        print("Bucket already existed, replacing")
        bucket_api.delete_bucket(existing_bucket)
    bucket_api.create_bucket(bucket_name=bucket_name_influx)


def awevent_to_influx(e: Event, bucket_id: str) -> dict:
    return influxdb_client.Point.from_dict(
        {
            "measurement": bucket_id,
            "tags": {},
            "fields": {"duration": e.duration.total_seconds()} | e.data,
            "time": e.timestamp,
        }
    )


def send_events_to_influxdb(events: list[Event], bucket_id_aw: str):
    print("Converting events...")
    points = [awevent_to_influx(e, bucket_id_aw) for e in events]

    print("Writing events...")
    write_api = client.write_api(write_options=SYNCHRONOUS)
    batch_size = 10_000
    for i in range(0, len(points), batch_size):
        write_api.write(
            bucket=bucket_name_influx, org=org, record=points[i : i + batch_size]
        )


if __name__ == "__main__":
    init_influxdb()
    aw = ActivityWatchClient(testing=True)
    copy_bucket_to_influxdb(aw, f"aw-watcher-afk_{hostname}")
    copy_bucket_to_influxdb(aw, f"aw-watcher-window_{hostname}")
