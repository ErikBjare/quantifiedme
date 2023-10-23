import json
from datetime import timedelta
from pathlib import Path

import click
import iso8601
import matplotlib.pyplot as plt
import pandas as pd

from ..config import load_config


def load_data():
    filepath = load_config()["data"]["oura"]
    filepath = Path(filepath).expanduser()
    with open(filepath) as f:
        data = json.load(f)
    return data


def load_sleep_df() -> pd.DataFrame:
    data = load_data()
    df = pd.DataFrame(data["sleep"])
    # summary_date is the "start" date
    # https://cloud.ouraring.com/docs/sleep
    # NOTE: Not sure why I have to subtract a day here, shouldn't be necessary according to docs,
    #       but necessary for it to be correct.
    df["summary_date"] = pd.to_datetime(df["summary_date"], utc=True) - timedelta(
        days=1
    )
    df["bedtime_start"] = pd.to_datetime(df["bedtime_start"], utc=True)
    df["bedtime_end"] = pd.to_datetime(df["bedtime_end"], utc=True)
    df = df.rename(
        columns={
            "summary_date": "timestamp",
            "bedtime_start": "start",
            "bedtime_end": "end",
        }
    )
    df = df.set_index("timestamp")
    df["duration"] = df["end"] - df["start"]
    return df[["start", "end", "duration", "score"]]  # type: ignore


def load_readiness_df() -> pd.DataFrame:
    data = load_data()
    df = pd.DataFrame(data["readiness"])
    df["summary_date"] = pd.to_datetime(df["summary_date"], utc=True)
    df = df.set_index("summary_date")
    return df


def load_activity_df() -> pd.DataFrame:
    data = load_data()
    df = pd.DataFrame(data["activity"])
    df["summary_date"] = pd.to_datetime(df["summary_date"], utc=True)
    df = df.set_index("summary_date")
    return df


def load_heartrate_df() -> pd.DataFrame:
    # new data format
    filepath = load_config()["data"]["oura-heartrate"]
    filepath = Path(filepath).expanduser()
    with open(filepath) as f:
        raw = json.load(f)
        data_heartrate = [
            (iso8601.parse_date(entry["timestamp"]), entry["bpm"])
            for entry in raw["heart_rate"]
        ]
        df = pd.DataFrame(data_heartrate, columns=["timestamp", "bpm"])

    filepath = load_config()["data"]["oura-sleep"]
    filepath = Path(filepath).expanduser()
    with open(filepath) as f:
        raw = json.load(f)
        nights_hr = [
            (
                entry["bedtime_start"],
                entry["heart_rate"]["items"],
                entry["heart_rate"]["interval"],
            )
            for entry in raw["sleep"]
        ]
        data_heartrate_sleep = [
            (iso8601.parse_date(start) + i * timedelta(seconds=interval), bpm)
            for (start, bpms, interval) in nights_hr
            for (i, bpm) in enumerate(bpms)
        ]
        df_heartrate_sleep = pd.DataFrame(
            data_heartrate_sleep, columns=["timestamp", "bpm"]
        )

    df = df.combine_first(df_heartrate_sleep)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")
    # drop zeros
    df = df[df["bpm"] > 0]

    # rename bpm to hr
    df = df.rename(columns={"bpm": "hr"})  # type: ignore

    return df


@click.command()
def oura():
    """TODO, just loads all data"""
    sleep = load_sleep_df()
    activity = load_activity_df()
    readiness = load_readiness_df()
    print("Entries")
    print(f"  Sleep: {len(sleep)}")
    print(f"  Activity: {len(activity)}")
    print(f"  Readiness: {len(readiness)}")


if __name__ == "__main__":
    # oura()
    load_heartrate_df().plot(kind="line", y="bpm", figsize=(20, 10))
    plt.show()
