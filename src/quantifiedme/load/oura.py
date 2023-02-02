import json
from datetime import timedelta
from pathlib import Path

import click
import pandas as pd
import iso8601
import matplotlib.pyplot as plt

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
    df["summary_date"] = pd.to_datetime(df["summary_date"])
    df = df.set_index("summary_date")
    return df


def load_readiness_df() -> pd.DataFrame:
    data = load_data()
    df = pd.DataFrame(data["readiness"])
    df["summary_date"] = pd.to_datetime(df["summary_date"])
    df = df.set_index("summary_date")
    return df


def load_activity_df() -> pd.DataFrame:
    data = load_data()
    df = pd.DataFrame(data["activity"])
    df["summary_date"] = pd.to_datetime(df["summary_date"])
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
    df = df.rename(columns={"bpm": "hr"})

    return df


@click.command()
def oura():
    """TODO, just loads all data"""
    load_sleep_df()
    load_activity_df()
    load_readiness_df()


if __name__ == "__main__":
    # oura()
    load_heartrate_df().plot(kind="line", y="bpm", figsize=(20, 10))
    plt.show()
