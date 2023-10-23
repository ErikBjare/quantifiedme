import multiprocessing
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ..cache import memory
from ..config import load_config


def load_sleep_df() -> pd.DataFrame:
    filepath = load_config()["data"]["fitbit"]
    filepath = Path(filepath).expanduser()
    assert filepath.exists()

    # filepath is the root folder of an unzipped Fitbit export
    # sleep data is in `Global Export Data/sleep-YYYY-MM-DD.json`
    # we need to combine all of these files into a single dataframe
    files = filepath.glob("Global Export Data/sleep-*.json")

    # load each file into a dataframe
    dfs = []
    for f in sorted(files):
        dfs.append(_load_sleep_file(f))

    # combine all the dataframes into a single dataframe
    df = pd.concat(dfs)

    return df


def _load_sleep_file(filepath):
    with open(filepath) as f:
        df = pd.read_json(f)
    df = df[["dateOfSleep", "startTime", "endTime", "duration", "efficiency"]]
    df = df.rename(
        columns={
            "dateOfSleep": "date",
            "startTime": "start",
            "endTime": "end",
            "efficiency": "score",
        }
    )
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["start"] = pd.to_datetime(df["start"], utc=True)
    df["end"] = pd.to_datetime(df["end"], utc=True)
    df["duration"] = pd.to_timedelta(df["duration"] / 1000, unit="s")
    df = df.set_index("date")
    return df


def _load_heartrate_file(filepath):
    # print(f"Loading {filepath}...")
    # json format is {"dateTime": "2020-01-01", "value": {"bpm": 60, "confidence": 0}}
    df = pd.read_json(filepath)
    df["timestamp"] = pd.to_datetime(df["dateTime"], utc=True)
    df["hr"] = df["value"].apply(lambda x: x["bpm"])
    df = df.set_index("timestamp")
    df = df[["hr"]]
    return df


@memory.cache
def load_heartrate_df() -> pd.DataFrame:
    # load heartrate data from Fitbit export
    filepath = load_config()["data"]["fitbit"]
    filepath = Path(filepath).expanduser()

    # filepath is the root folder of an unzipped Fitbit export
    # heartrate data is split into daily files in `Global Export Data/heart_rate-YYYY-MM-DD.json`
    # we need to combine all of these files into a single dataframe

    # get all the files in the folder
    files = filepath.glob("Global Export Data/heart_rate-*.json")

    # load each file into a dataframe
    # parallelize to speed up the process
    pool = multiprocessing.Pool(20)
    dfs = pool.map(_load_heartrate_file, sorted(files))

    # combine all the dataframes into a single dataframe
    df = pd.concat(dfs)

    # rename column bpm to hr
    df = df.rename(columns={"bpm": "hr"})

    return df


if __name__ == "__main__":
    df = load_heartrate_df()
    print(df)
    df.plot()

    plt.show()
