from ..config import load_config

from pathlib import Path

import pandas as pd
import json


def _load_heartrate_file(filepath):
    print(f"Loading {filepath}...")
    with open(filepath) as f:
        data = json.load(f)
        df = pd.DataFrame(
            [(entry["dateTime"], entry["value"]["bpm"]) for entry in data],
            columns=["timestamp", "bpm"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
        return df


def load_heartrate_df() -> pd.DataFrame:
    # load heartrate data from Fitbit export
    filepath = load_config()["data"]["fitbit"]
    filepath = Path(filepath).expanduser()

    # filepath is the root folder of an unzipped Fitbit export
    # heartrate data is split into daily files in `Physical Activity/heart_rate-YYYY-MM-DD.json`
    # we need to combine all of these files into a single dataframe

    # get all the files in the folder
    files = filepath.glob("Physical Activity/heart_rate-*.json")

    # load each file into a dataframe
    # parallelize to speed up the process
    import multiprocessing

    pool = multiprocessing.Pool(20)
    dfs = pool.map(_load_heartrate_file, sorted(files))

    # combine all the dataframes into a single dataframe
    df = pd.concat(dfs)

    return df


if __name__ == "__main__":
    df = load_heartrate_df()
    print(df)
    df.plot()
    import matplotlib.pyplot as plt

    plt.show()
