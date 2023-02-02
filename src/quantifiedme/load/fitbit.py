from ..config import load_config

from pathlib import Path

import pandas as pd


def _load_heartrate_file(filepath):
    #print(f"Loading {filepath}...")
    # json format is {"dateTime": "2020-01-01", "value": {"bpm": 60, "confidence": 0}}
    df = pd.read_json(filepath)
    df["timestamp"] = pd.to_datetime(df["dateTime"], utc=True)
    df["hr"] = df["value"].apply(lambda x: x["bpm"])
    df = df.set_index("timestamp")
    df = df[["hr"]]
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

    # rename column bpm to hr
    df = df.rename(columns={"bpm": "hr"})

    return df


if __name__ == "__main__":
    df = load_heartrate_df()
    print(df)
    df.plot()
    import matplotlib.pyplot as plt

    plt.show()
