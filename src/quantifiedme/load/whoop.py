"""
Loads Whoop data

Requires the full GDPR-compliant export (which includes granular HR data), which is available at: https://privacy.whoop.com/
"""

from pathlib import Path

import pandas as pd

from ..config import load_config


def load_heartrate_df() -> pd.DataFrame:
    whoop_export_dir = load_config()["data"]["whoop"]
    dfs = []
    for file in (Path(whoop_export_dir) / "Health").expanduser().iterdir():
        if file.name.startswith("metrics") and file.name.endswith(".csv"):
            df = pd.read_csv(Path(file).expanduser(), parse_dates=True)
            dfs.append(df)

    # df columns are: hr, accel_x, accel_y, accel_z, skin_temp, ts
    # combine dfs together
    df = pd.concat(dfs, ignore_index=True)
    df = df.set_index(pd.DatetimeIndex(df["ts"]))
    del df["ts"]

    # drop non-HR columns
    df = df[["hr"]]

    # sort
    df = df.sort_index()
    
    # rename index to timestamp
    df.index.name = "timestamp"

    return df


def test_load_whoop():
    df = load_heartrate_df()
    print(df.head())


if __name__ == "__main__":
    test_load_whoop()