"""
Loads Whoop data

Requires the full GDPR-compliant export (which includes granular HR data), which is available at: https://privacy.whoop.com/
"""

from datetime import timedelta
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


def load_sleep_df() -> pd.DataFrame:
    whoop_export_dir = load_config()["data"]["whoop"]
    filename = Path(whoop_export_dir) / "Health" / "sleeps.csv"
    df = pd.read_csv(filename.expanduser(), parse_dates=True)

    # df columns are: "created_at","updated_at","activity_id","score","quality_duration","latency","max_heart_rate","average_heart_rate","debt_pre","debt_post","need_from_strain","sleep_need","habitual_sleep_need","disturbances","time_in_bed","light_sleep_duration","slow_wave_sleep_duration","rem_sleep_duration","cycles_count","wake_duration","arousal_time","no_data_duration","in_sleep_efficiency","credit_from_naps","hr_baseline","respiratory_rate","sleep_consistency","algo_version","projected_score","projected_sleep","optimal_sleep_times","kilojoules","user_id","during","timezone_offset","survey_response_id","percent_recorded","auto_detected","state","responded","team_act_id","source","is_significant","is_normal","is_nap"
    # we are interested in the "during" column, which is a JSON string of a 2-tuple with isoformat timestamps
    def parse_during(x):
        try:
            return eval(x.replace(")", "]"))
        except:
            print(x)
            raise

    df["start"] = pd.to_datetime(df["during"].apply(lambda x: parse_during(x)[0]))
    df["end"] = pd.to_datetime(df["during"].apply(lambda x: parse_during(x)[1]))
    df["duration"] = df["end"] - df["start"]

    # keep only the columns we want
    df = df[["start", "end", "duration", "score"]]

    # set index and sort
    offset = timedelta(hours=8)
    df = df.set_index(pd.to_datetime(pd.DatetimeIndex(df["start"] - offset).date, utc=True))  # type: ignore
    df = df.sort_index()

    # rename index to timestamp
    df.index.name = "timestamp"

    return df
