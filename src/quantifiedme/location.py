import json
import glob
from typing import Tuple, Dict
from datetime import datetime
from pathlib import Path

import click
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import joblib

from .config import load_config

memory = joblib.Memory(".cache/joblib")


def location_history_to_df(fn) -> pd.DataFrame:
    with open(fn, "r") as f:
        locs = json.load(f)["locations"]
    for loc in locs:
        loc["lat"] = loc.pop("latitudeE7") / 10_000_000
        loc["long"] = loc.pop("longitudeE7") / 10_000_000
        loc["timestamp"] = pd.Timestamp(int(loc.pop("timestampMs")), unit="ms")
        for p in ["velocity", "verticalAccuracy", "altitude", "activity", "heading"]:
            if p in loc:
                loc.pop(p)
    df = pd.DataFrame(locs)
    # no idea why this is typed as None, needing the type-ignore on next lint
    df = df.set_index("timestamp")
    # Remove duplicates in index
    df = df[~df.index.duplicated(keep="first")]  # type: ignore
    df = df.resample("10Min").ffill()
    return df


@memory.cache
def load_all_dfs() -> Dict[str, pd.DataFrame]:
    dfs = {}
    path = load_config()["data"]["location"]
    for filepath in glob.glob(path + "/*.json"):
        name = Path(filepath).name.replace(".json", "")
        df = location_history_to_df(filepath)
        dfs[name] = df
    return dfs


def colocate(df_person1, df_person2, verbose=False):
    df = df_person1.join(df_person2, lsuffix="_a", rsuffix="_b")
    df["dist"] = (
        (df["lat_a"] - df["lat_b"]) ** 2 + (df["long_a"] - df["long_b"]) ** 2
    ).pow(1 / 2)

    df_close = df[df["dist"] < 0.01].copy()
    df_close["duration"] = df.index.freq.nanos / 3600e9  # Normalize to hours
    df_close = df_close.resample("24H").apply({"duration": np.sum})
    df_close = df_close["duration"]
    # print(df_close)
    if verbose:
        print(df_close)

    return df_close


def _proximity_to_location(
    df: pd.DataFrame, loc: Tuple[float, float], threshold_radius=0.001, verbose=False
) -> pd.Series:
    lat, lon = loc
    dist = ((df["lat"] - lat) ** 2 + (df["long"] - lon) ** 2) ** 0.5
    dist = dist[dist < threshold_radius]
    dist = pd.DataFrame(dist, columns=["dist"])
    dist["duration"] = 10 / 60
    dist = dist.resample("24H").apply({"duration": np.sum})
    if verbose:
        print(dist)
    return dist["duration"]


def plot_df_duration(df, title, save: str = None) -> None:
    # print('Plotting...')
    ax = df.plot.area(label=f"{title}", legend=True)
    ax = df.rolling(7, min_periods=2).mean().plot(label=f"{title} 7d SMA", legend=True)
    ax = (
        df.rolling(30, min_periods=2).mean().plot(label=f"{title} 30d SMA", legend=True)
    )
    # df.ewm(7).mean().plot(ax=ax, label='7d EMA')
    # df.ewm(30).mean().plot(ax=ax, label='30d EMA')
    ax.set_ylabel("Hours")
    ax.set_xlabel("")
    plt.ylim(0)

    if save:
        plt.savefig(save, bbox_inches="tight")
    else:
        plt.show()


def main_plot(dfs, me, other, start=None, save=None):
    coords = load_config()["locations"]

    df = dfs[me]
    if start:
        df = df[start < df.index]

    if other in coords:
        loc = coords[other]
        df = _proximity_to_location(
            df, (loc["lat"], loc["long"]), threshold_radius=loc["accuracy"]
        )
    else:
        # df = colocate(dfs[me], dfs[args.other], start=args.start)
        df_other = dfs[other]
        if start:
            df_other = df_other[start < df_other.index]
        df = colocate(df, df_other)

    plot_df_duration(df, other, save)


@click.command()
@click.argument("name")
@click.option("--start", default=None, type=click.DateTime(), help="query from date")
@click.option("--save", is_flag=True)
def locate(name: str, start: datetime, save: bool):
    # TODO: Pick up from config file
    me = "erik"

    dfs = load_all_dfs()
    df = dfs[me]

    if start:
        df = df[start < df.index]

    main_plot(dfs, me, name)


if __name__ == "__main__":
    locate()
