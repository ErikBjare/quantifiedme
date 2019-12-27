import json
import glob
import argparse
from typing import Tuple
from datetime import datetime

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import joblib

memory = joblib.Memory('.cache/joblib')


def location_history_to_df(fn):
    with open(fn, 'r') as f:
        locs = json.load(f)['locations']
    for loc in locs:
        loc['lat'] = loc.pop('latitudeE7') / 10_000_000
        loc['long'] = loc.pop('longitudeE7') / 10_000_000
        loc['timestamp'] = pd.Timestamp(int(loc.pop('timestampMs')), unit='ms')
        for p in ['velocity', 'verticalAccuracy', 'altitude', 'activity', 'heading']:
            if p in loc:
                loc.pop(p)
    df = pd.DataFrame(locs).set_index('timestamp')
    df = df.resample('10Min').ffill()
    return df


@memory.cache
def load_all_dfs():
    dfs = {}
    for fn in glob.glob("data/location/*.json"):
        name = fn.replace("data/location/", '').replace(".json", '')
        df = location_history_to_df(fn)
        dfs[name] = df
    return dfs


def _datetime_arg(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('other')
    parser.add_argument('--start', type=_datetime_arg)
    parser.add_argument('--save')
    return parser.parse_args()


def colocate(df_person1, df_person2, verbose=False):
    df = df_person1.join(df_person2, lsuffix='_a', rsuffix='_b')
    df['dist'] = ((df['lat_a'] - df['lat_b'])**2 + (df['long_a'] - df['long_b'])**2).pow(1 / 2)

    df_close = df[df['dist'] < 0.01].copy()
    df_close['duration'] = df.index.freq.nanos / 3600e9  # Normalize to hours
    df_close = df_close.resample('24H').apply({'duration': np.sum})
    df_close = df_close['duration']
    # print(df_close)
    if verbose:
        print(df_close)

    return df_close


def _proximity_to_location(df: pd.DataFrame, loc: Tuple[float, float], threshold_radius=0.001, verbose=False):
    lat, lon = loc
    dist = ((df['lat'] - lat)**2 + (df['long'] - lon)**2) ** .5
    dist = dist[dist < threshold_radius]
    dist = pd.DataFrame(dist, columns=['dist'])
    dist['duration'] = 10 / 60
    dist = dist.resample('24H').apply({'duration': np.sum})
    if verbose:
        print(dist)
    return dist['duration']


def plot_df_duration(df, title, save: str = None):
    # print('Plotting...')
    ax = df.plot.area(label=f'{title}', legend=True)
    ax = df.rolling(7, min_periods=2).mean().plot(label=f'{title} 7d SMA', legend=True)
    ax = df.rolling(30, min_periods=2).mean().plot(label=f'{title} 30d SMA', legend=True)
    # df.ewm(7).mean().plot(ax=ax, label='7d EMA')
    # df.ewm(30).mean().plot(ax=ax, label='30d EMA')
    ax.set_ylabel('Hours')
    ax.set_xlabel('')
    plt.ylim(0)

    if save:
        plt.savefig(save, bbox_inches='tight')
    else:
        plt.show()


def main_plot(dfs, me, other, start=None, save=None):
    coords = {
        "kamnars": (55.722030, 13.217581, 0.001),
        "baravagen": (55.717222, 13.194150, 0.001),
        "actic": (55.722379, 13.213125, 0.001),
        "victoria": (55.7189, 13.2008, 0.001),
        "lund": (55.705377, 13.192200, 0.05),
        "lth": (55.711264, 13.209850, 0.005),
    }

    df = dfs[me]
    if start:
        df = df[start < df.index]

    if other in coords:
        df = _proximity_to_location(df, coords[other][:2], threshold_radius=coords[other][2])
    else:
        # df = colocate(dfs[me], dfs[args.other], start=args.start)
        df_other = dfs[other]
        if start:
            df_other = df_other[start < df_other.index]
        df = colocate(df, df_other)

    plot_df_duration(df, other, save)


def main():
    args = _parse_args()
    me = "erik"

    dfs = load_all_dfs()
    # print(dfs[me])
    df = dfs[me]

    if args.start:
        df = df[args.start < df.index]

    main_plot(dfs, me, args.other)


if __name__ == "__main__":
    main()
