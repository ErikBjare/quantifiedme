import json
import glob

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
    df = df.resample('1H').ffill()
    return df


@memory.cache
def load_all_dfs():
    dfs = []
    for fn in glob.glob("data/location/*.json"):
        df = location_history_to_df(fn)
        dfs.append(df)
    return dfs


def main():
    dfs = load_all_dfs()
    df = dfs[0].join(dfs[1], lsuffix='_a', rsuffix='_b')
    df['dist'] = ((df['lat_a'] - df['lat_b'])**2 + (df['long_a'] - df['long_b'])**2).pow(1 / 2)

    df_close = df[df['dist'] < 0.01].copy()
    df_close['duration'] = 1   # 1 hour
    df_close = df_close.resample('1D').apply({'duration': np.sum})
    df_close = df_close
    # print(df_close)

    ax = df_close.plot()
    df_close.rolling(7, min_periods=2).mean().plot(ax=ax)
    df_close.rolling(30, min_periods=2).mean().plot(ax=ax)
    # df_close.ewm(7).mean().plot(ax=ax, label='7d EMA')
    # df_close.ewm(30).mean().plot(ax=ax, label='30d EMA')
    plt.tight_layout()
    plt.ylim(0)
    plt.show()


if __name__ == "__main__":
    main()
