"""
Aggregates sleep data from Fitbit, Oura, and Whoop into a single dataframe.
"""

from datetime import datetime, timedelta, timezone

import pandas as pd

from ..load.fitbit import load_sleep_df as load_fitbit_sleep_df
from ..load.oura import load_sleep_df as load_oura_sleep_df
from ..load.whoop import load_sleep_df as load_whoop_sleep_df


def load_sleep_df(ignore: list[str] = []) -> pd.DataFrame:
    """
    Loads sleep data from Fitbit, Oura, and Whoop into a single dataframe.
    """
    df = pd.DataFrame()

    # Fitbit
    #df = join(df, load_fitbit_sleep_df(), rsuffix="_fitbit")

    # Oura
    if "oura" not in ignore:
        df_oura = load_oura_sleep_df()
        df = join(df, df_oura.add_suffix("_oura"))

    # Whoop
    if "whoop" not in ignore:
        df_whoop = load_whoop_sleep_df()
        df = join(df, df_whoop.add_suffix("_whoop"))

    # perform some aggregations
    keys = list(set(col.split("_")[0] for col in df.columns) & {"duration", "score"})
    for key in keys:
        subkeys = df.columns[df.columns.str.startswith(key)]
        df[key] = df[subkeys].mean(axis=1)
    df = df[keys]

    return df


def join(df_target, df_source, **kwargs) -> pd.DataFrame:
    if df_target.empty:
        return df_source
    else:
        return df_target.join(df_source, **kwargs)


if __name__ == "__main__":
    df = load_sleep_df()
    print(df)
    """
    df["duration_whoop"].plot()
    import matplotlib.pyplot as plt

    plt.show()
    """