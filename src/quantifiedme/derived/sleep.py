"""
Aggregates sleep data from Fitbit, Oura, and Whoop into a single dataframe.
"""

import logging

import click
import matplotlib.pyplot as plt
import pandas as pd

from ..load.fitbit import load_sleep_df as load_fitbit_sleep_df
from ..load.oura import load_sleep_df as load_oura_sleep_df
from ..load.whoop import load_sleep_df as load_whoop_sleep_df

logger = logging.getLogger(__name__)


def _check_index(df: pd.DataFrame, src=None) -> bool:
    # check for index duplicates
    duplicates = df.index.duplicated().sum()
    if duplicates:
        logger.warning(
            f"Found {duplicates} duplicate index entries"
            + (f" in {src}" if src else "")
        )
    return bool(duplicates == 0)


def _merge_several_sleep_records_per_day(df: pd.DataFrame) -> pd.DataFrame:
    duplicates = df.index.duplicated().sum()
    if duplicates:
        df = df.groupby(df.index).agg({"start": "min", "end": "max", "score": "max"})  # type: ignore
        df["duration"] = df["end"] - df["start"]
        logger.warning(f"Merged {duplicates} duplicate sleep entries")
    return df


def load_sleep_df(ignore: list[str] = [], aggregate=True) -> pd.DataFrame:
    """
    Loads sleep data from Fitbit, Oura, and Whoop into a single dataframe.
    """
    df: pd.DataFrame = pd.DataFrame()

    # Fitbit
    if "fitbit" not in ignore:
        df_fitbit = load_fitbit_sleep_df()
        df_fitbit = _merge_several_sleep_records_per_day(df_fitbit)
        df = join(df, df_fitbit.add_suffix("_fitbit"))

    # Oura
    if "oura" not in ignore:
        df_oura = load_oura_sleep_df()
        df_oura = _merge_several_sleep_records_per_day(df_oura)
        df = join(df, df_oura.add_suffix("_oura"))

    # Whoop
    if "whoop" not in ignore:
        df_whoop = load_whoop_sleep_df()
        df_whoop = _merge_several_sleep_records_per_day(df_whoop)
        df = join(df, df_whoop.add_suffix("_whoop"))

    assert _check_index(df, "all")

    # perform some aggregations
    if aggregate:
        keys: list[str] = list(
            set(col.split("_")[0] for col in df.columns) & {"duration", "score"}
        )
        for key in keys:
            subkeys = df.columns[df.columns.str.startswith(key)]
            df[key] = df[subkeys].mean(axis=1)
        df = df[keys]  # type: ignore

    df.index.name = "date"
    return df


def join(df_target, df_source, **kwargs) -> pd.DataFrame:
    if df_target.empty:
        return df_source
    else:
        return df_target.join(df_source, **kwargs)


@click.command()
@click.option("--aggregate/--no-aggregate", default=True)
@click.option("--plot/--no-plot", default=False)
@click.option("--dropna/--no-dropna", default=True)
def sleep(aggregate=False, plot=False, dropna=True):
    """Loads sleep data"""
    df = load_sleep_df(aggregate=aggregate)
    if dropna:
        df = df.dropna()
    if not aggregate:
        sources = ["fitbit", "oura", "whoop"]
        cols = ["duration", "score"]
        print(df[[f"{c}_{s}" for s in sources for c in cols]])
        # compare durations to ensure they are matching
        df_durations = df[[f"duration_{s}" for s in sources]].apply(
            lambda x: x.dt.seconds / 60 / 60
        )
        print(df_durations.head())
        if plot:
            df_durations.iloc[-30:].plot(kind="bar")
            plt.show()
    else:
        print(df[["duration", "score"]])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sleep()
