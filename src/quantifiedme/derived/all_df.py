import os
import logging
from typing import Literal, TypeAlias
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from aw_core import Event

from ..load.location import load_daily_df as load_location_daily_df
from ..load.qslang import load_daily_df as load_drugs_df

from .heartrate import load_heartrate_summary_df
from .screentime import load_screentime_cached, load_category_df
from .sleep import load_sleep_df

Sources = Literal["screentime", "heartrate", "drugs", "location", "sleep"]


def load_all_df(
    fast=True, screentime_events: list[Event] | None = None, ignore: list[Sources] = []
) -> pd.DataFrame:
    """
    Loads a bunch of data into a single dataframe with one row per day.
    Serves as a useful starting point for further analysis.
    """
    df = pd.DataFrame()
    since = datetime.now(tz=timezone.utc) - timedelta(days=30 if fast else 2 * 365)

    if "screentime" not in ignore:
        print("Adding screentime")
        if screentime_events is None:
            screentime_events = load_screentime_cached(fast=fast, since=since)
        df_time = load_category_df(screentime_events)
        df_time = df_time[["Work", "Media", "ActivityWatch"]]
        df = join(df, df_time.add_prefix("time:"))

    if "heartrate" not in ignore:
        print("Adding heartrate")
        df_hr = load_heartrate_summary_df(freq="D")
        # translate daily datetime column to a date column
        df_hr.index = df_hr.index.date  # type: ignore
        df = join(df, df_hr)

    if "drugs" not in ignore:
        print("Adding drugs")
        # keep only columns starting with "tag"
        df_drugs = load_drugs_df()
        df_drugs = df_drugs[df_drugs.columns[df_drugs.columns.str.startswith("tag")]]
        df = join(df, df_drugs)

    if "location" not in ignore:
        print("Adding location")
        # TODO: add boolean for if sleeping together
        df_location = load_location_daily_df()
        df_location.index = df_location.index.date  # type: ignore
        df = join(df, df_location.add_prefix("loc:"))

    if "sleep" not in ignore:
        df_sleep = load_sleep_df()
        df = join(df, df_sleep.add_prefix("sleep:"))

    # look for all-na columns, emit a warning, and drop them
    na_cols = df.columns[df.isna().all()]
    if len(na_cols) > 0:
        print(f"Warning: dropping all-NA columns: {str(list(na_cols))}")
        df = df.drop(columns=na_cols)

    return df


def join(df_target: pd.DataFrame, df_source: pd.DataFrame) -> pd.DataFrame:
    if not df_target.empty:
        check_new_data_in_range(df_source, df_target)
    print(
        f"Adding new columns: {str(list(df_source.columns.difference(df_target.columns)))}"
    )
    return df_target.join(df_source) if not df_target.empty else df_source


DateLike: TypeAlias = datetime | date | pd.Timestamp


def datelike_to_date(d: DateLike) -> date:
    if isinstance(d, datetime) or isinstance(d, pd.Timestamp):
        return d.date()
    elif isinstance(d, date):
        return d
    else:
        raise ValueError(f"Invalid type for datelike: {type(d)}")


def check_new_data_in_range(df_source: pd.DataFrame, df_target: pd.DataFrame) -> None:
    # check that source data covers target data, or emit warning
    source_start = datelike_to_date(df_source.index.min())
    source_end = datelike_to_date(df_source.index.max())
    target_start = datelike_to_date(df_target.index.min())
    target_end = datelike_to_date(df_target.index.max())

    # check the worst case
    if source_start > target_end or source_end < target_start:
        print(
            f"Warning: source data does not cover ANY of target data: ({source_start}/{source_end}) not in ({target_start}/{target_end})"
        )
    elif source_start > target_start:
        print(
            f"Warning: source data starts after target data (partial): {source_start} > {target_start}"
        )
    elif source_end < target_end:
        print(
            f"Warning: source data ends before target data (partial): {source_end} < {target_end}"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # print a summary of all data
    df = load_all_df(fast=os.environ.get("FAST", "1") == "1")
    print(df)
    print(df.describe())

    # check for missing data
    df_days_na = df.isna().sum()
    df_days_na = df_days_na[df_days_na > 0]
    if len(df_days_na) > 0:
        print(f"Missing data for {len(df_days_na)} out of {len(df.columns)} columns")
        print(df_days_na)
    print("Total days: ", len(df))

    # keep days with full coverage
    df = df.dropna()
    print("Total days with full coverage: ", len(df))

    print("Final dataframe:")
    print(df)
