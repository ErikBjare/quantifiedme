import logging
from datetime import timedelta

import pandas as pd
import numpy as np
from joblib import Memory
import pint

from qslang import Event
from qslang.main import main as qslang_main, load_events
from ..config import load_config

logger = logging.getLogger(__name__)

ureg = pint.UnitRegistry()
ureg.define("mc- = 10**-6 = µ-")

memory = Memory(".cache", verbose=0)
# memory.clear()


class DuplicateFilter:
    """
    Filters away duplicate log messages.
    Modified version of: https://stackoverflow.com/a/31953563/965332
    """

    def __init__(self, logger):
        self.msgs = set()
        self.logger = logger

    def filter(self, record):
        msg = str(record.msg)
        is_duplicate = msg in self.msgs
        if not is_duplicate:
            self.msgs.add(msg)
        return not is_duplicate

    def __enter__(self):
        self.logger.addFilter(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.removeFilter(self)


@memory.cache
def load_df(events: list[Event] | None = None) -> pd.DataFrame:
    if events is None:
        events = load_events()
    events = [e for e in events]

    with DuplicateFilter(logger):
        for e in events:
            if e.type != "dose":
                continue
            try:
                # event data can contain either:
                #  - "amount" (string-formatted amount)
                #  - "dose" (dict with "amount" and "unit" keys)

                # If amount is unknown, we should use the mean dose for the substance in its place,
                # so we'll set it to np.nan in the meantime
                if "?" in e.data.get("amount", ""):
                    e.data["amount_pint"] = np.nan
                else:
                    dose = e.data.get("dose")
                    if dose:
                        amount = dose.get("amount", "")
                        unit = dose.get("unit", "")
                        e.data["amount_pint"] = ureg(f"{amount}{unit}")
                    else:
                        assert "amount" in e.data
                        e.data["amount_pint"] = ureg(e.data["amount"])

                    if not isinstance(e.data["amount_pint"], (int, float)):
                        e.data["amount_pint"] = (
                            e.data["amount_pint"].to_base_units().magnitude
                        )
            except pint.UndefinedUnitError as e:
                logger.warning(e)

    date_offset = timedelta(hours=load_config()["me"]["date_offset_hours"])

    df = pd.DataFrame(
        [
            {
                "timestamp": e.timestamp,
                "date": (e.timestamp - date_offset).date(),
                "substance": e.data.get("substance"),
                "dose": e.data.get("amount_pint") or 0,
                # FIXME: Only supports one tag
                "tag": list(sorted(e.data.get("tags") or {"none"}))[0],
                "data": e.data,
            }
            for e in events
        ]
    )

    # Replace NaN with mean of non-zero doses
    for substance in set(df["substance"]):
        doses = df[df["substance"] == substance]['dose']
        mean_dose = doses.dropna().mean()
        df.loc[df["substance"] == substance, ['dose']] = doses.replace(np.nan, mean_dose)

    return df


def to_series(
    df: pd.DataFrame, tag: str | None = None, substance: str | None = None
) -> pd.Series:
    """
    Takes a dataframe with multiple dose events and returns a date series with the
    total daily dose (if substance specified) or the daily count of doses (if tag specified).
    """
    assert tag or substance
    key = "tag" if tag else "substance"
    val = tag if tag else substance

    # Filter for the tag/substance we want
    df = df[df[key] == val]

    # Drop the tag/substance column
    df.drop(key, axis=1)

    if substance:
        # Sum weight of doses for same substance
        series = df.groupby(["date"]).agg({"dose": "sum"})["dose"]
    else:
        # Count doses if different substance with same tag
        series = df.groupby(["date"]).agg({"timestamp": "size"})["timestamp"]
        # series = series.clip(0, 1)

    # Reindex
    series.index = pd.DatetimeIndex(series.index)

    # Resample and insert zero for days with no data
    series = series.resample("1D").asfreq().replace(np.nan, 0)

    return series


def to_df_daily(events: list[Event]):
    """Returns a daily dataframe"""
    df_src = load_df(events)
    df = pd.DataFrame()
    tags = {tag for e in events for tag in e.data.get("tags", [])}
    print(tags)
    for tag in tags:
        df[f"tag:{tag}"] = to_series(df_src, tag=tag)

    substances = {s for s in df_src["substance"] if s}
    for subst in substances:
        colname = subst.lower().replace("-", "").replace(" ", "")
        df[colname] = to_series(df_src, substance=subst)


def _missing_dates():
    # Useful helper function to find dates without any entries
    df = load_df()
    dates_with_entries = sorted(
        {(d + timedelta(hours=-4)).date() for d in df["timestamp"]}
    )
    all_dates = sorted(
        d.date()
        for d in pd.date_range(min(dates_with_entries), max(dates_with_entries))
    )
    dates_without_entries = sorted(set(all_dates) - set(dates_with_entries))
    print(dates_without_entries)


# Entrypoint
main = qslang_main


if __name__ == "__main__":
    _missing_dates()
    # main()
