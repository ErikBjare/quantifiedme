import logging
from datetime import timedelta

import pandas as pd
import numpy as np
from joblib import Memory
import pint

from qslang.main import qslang as qslang_entry, load_events

logger = logging.getLogger(__name__)

ureg = pint.UnitRegistry()
ureg.define('mc- = 10**-6')

memory = Memory('.cache', verbose=0)
#memory.clear()


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
def load_df():
    events = load_events()
    events = [e for e in events]

    with DuplicateFilter(logger):
        for e in events:
            try:
                e.data["amount_pint"] = ureg(e.data.get("amount") or 0)
                if not isinstance(e.data["amount_pint"], (int, float)):
                    e.data["amount_pint"] = e.data["amount_pint"].magnitude
            except pint.UndefinedUnitError as e:
                logger.warning(e)

    # TODO: Load from config.toml
    day_offset = timedelta(hours=-4)

    df = pd.DataFrame([{
        'timestamp': e.timestamp,
        'date': (e.timestamp + day_offset).date(),
        'substance': e.data.get('substance'),
        'dose': e.data.get("amount_pint") or 0,
        'tag': list(sorted(e.data.get('tags') or set(['none'])))[0]  # TODO: Needs a "main category"
    } for e in events])

    return df


def to_series(df, tag=None, substance=None):
    assert tag or substance
    key = 'tag' if tag else 'substance'
    val = tag if tag else substance

    # Filter for the tag/substance we want
    df = df[df[key] == val]

    # Drop the tag/substance column
    df.drop(key, axis=1)

    if substance:
        # Sum weight of doses for same substance
        series = df.groupby(['date']).agg({'dose': 'sum'})['dose']
    else:
        # Count doses if different substance with same tag
        series = df.groupby(['date']).agg({'timestamp': 'size'})['timestamp']
        #series = series.clip(0, 1)

    # Reindex
    series.index = pd.DatetimeIndex(series.index)

    # Resample and insert zero for days with no data
    series = series.resample('1D').asfreq().replace(np.nan, 0)

    return series


def _missing_dates():
    # Useful helper function to find dates without any entries
    df = load_df()
    dates_with_entries = sorted(set((d + timedelta(hours=-4)).date() for d in df["timestamp"]))
    all_dates = sorted(d.date() for d in pd.date_range(min(dates_with_entries), max(dates_with_entries)))
    dates_without_entries = sorted(set(all_dates) - set(dates_with_entries))
    print(dates_without_entries)


# Entrypoint
qslang = qslang_entry


if __name__ == "__main__":
    _missing_dates()
    #qslang()
