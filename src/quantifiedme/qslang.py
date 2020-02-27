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


@memory.cache
def load_df():
    events = load_events()
    events = [e for e in events]

    for e in events:
        try:
            e.data["amount_pint"] = ureg(e.data.get("amount") or 0)
            try:
                if not isinstance(e.data["amount_pint"], int):
                    e.data["amount_pint"] = e.data["amount_pint"].magnitude
            except Exception as e:
                print(e)
                #pass
                break
        except pint.UndefinedUnitError:
            logger.warning(e)
            pass

    df = pd.DataFrame([{
        'timestamp': e.timestamp,
        'date': (e.timestamp + timedelta(hours=-4)).date(),
        'substance': e.data.get('substance'),
        'dose': e.data.get("amount_pint") or 0,
        'tag': list(sorted(e.data.get('tags') or set(['none'])))[0]  # TODO: Needs a "main category"
    } for e in events])

    df['date'] = pd.to_datetime(df['date'])
    return df


def to_series(tag=None, substance=None):
    assert tag or substance
    #print(tag or subst)
    key = 'tag' if tag else 'substance'
    val = tag if tag else substance
    df = load_df()
    df = df[df[key] == val]
    #print(df)
    df.drop(key, axis=1)
    if substance:
        # Sum weight of doses for same substance
        series = df.groupby(['date']).agg({'dose': 'sum'})['dose']
    else:
        # Count doses if different substance with same tag
        series = df.groupby(['date']).agg({'timestamp': 'size'})['timestamp']
        #series = series.clip(0, 1)
    series = series.resample('1D').asfreq().replace(np.nan, 0)
    return series


# Entrypoint
qslang = qslang_entry


if __name__ == "__main__":
    df = load_df()
    print(df)
    #qslang()
