from aw_core import Event

from .heartrate import load_heartrate_daily_df
from .screentime import load_category_df


def load_all_df(events: list[Event]):
    df = load_category_df(events)
    df = df.join(load_heartrate_daily_df(events))
    return df
