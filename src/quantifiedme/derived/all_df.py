from aw_core import Event
from typing import Literal

from .heartrate import load_heartrate_daily_df
from .screentime import load_category_df

Sources = Literal["activitywatch", "heartrate"]

def load_all_df(events: list[Event], ignore: list[Sources] = []):
    df = load_category_df(events)
    if "heartrate" not in ignore:
        df = df.join(load_heartrate_daily_df(events))
    return df
