from .heartrate import load_heartrate_daily_df
from .screentime import load_screentime_daily_df


def load_all_df(events: list[Event]):
    df = load_screentime_daily_df(events)
    df = df.join(load_heartrate_daily_df(events))
    return df