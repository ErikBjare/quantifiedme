import json
import pandas as pd

from quantifiedme.config import load_config


def load_activity_history() -> pd.DataFrame:
    """
    Load activity history from Google Takeout.

    Specifically the search history, for now.
    """
    config = load_config()
    activity_file = config["data"]["google_takeout"]["activity"]
    with open(activity_file) as f:
        activity = pd.DataFrame(json.load(f))

    activity["time"] = pd.to_datetime(activity["time"])
    # set the index to the time
    activity = activity.set_index("time")
    return activity


if __name__ == "__main__":
    activity = load_activity_history()
    # print(activity.describe())
    print(activity.keys())
    print(activity[:10][["title", "titleUrl"]])
    print(f"Length: {len(activity)}")

    import matplotlib.pyplot as plt

    # plot a histogram with count of serches by hour of day
    # activity["hour"] = activity.index.hour
    # activity["hour"].hist(bins=24)
    # plt.show()

    # now plot by day of week
    # activity["day"] = activity.index.dayofweek
    # activity["day"].hist(bins=7)
    # plt.show()

    # now plot by year-month
    activity["year-month"] = activity.index.strftime("%Y-%m")
    activity["year-month"].hist(bins=12 * 11)
    plt.show()
