from pathlib import Path

import calplot
import click
import matplotlib.pyplot as plt
import pandas as pd

from ..config import load_config


def load_df():
    filename = load_config()["data"]["habitbull"]
    df = pd.read_csv(Path(filename).expanduser(), parse_dates=True)
    del df["HabitDescription"]
    del df["HabitCategory"]
    df = df.set_index(["CalendarDate", "HabitName"]).sort_index()
    return df


def plot_calendar(df, habitname, show=True, year=None):
    df = df[df.index.get_level_values("HabitName").isin([habitname])].reset_index()
    df = df.set_index(pd.DatetimeIndex(df["CalendarDate"]))
    if year:
        calplot.yearplot(df["Value"], year=year)
    else:
        calplot.calplot(df["Value"])
    if show:
        plt.show()


@click.command()
@click.argument("habitname", required=False)
@click.option("--year", default=None, type=int)
def habits(habitname: str | None = None, year: int | None = None):
    """Plot a habit calendar."""
    df = load_df()
    if habitname:
        plot_calendar(df, habitname, year=year)
    else:
        print("Habits:")
        for habit in list(set(df.index.get_level_values(1))):
            print(f" - {habit}")
        print("Specify a habit to plot it.")


if __name__ == "__main__":
    habits()
