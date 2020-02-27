import click
import pandas as pd
import matplotlib.pyplot as plt
import calmap

from .config import load_config


def read_csv(filename):
    df = pd.read_csv(filename, parse_dates=True)
    df = df.set_index(['CalendarDate', 'HabitName']).sort_index()
    return df


def plot_calendar(df, habitname, show=True, year=None):
    df = df[df.index.get_level_values('HabitName').isin([habitname])].reset_index()
    df = df.set_index(pd.DatetimeIndex(df['CalendarDate']))
    if year:
        calmap.yearplot(df['Value'], year=year)
    else:
        calmap.calendarplot(df['Value'])
    if show:
        plt.show()


def test_read_csv():
    df = read_csv('data/habitbulldata.csv')
    print(df[['Value', 'CommentText']])
    print(df.columns)
    # assert 0


def test_plot_calendar():
    df = read_csv('data/habitbulldata.csv')
    plot_calendar(df, "Socialized", show=False)


@click.command()
@click.argument('habitname', required=False)
@click.option('--year', default=None, type=int)
def habits(habitname: str = None, year: int = None):
    filename = load_config()["data"]["habitbull"]
    df = read_csv(filename)
    del df['HabitDescription']
    del df['HabitCategory']
    if habitname:
        plot_calendar(df, habitname, year=year)
    else:
        print("Habits:")
        for habit in list(set(df.index.get_level_values(1))):
            print(f" - {habit}")
        print("Specify a habit to plot it.")


if __name__ == "__main__":
    habits()
