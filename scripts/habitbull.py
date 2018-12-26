#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import calmap


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


if __name__ == "__main__":
    df = read_csv('data/habitbulldata.csv')
    del df['HabitDescription']
    del df['HabitCategory']
