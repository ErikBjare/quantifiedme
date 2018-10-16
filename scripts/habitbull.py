#!/usr/bin/env python3

import pandas as pd


def read_csv(filename):
    df = pd.read_csv(filename, parse_dates=True)
    df = df.set_index(['CalendarDate', 'HabitName']).sort_index()
    return df


def test_read_csv():
    df = read_csv('data/habitbulldata.csv')
    print(df[['Value', 'CommentText']])
    print(df.columns)
