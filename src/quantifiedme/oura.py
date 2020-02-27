import json

import click
import pandas as pd

from .config import load_config


def load_data():
    filepath = load_config()["data"]["oura"]
    with open(filepath) as f:
        data = json.load(f)
    return data


def load_sleep_df():
    data = load_data()

    df = pd.DataFrame(data['sleep'])
    df['summary_date'] = pd.to_datetime(df['summary_date'])
    df = df.set_index('summary_date')

    df['weekend'] = [1 if (d.weekday() + 1) % 7 >= 5 else 0 for d in df.index]

    return df


def load_activity_df():
    data = load_data()

    df = pd.DataFrame(data['activity'])
    df['summary_date'] = pd.to_datetime(df['summary_date'])
    df = df.set_index('summary_date')

    df['weekend'] = [1 if (d.weekday() + 1) % 7 >= 5 else 0 for d in df.index]
    print(df)

    return df


@click.command()
def oura():
    load_sleep_df()
    load_activity_df()


if __name__ == "__main__":
    oura()
