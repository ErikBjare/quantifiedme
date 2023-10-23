import click
import pandas as pd

from ..load import fitbit, oura, whoop


# load heartrate from multiple sources, combine into a single dataframe
def load_heartrate_df() -> pd.DataFrame:
    dfs = []

    print("# Loading Oura heartrate data")
    oura_df = oura.load_heartrate_df()
    oura_df["source"] = "oura"
    dfs.append(oura_df)

    print("# Loading Fitbit heartrate data")
    fitbit_df = fitbit.load_heartrate_df()
    fitbit_df["source"] = "fitbit"
    dfs.append(fitbit_df)

    print("# Loading Whoop heartrate data")
    whoop_df = whoop.load_heartrate_df()
    whoop_df["source"] = "whoop"
    dfs.append(whoop_df)

    df = pd.concat(dfs)
    df = df.sort_index()
    return df


def load_heartrate_minutes_df():
    """We consider using minute-resolution a decent starting point for summary heartrate data.

    NOTE: ignores source, combines all sources into a single point per freq.
    """
    df = load_heartrate_df().drop(columns=["source"])
    df = df.resample("1min").mean()
    return df


def load_heartrate_summary_df(
    zones={"resting": 0, "low": 100, "med": 140, "high": 160}, freq="D"
) -> pd.DataFrame:
    """
    Load heartrates, group into freq, bin by zone, and return a dataframe.
    """
    source_df = load_heartrate_minutes_df()
    df = pd.DataFrame()
    df["hr_mean"] = source_df["hr"].groupby(pd.Grouper(freq=freq)).mean()

    # compute time spent in each zone
    df_zones = pd.cut(
        source_df["hr"], bins=[*zones.values(), 300], labels=[*zones.keys()]
    )
    for zone in zones.keys():
        df[f"hr_duration_{zone}"] = df_zones[df_zones == zone].groupby(
            pd.Grouper(freq=freq)
        ).count() * pd.Timedelta(minutes=1)
    # replace NaT with 0
    df = df.fillna(pd.Timedelta(0))
    return df


@click.command()
@click.option("--freq", default="D")
def heartrate(freq: str):
    """Loads heartrate data."""
    df = load_heartrate_summary_df(freq=freq)
    print(df)


if __name__ == "__main__":
    heartrate()
