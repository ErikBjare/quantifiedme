from datetime import timezone

import pandas as pd


# load heartrate from multiple sources, combine into a single dataframe
def load_heartrate_df() -> pd.DataFrame:
    from ..load import oura, fitbit, whoop

    dfs = []

    oura_df = oura.load_heartrate_df()
    oura_df["source"] = "oura"
    dfs.append(oura_df)

    fitbit_df = fitbit.load_heartrate_df()
    fitbit_df["source"] = "fitbit"
    dfs.append(fitbit_df)

    whoop_df = whoop.load_heartrate_df()
    whoop_df["source"] = "whoop"
    dfs.append(whoop_df)

    df = pd.concat(dfs)
    df = df.sort_index()
    return df


def load_heartrate_daily_df(
    zones={"low": 100, "med": 140, "high": 160}
) -> pd.DataFrame:
    """Load heartrates, group into day, bin by zone, and return a dataframe."""
    df = load_heartrate_df()
    df = df.groupby(pd.Grouper(freq="D")).mean()
    df["zone"] = pd.cut(
        df["heartrate"], bins=[0, *zones.values(), 300], labels=zones.keys()
    )
    return df


if __name__ == "__main__":
    df = load_heartrate_df()
    print(df)
    print(df.describe())

    df = load_heartrate_daily_df()
    print(df)
