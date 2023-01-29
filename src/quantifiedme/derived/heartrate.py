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

    #whoop_df = whoop.load_heartrate_df()
    #whoop_df["source"] = "whoop"
    #dfs.append(whoop_df)

    df = pd.concat(dfs)
    df = df.sort_index()
    return df


if __name__ == "__main__":
    df = load_heartrate_df()
    print(df)
    print(df.describe())