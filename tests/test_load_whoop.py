from quantifiedme.load.whoop import load_heartrate_df, load_sleep_df


def test_load_whoop_heartrate():
    df = load_heartrate_df()
    print(df.head())


def test_load_whoop_sleep():
    df = load_sleep_df()
    print(df.head())
