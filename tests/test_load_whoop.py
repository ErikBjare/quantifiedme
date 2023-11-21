import pytest
from quantifiedme.config import load_config
from quantifiedme.load.whoop import load_heartrate_df, load_sleep_df

has_whoop_config = load_config().get("data", {}).get("whoop", False)


@pytest.mark.skipif(has_whoop_config, reason="no whoop config")
def test_load_whoop_heartrate():
    df = load_heartrate_df()
    print(df.head())


@pytest.mark.skipif(has_whoop_config, reason="no whoop config")
def test_load_whoop_sleep():
    df = load_sleep_df()
    print(df.head())
