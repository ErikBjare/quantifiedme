from datetime import datetime, timedelta, timezone


def test_load_qslang():
    from quantifiedme.qslang import load_df, to_series

    df = load_df()
    assert not df.empty

    for tag in ['alcohol']:
        series = to_series(df, tag=tag)
        assert not series[series != 0].empty

    for subst in ["Caffeine"]:
        series = to_series(df, substance=subst)
        assert not series[series != 0].empty


def test_load_activitywatch():
    from quantifiedme.activitywatch import load_complete_timeline, load_category_df
    events = load_complete_timeline(datetime.now(tz=timezone.utc) - timedelta(days=90), datasources=["fake"])
    assert not load_category_df(events).empty


def test_load_location():
    from quantifiedme.location import load_all_dfs
    assert load_all_dfs()


def test_load_habitbull():
    from quantifiedme.habitbull import load_df
    assert not load_df().empty


def test_load_oura():
    from quantifiedme.oura import load_sleep_df, load_activity_df

    assert not load_sleep_df().empty

    activity_df = load_activity_df()
    assert not activity_df.empty
