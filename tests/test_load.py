from datetime import datetime, timedelta, timezone

from aw_core import Event

now = datetime.now(tz=timezone.utc)


def test_load_qslang():
    from quantifiedme.qslang import load_df, to_series

    df = load_df()
    assert not df.empty

    for tag in ["alcohol"]:
        series = to_series(df, tag=tag)
        assert not series[series != 0].empty

    for subst in ["Caffeine"]:
        series = to_series(df, substance=subst)
        assert not series[series != 0].empty

        # Check that all values in reasonable range
        series_nonzero = series[series != 0]

        # More than 10mg
        assert (10e-6 <= series_nonzero).all()

        # Less than 500mg
        assert (series_nonzero <= 500e-6).all()

    for subst in ["Phenibut"]:
        series = to_series(df, substance=subst)
        assert not series[series != 0].empty

        # Check that all values in reasonable range
        series_nonzero = series[series != 0]

        # More than 100mg
        assert (100e-6 <= series_nonzero).all()

        # Less than 5000mg
        assert (series_nonzero <= 5000e-6).all()


def test_qslang_unknown_dose():
    from quantifiedme.qslang import load_df, to_series

    events = [
        Event(timestamp=now, data={"substance": "Caffeine", "amount": "?g"}),
        Event(timestamp=now, data={"substance": "Caffeine", "amount": "100mg"}),
        Event(timestamp=now, data={"substance": "Caffeine", "amount": "200mg"}),
    ]
    df = load_df(events)
    assert 0.00015 == df.iloc[0]["dose"]


def test_load_activitywatch():
    from quantifiedme.activitywatch import load_complete_timeline, load_category_df

    events = load_complete_timeline(
        datetime.now(tz=timezone.utc) - timedelta(days=90), datasources=["fake"]
    )
    assert not load_category_df(events).empty


def test_load_location():
    from quantifiedme.location import load_all_dfs

    assert load_all_dfs()


def test_load_habitbull():
    from quantifiedme.habitbull import load_df

    assert not load_df().empty


def test_load_oura():
    from quantifiedme.oura import load_sleep_df, load_activity_df, load_readiness_df

    assert not load_sleep_df().empty

    activity_df = load_activity_df()
    assert not activity_df.empty

    assert not load_readiness_df().empty
