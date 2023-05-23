from datetime import datetime, timedelta, timezone
from pathlib import Path
import pickle

import pytest

import pandas as pd

from aw_core import Event

from quantifiedme.config import has_config
from quantifiedme.load.qslang import load_df, to_series
from quantifiedme.derived.screentime import load_screentime, load_category_df
from quantifiedme.load.toggl_ import load_toggl
from quantifiedme.load.habitbull import load_df as load_habitbull_df
from quantifiedme.load.location import load_all_dfs
from quantifiedme.load.oura import (
    load_sleep_df,
    load_activity_df,
    load_readiness_df,
)
from quantifiedme.derived.all_df import load_all_df
from quantifiedme.derived.screentime import classify

from qslang import Event as QSEvent

now = datetime.now(tz=timezone.utc)


@pytest.fixture(scope="session", autouse=True)
def setup():
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.max_columns", None)
    # pd.set_option('display.max_rows', None)


def load_example_events() -> list[Event]:
    events_cached_fast = Path("notebooks/events_fast.pickle")
    if events_cached_fast.exists():
        with events_cached_fast.open("rb") as f:
            events = pickle.load(f)
    else:
        events = [
            Event(timestamp=now, data={"title": "Mozilla Firefox", "app": "Firefox"}),
        ]

    events = classify(events, personal=False)
    return events


def test_load_example_events():
    events = load_example_events()
    assert events


def test_load_all_df():
    events = load_example_events()
    df = load_all_df(events, ignore=["heartrate", "location", "sleep"])
    print(df)


@pytest.mark.skipif(not has_config(), reason="no config available for test data")
def test_load_qslang():
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
        assert (series_nonzero <= 500e-6).all(), series_nonzero[
            series_nonzero >= 500e-6
        ]

    for subst in ["Phenibut"]:
        series = to_series(df, substance=subst)
        assert not series[series != 0].empty

        # Check that all values in reasonable range
        series_nonzero = series[series != 0]

        # More than 100mg
        assert (100e-6 <= series_nonzero).all()

        # Less than 5000mg
        # FIXME: Seems some entries have >5000mg?
        print(series_nonzero[series_nonzero >= 5000e-6])
        # assert (series_nonzero <= 5000e-6).all()


def test_qslang_unknown_dose():
    events = [
        QSEvent(
            timestamp=now, type="dose", data={"substance": "Caffeine", "amount": "?g"}
        ),
        QSEvent(
            timestamp=now,
            type="dose",
            data={"substance": "Caffeine", "amount": "100mg"},
        ),
        QSEvent(
            timestamp=now,
            type="dose",
            data={"substance": "Caffeine", "amount": "200mg"},
        ),
    ]
    df = load_df(events)
    assert 0.00015 == df.iloc[0]["dose"]


def test_load_screentime():
    events = load_screentime(
        datetime.now(tz=timezone.utc) - timedelta(days=90),
        datasources=["fake"],
        hostnames=[],
        personal=False,
    )
    assert not load_category_df(events).empty


def test_load_toggl():
    pytest.skip("Broken")

    now = datetime.now(tz=timezone.utc)
    events = load_toggl(now - timedelta(days=90), now)
    print(events)
    assert events
    # assert False


@pytest.mark.skipif(not has_config(), reason="no config available for test data")
def test_load_location():
    assert load_all_dfs()


@pytest.mark.skipif(not has_config(), reason="no config available for test data")
def test_load_habitbull():
    assert not load_habitbull_df().empty


@pytest.mark.skipif(not has_config(), reason="no config available for test data")
def test_load_oura():
    assert not load_sleep_df().empty

    activity_df = load_activity_df()
    assert not activity_df.empty

    assert not load_readiness_df().empty
