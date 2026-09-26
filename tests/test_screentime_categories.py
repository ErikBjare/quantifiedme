from datetime import datetime, timedelta, timezone

import aw_research.classify
from aw_core import Event
from quantifiedme.derived.screentime import load_category_df

CLASSES: list[tuple[str, str, str | None]] = [
    ("^$", "Work", None),
    ("Programming", "Programming", "Work"),
    ("Steam", "Games", "Media"),
    ("^$", "Media", None),
    ("Secret", "P", None),
    ("SecretGame", "Games(P)", "P"),
]


def _event(title: str, hours: float) -> Event:
    return Event(
        timestamp=datetime(2026, 1, 1, 12, tzinfo=timezone.utc),
        duration=timedelta(hours=hours),
        data={"title": title, "app": "app"},
    )


def test_category_df_no_substring_match(monkeypatch):
    """Category names that are substrings of others (P, Games) must not be inflated."""
    # restore the module-level classifier config after the test
    for attr in ["classes", "parent_categories"]:
        monkeypatch.setattr(
            aw_research.classify, attr, getattr(aw_research.classify, attr)
        )
    aw_research.classify._init_classes(new_classes=CLASSES)
    events = aw_research.classify.classify(
        [_event("Programming", 1), _event("Steam", 2), _event("SecretGame", 4)]
    )
    df = load_category_df(events)
    assert df["Programming"].sum() == 1
    assert df["Work"].sum() == 1
    assert df["P"].sum() == 4
    assert df["Games"].sum() == 2
    assert df["Games(P)"].sum() == 4
    assert df["Media"].sum() == 2
