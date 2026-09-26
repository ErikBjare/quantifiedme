from datetime import datetime, timedelta, timezone

import aw_research.classify
from aw_core import Event

from quantifiedme.derived import screentime
from quantifiedme.derived.screentime import server_classes_to_aw_research

CLASSES: list[tuple[list[str], dict]] = [
    (["Work"], {"type": "regex", "regex": "Google Docs"}),
    (["Work", "Programming"], {"type": "regex", "regex": "GitHub"}),
    (
        ["Work", "Programming", "ActivityWatch"],
        {"type": "regex", "regex": "activitywatch", "ignore_case": True},
    ),
    (["Media", "Games"], {"type": "regex", "regex": "Factorio"}),
    (["Media", '"Social"'], {"type": "regex", "regex": "reddit"}),
    (["P", "Games"], {"type": "none"}),
    (["P", "Games", "Some Game"], {"type": "regex", "regex": "Some Game"}),
    (["Uncategorized"], {"type": "none"}),
]


def test_tags_are_unique_and_parents_registered():
    rules = server_classes_to_aw_research(CLASSES)
    tags = [tag for _, tag, _ in rules]
    assert len(tags) == len(set(tags))
    by_tag = {tag: (regex, parent) for regex, tag, parent in rules}
    # ambiguous leaf names are qualified, unambiguous ones are not
    assert by_tag["Media>Games"][1] == "Media"
    assert by_tag["P>Games"][1] == "P"
    assert by_tag["Some Game"][1] == "P>Games"
    assert by_tag["Programming"][1] == "Work"
    # quotes stripped, missing top-level parents created, Uncategorized skipped
    assert "Social" in by_tag
    assert by_tag["P"][1] is None
    assert "Uncategorized" not in by_tag
    # ignore_case becomes an inline flag
    assert by_tag["ActivityWatch"][0] == "(?i)activitywatch"


def test_classify_with_server_classes():
    aw_research.classify._init_classes(
        new_classes=server_classes_to_aw_research(CLASSES)
    )
    now = datetime.now(tz=timezone.utc)
    titles = ["ActivityWatch - GitHub", "Some Game", "Factorio", "Something else"]
    events = [
        Event(
            timestamp=now, duration=timedelta(minutes=1), data={"app": "x", "title": t}
        )
        for t in titles
    ]
    events = aw_research.classify.classify(events)
    assert events[0].data["$tags"] == {"ActivityWatch", "Programming", "Work"}
    assert events[1].data["$tags"] == {"Some Game", "P>Games", "P"}
    assert events[2].data["$tags"] == {"Media>Games", "Media"}
    assert events[3].data["$tags"] == {"Uncategorized"}


def test_tags_unique_when_names_contain_separator():
    rules = server_classes_to_aw_research(
        [
            (["A>B"], {"type": "regex", "regex": "x"}),
            (["A", "B"], {"type": "regex", "regex": "y"}),
            (["C", "B"], {"type": "regex", "regex": "z"}),
        ]
    )
    tags = [tag for _, tag, _ in rules]
    assert len(tags) == len(set(tags))


class _FakeClient:
    def __init__(self, classes=None, fail=False):
        self.classes = classes
        self.fail = fail

    def get_setting(self, key):
        assert key == "classes"
        if self.fail:
            raise ConnectionError("no server")
        return self.classes


def test_load_server_classes(monkeypatch):
    stored = [{"id": 0, "name": ["Work"], "rule": {"type": "regex", "regex": "x"}}]
    monkeypatch.setattr(screentime, "_get_aw_client", lambda t: _FakeClient(stored))
    assert screentime.load_server_classes() == [(["Work"], stored[0]["rule"])]


def test_load_server_classes_fallback(monkeypatch):
    from aw_client.classes import default_classes

    for client in [_FakeClient(fail=True), _FakeClient(classes=[])]:
        monkeypatch.setattr(screentime, "_get_aw_client", lambda t, c=client: c)
        assert screentime.load_server_classes() == default_classes


def test_classify_uses_server_by_default(monkeypatch):
    monkeypatch.setattr(screentime, "load_config", lambda use_example: {"data": {}})
    monkeypatch.setattr(screentime, "load_server_classes", lambda testing: CLASSES)
    now = datetime.now(tz=timezone.utc)
    events = [
        Event(timestamp=now, duration=timedelta(minutes=1), data={"app": "Factorio"})
    ]
    events = screentime.classify(events, personal=True)
    assert events[0].data["$tags"] == {"Media>Games", "Media"}


def test_tags_independent_of_rule_order():
    classes: list[tuple[list[str], dict]] = [
        (["A>B"], {"type": "regex", "regex": "x"}),
        (["A", "B"], {"type": "regex", "regex": "y"}),
        (["C", "B"], {"type": "regex", "regex": "z"}),
    ]
    fwd = {t: r for r, t, _ in server_classes_to_aw_research(classes)}
    rev = {t: r for r, t, _ in server_classes_to_aw_research(classes[::-1])}
    assert fwd == rev
