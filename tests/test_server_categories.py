from datetime import datetime, timedelta, timezone

import aw_research.classify
from aw_core import Event

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
