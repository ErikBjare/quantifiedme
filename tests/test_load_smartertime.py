import sys
from datetime import datetime
from pathlib import Path

import pytest
from quantifiedme.load.smartertime import (
    _load_smartertime_events,
    convert_csv_to_awbucket,
)

test_file = Path(
    "~/Programming/quantifiedme/data/smartertime/smartertime_export_erb-f3_2022-02-01_efa36e6a.awbucket.json"
).expanduser()


@pytest.mark.skipif(not test_file.exists(), reason="test file not found")
def test_load_smartertime_events():
    events = _load_smartertime_events(
        datetime(2020, 1, 1),
        filepath=test_file,
    )
    assert len(events) > 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "convert":
        assert len(sys.argv) > 2
        filename = sys.argv.pop()
        convert_csv_to_awbucket(filename)
