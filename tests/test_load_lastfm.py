"""Tests for the Last.fm scrobble data loader."""

from pathlib import Path

import pandas as pd
import pytest

from quantifiedme.load.lastfm import load_listening_daily_df, load_scrobbles_df

SAMPLE_CSV = """\
Artist,Album,Track,Date
Radiohead,OK Computer,Paranoid Android,2024-01-15 20:30:00
Radiohead,OK Computer,Karma Police,2024-01-15 20:34:00
Daft Punk,Discovery,One More Time,2024-01-15 21:00:00
Radiohead,In Rainbows,Reckoner,2024-01-16 10:15:00
Boards of Canada,Music Has the Right to Children,Roygbiv,2024-01-16 10:20:00
"""


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    p = tmp_path / "lastfm-export.csv"
    p.write_text(SAMPLE_CSV)
    return p


def test_load_scrobbles_df(sample_csv: Path) -> None:
    df = load_scrobbles_df(path=sample_csv)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    assert df.index.name == "timestamp"
    assert isinstance(df.index, pd.DatetimeIndex)
    assert str(df.index.tz) == "UTC"

    # Check columns
    assert "artist" in df.columns
    assert "album" in df.columns
    assert "track" in df.columns

    # Check content
    assert df.iloc[0]["artist"] == "Radiohead"
    assert df.iloc[2]["track"] == "One More Time"


def test_load_scrobbles_df_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_scrobbles_df(path=Path("/nonexistent/lastfm.csv"))


def test_load_listening_daily_df(sample_csv: Path) -> None:
    df = load_listening_daily_df(path=sample_csv)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2  # 2 days
    assert df.index.name == "timestamp"

    # Check columns
    assert "scrobble_count" in df.columns
    assert "unique_artists" in df.columns
    assert "unique_tracks" in df.columns
    assert "listening_hours" in df.columns

    # Day 1 had 3 scrobbles, 2 unique artists
    day1 = df.iloc[0]
    assert day1["scrobble_count"] == 3
    assert day1["unique_artists"] == 2

    # Day 2 had 2 scrobbles
    day2 = df.iloc[1]
    assert day2["scrobble_count"] == 2
