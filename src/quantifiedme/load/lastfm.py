"""
Loads Last.fm scrobble history.

Data can be exported using tools like:
- https://github.com/benfoxall/lastfm-to-csv
- https://mainstream.ghan.nl/export.html

The expected CSV format has columns: Artist, Album, Track, Date
"""

from pathlib import Path

import pandas as pd

from ..config import load_config


def load_scrobbles_df(path: Path | None = None) -> pd.DataFrame:
    """
    Load Last.fm scrobble history.

    Returns a DataFrame indexed by timestamp with columns:
    - artist
    - album
    - track
    """
    if path is None:
        config = load_config()
        path = Path(config["data"]["lastfm"]).expanduser()
    else:
        path = Path(path).expanduser()

    if not path.exists():
        raise FileNotFoundError(f"Last.fm export not found at {path}")

    # lastfm-to-csv exports use comma separator with these columns
    df = pd.read_csv(path, parse_dates=["Date"])

    # Normalize column names
    df.columns = df.columns.str.lower().str.strip()

    # Set up datetime index
    df = df.rename(columns={"date": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")
    df = df.sort_index()

    return df


def load_listening_daily_df(path: Path | None = None) -> pd.DataFrame:
    """
    Aggregate scrobbles into daily listening statistics.

    Returns a DataFrame indexed by date with columns:
    - scrobble_count: number of tracks played
    - unique_artists: number of distinct artists
    - unique_tracks: number of distinct tracks
    - listening_hours: estimated listening time (assuming ~3.5 min/track)
    """
    df = load_scrobbles_df(path)

    daily = df.resample("D").agg(
        scrobble_count=("artist", "count"),
        unique_artists=("artist", "nunique"),
        unique_tracks=("track", "nunique"),
    )
    # Estimate listening hours (~3.5 min average track length)
    daily["listening_hours"] = daily["scrobble_count"] * 3.5 / 60

    daily.index.name = "timestamp"
    return daily
