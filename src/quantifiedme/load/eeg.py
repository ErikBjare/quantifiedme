"""
Load EEG data into a pandas dataframe.

We will load data from Neurosity's file exports provided by Neurofusion.

We will also load signal quality data, to help us filter out bad data.
"""

from collections import defaultdict
from pathlib import Path

import pandas as pd

data_dir = Path(__file__).parent.parent.parent.parent / "data"
fusion_dir = data_dir / "eeg" / "fusion"
fusion_dir_erik = fusion_dir / "erik"

# My device ID
erik_id = "099d8993-9dee-4518-9c36-bec8f5a91441"

# FIXME: don't hardcode
data_raw = fusion_dir_erik / f"{erik_id}_neurosity_rawBrainwaves_1678196988.json"
data_sig = fusion_dir_erik / f"{erik_id}_neurosity_signalQuality_1678196988.json"
data_pbb = fusion_dir_erik / f"{erik_id}_neurosity_powerByBand_1678196988.json"

# the common band names, ordered by frequency
bands_ordered = ["delta", "theta", "alpha", "beta", "gamma"]


def load_data():
    filesets = load_fileset()
    df = pd.DataFrame()
    for timestamp, files in filesets.items():
        result = load_session(files)
        # pprint(result)
        df = pd.concat([df, pd.DataFrame(result)], ignore_index=True)
    return df


def load_session(files: dict) -> dict:
    """
    Takes a session of EEG data, computes some metrics, and returns them.

    Potential metrics:
     - average power by band
     - average power by channel
     - relative power by band
     - average focus/calm score
    """
    # Best channels are usually: CP3, CP4, PO3, PO4

    # NOTE: unixTimestamps are int/seconds but samples more often than 1Hz,
    #       so several rows per timestamp and missing sub-second resolution.
    df_pbb = pd.read_json(files["powerByBand"])
    df_pbb.set_index("unixTimestamp", inplace=True)
    df_pbb.index = pd.to_datetime(df_pbb.index, unit="s")

    # we have to deal with the channels, such as CP3_alpha, CP3_beta, etc.
    # for now, we will just average them all together
    channels, bands = zip(*[c.split("_") for c in df_pbb.columns])
    channels, bands = tuple(set(channels)), tuple(set(bands))

    df = pd.DataFrame(index=df_pbb.index)
    for band in bands:
        channels_with_band = [c for c in df_pbb.columns if c.endswith(band)]
        df[band] = df_pbb[channels_with_band].mean(axis=1)
    average_band_power = df.mean()[bands_ordered]

    df = pd.DataFrame(index=df_pbb.index)
    for channel in channels:
        bands_for_channel = [c for c in df_pbb.columns if c.startswith(channel)]
        df[channel] = df_pbb[bands_for_channel].mean(axis=1)
    average_channel_power = df.mean()[channels]

    df_calm = pd.read_json(files["calm"])
    avg_calm_score = df_calm["probability"].mean()
    time_spent_calm = (df_calm["probability"] > 0.5).sum() / len(df_calm)

    df_focus = pd.read_json(files["focus"])
    avg_focus_score = df_focus["probability"].mean()
    time_spent_focused = (df_focus["probability"] > 0.5).sum() / len(df_focus)

    return {
        "timestamp": df_pbb.index[0],
        "duration": df_pbb.index[-1] - df_pbb.index[0],
        "avg_power_per_channel_by_band": {
            channel: {band: df_pbb[channel + "_" + band].mean() for band in bands}
            for channel in channels
        },
        "avg_power_by_band": dict(average_band_power),
        "avg_power_by_channel": dict(average_channel_power),
        "avg_calm_score": avg_calm_score,
        "avg_focus_score": avg_focus_score,
        "time_spent_calm": time_spent_calm,
        "time_spent_focused": time_spent_focused,
        # `relative_power` keys are 2-tuples (band1, band2), values are ratios
        # maybe doesn't need to be computed here,
        # can be computed later from `avg_power_by_band`
        # "relative_power": {},
    }


def load_sessions():
    """Loads all sessions of EEG data, compute & return metric for each session."""
    filesets = load_fileset()
    sessions: dict[str, dict] = {}
    for timestamp, files in filesets.items():
        sessions[timestamp] = load_session(files)
    return sessions


def load_fileset(
    dir=data_dir / "eeg" / "fusion" / "ore" / "resting",
) -> dict[str, dict[str, str]]:
    """
    Iterated over files in a directory with recordings,
    groups them by their session/start timestamp,
    and returns the filenames for each session as a dict.
    """
    files = list(dir.glob("*.json"))
    files.sort()
    sessions: dict[str, dict[str, str]] = defaultdict(dict)
    for f in files:
        if f.stem == "events":
            continue
        *type, session = f.stem.split("_")
        sessions[session]["_".join(type)] = str(f)
    return sessions


if __name__ == "__main__":
    df = load_data()
    print(df)
    print(df.describe())
