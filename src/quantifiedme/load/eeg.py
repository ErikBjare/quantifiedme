"""
Load EEG data into a pandas dataframe.

We will load data from Neurosity's file exports provided by Neurofusion.

We will also load signal quality data, to help us filter out bad data.
"""

from pathlib import Path
from collections import defaultdict

import pandas as pd

from matplotlib import pyplot as plt

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


def load_data(files: dict):
    # df_raw = pd.read_json(files["rawBrainwaves"])
    # print(df_raw)
    # print(df_raw.describe())

    # df_sig = pd.read_json(files["signalQuality"])
    # print(df_sig)
    # print(df_sig.describe())

    pass


def load_session(files: dict) -> dict:
    """
    Takes a session of EEG data, computes some metrics, and returns them.

    Potential metrics:
     - average power by band
     - average power by channel
     - relative power by band
     - average focus/calm score
    """
    raise NotImplementedError
    return {
        "avg_power_by_band": {},
        "avg_power_by_channel": {},
        "avg_calm_score": 0.0,
        "avg_focus_score": 0.0,
        # `relative_power` keys are 2-tuples (band1, band2), values are ratios
        # maybe doesn't need to be computed here,
        # can be computed later from `avg_power_by_band`
        "relative_power": {},
    }


def load_sessions():
    """Loads all sessions of EEG data, compute & return metric for each session."""
    filesets = load_fileset()
    sessions: dict[str, dict] = {}
    for timestamp, files in filesets.items():
        sessions[timestamp] = load_session(files)
    return sessions


def _process_pbb(files: dict):
    """Process powerByBand data."""
    # Best channels are usually: CP3, CP4, PO3, PO4

    # NOTE: unixTimestamps are int/seconds but samples more often than 1Hz,
    #       so several rows per timestamp and missing sub-second resolution.
    df_pbb = pd.read_json(files["powerByBand"])
    df_pbb.set_index("unixTimestamp", inplace=True)
    df_pbb.index = pd.to_datetime(df_pbb.index, unit="s")
    df_pbb = df_pbb.resample("5s").mean()
    print(df_pbb)
    print(df_pbb.describe())

    # we have to deal with the channels, such as CP3_alpha, CP3_beta, etc.
    # for now, we will just average them all together
    channels, bands = zip(*[c.split("_") for c in df_pbb.columns])
    channels, bands = list(set(channels)), list(set(bands))
    df = pd.DataFrame(index=df_pbb.index)
    for band in bands:
        channel_with_band = [c for c in df_pbb.columns if c.endswith(band)]
        df[band] = df_pbb[channel_with_band].mean(axis=1)

    # plot bands as bar chart
    print(df.describe())
    df.mean()[bands_ordered].plot.bar()
    plt.title("Average power by band, time: " + str(df.index[0]))
    plt.show()


def load_fileset(dir=data_dir / "eeg" / "fusion" / "ore" / "resting"):
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
    filesets = load_fileset()
    for timestamp, files in filesets.items():
        # print(files)
        # load_data(files)
        result = _process_pbb(files)
