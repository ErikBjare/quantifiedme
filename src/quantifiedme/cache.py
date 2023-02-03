from pathlib import Path

from joblib import Memory

cache_dir = Path("~/.cache/quantifiedme").expanduser()

memory = Memory(location=cache_dir, verbose=0)
