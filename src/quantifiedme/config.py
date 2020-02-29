from typing import MutableMapping, Any
from pathlib import Path
import toml
import appdirs


def load_config() -> MutableMapping[str, Any]:
    filepath = Path(appdirs.user_config_dir("quantifiedme")) / "config.toml"
    with open(filepath) as f:
        return toml.load(f)
