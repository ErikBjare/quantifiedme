from typing import MutableMapping, Any
from pathlib import Path
import logging

import toml
import appdirs

logger = logging.getLogger(__name__)


srcdir = Path(__file__).resolve().parent.parent
rootdir = srcdir.parent


def _get_config_path():
    return Path(appdirs.user_config_dir("quantifiedme")) / "config.toml"


def load_config() -> MutableMapping[str, Any]:
    filepath = _get_config_path()
    if not filepath.exists():
        logger.warning("No config found, falling back to example config")
        filepath = Path(rootdir) / "config.example.toml"
    with open(filepath) as f:
        return toml.load(f)


def has_config() -> bool:
    filepath = _get_config_path()
    return filepath.exists()


if __name__ == "__main__":
    load_config()
