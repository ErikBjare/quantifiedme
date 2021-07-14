from typing import MutableMapping, Any
from pathlib import Path
import logging

import toml
import appdirs

logger = logging.getLogger(__name__)


srcdir = Path(__file__).resolve().parent.parent
rootdir = srcdir.parent


def load_config() -> MutableMapping[str, Any]:
    filepath = Path(appdirs.user_config_dir("quantifiedme")) / "config.toml"
    if not filepath.exists():
        logger.warning("No config found, falling back to example config")
        filepath = Path(rootdir) / "config.toml.example"
    with open(filepath) as f:
        return toml.load(f)


if __name__ == "__main__":
    load_config()
