import logging
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

import platformdirs
import toml

logger = logging.getLogger(__name__)


srcdir = Path(__file__).resolve().parent.parent
rootdir = srcdir.parent


def _get_config_path(use_example=False) -> Path:
    if use_example:
        return Path(rootdir) / "config.example.toml"
    return Path(platformdirs.user_config_dir("quantifiedme")) / "config.toml"


def load_config(use_example=False) -> MutableMapping[str, Any]:
    # fallback default
    filepath = _get_config_path(use_example=True)

    if (config_path := _get_config_path(use_example)).exists():
        filepath = config_path
    else:
        logger.warning("No config found, falling back to example config")

    with open(filepath) as f:
        logger.debug("Loading config from %s", filepath)
        return toml.load(f)


def has_config() -> bool:
    filepath = _get_config_path()
    return filepath.exists()


if __name__ == "__main__":
    load_config()
