from typing import MutableMapping, Any
from pathlib import Path
import logging

import toml
import appdirs

logger = logging.getLogger(__name__)


def load_config() -> MutableMapping[str, Any]:
    filepath = Path(appdirs.user_config_dir("quantifiedme")) / "config.toml"
    if not filepath.exists():
        logger.warning("No config found, falling back to example config")
        filepath = Path(appdirs.user_config_dir("quantifiedme")) / "config.toml.example"
    with open(filepath) as f:
        return toml.load(f)
