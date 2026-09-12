"""Configuration module.

Load the project's YAML configuration. Every pipeline stage and the inference
API read paths and parameters from this single file instead of hard-coding them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Final

import yaml

DEFAULT_CONFIG_PATH: Final[Path] = Path("config/config.yaml")


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load and parse the YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        The parsed configuration as a dictionary.
    """
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)
