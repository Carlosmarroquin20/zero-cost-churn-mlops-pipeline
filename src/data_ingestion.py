"""Data ingestion module.

Download the Telco Customer Churn dataset from Kaggle using the Kaggle API
and store it under ``data/raw`` with a normalized file name. The resulting
file is intended to be versioned with DVC (not Git).

Usage:
    python -m src.data_ingestion --config config/config.yaml
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from typing import Any, Dict

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load and parse the YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        The parsed configuration as a dictionary.
    """
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def download_dataset(config: Dict[str, Any]) -> Path:
    """Download the Telco Churn dataset from Kaggle and normalize its name.

    The Kaggle client is imported lazily so this module can be imported
    (for example, by the test suite) without requiring valid Kaggle
    credentials to be present at import time.

    Args:
        config: Parsed project configuration.

    Returns:
        Path to the stored raw CSV file.

    Raises:
        FileNotFoundError: If the expected source file is missing after
            the download completes.
    """
    from kaggle.api.kaggle_api_extended import KaggleApi

    data_cfg = config["data"]
    raw_dir = Path(data_cfg["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    logger.info("Downloading dataset '%s' from Kaggle...", data_cfg["kaggle_dataset"])
    api.dataset_download_files(
        data_cfg["kaggle_dataset"],
        path=str(raw_dir),
        unzip=True,
    )

    source_file = raw_dir / data_cfg["source_filename"]
    target_file = raw_dir / data_cfg["raw_filename"]

    if not source_file.exists():
        raise FileNotFoundError(
            f"Expected source file not found after download: {source_file}"
        )

    shutil.move(str(source_file), str(target_file))
    logger.info("Raw dataset stored at: %s", target_file)
    return target_file


def main() -> None:
    """Command-line entry point for the data ingestion step."""
    parser = argparse.ArgumentParser(
        description="Download the Telco Customer Churn dataset from Kaggle."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    download_dataset(config)


if __name__ == "__main__":
    main()
