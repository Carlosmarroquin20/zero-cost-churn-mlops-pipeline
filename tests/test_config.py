"""Unit tests for the configuration module."""

from pathlib import Path
from typing import Any, Dict

from src.config import DEFAULT_CONFIG_PATH, get_model_path, load_config


def test_load_config_returns_expected_keys() -> None:
    """The configuration file should expose the core data settings."""
    config = load_config(DEFAULT_CONFIG_PATH)

    assert "data" in config
    assert config["data"]["kaggle_dataset"] == "blastchar/telco-customer-churn"
    assert config["data"]["raw_filename"].endswith(".csv")
    assert config["data"]["raw_dir"] == "data/raw"


def test_get_model_path_joins_directory_and_file_name(
    config: Dict[str, Any],
) -> None:
    """The model path combines the models directory and the artifact name."""
    config["paths"]["models_dir"] = "artifacts"
    config["model"]["artifact_filename"] = "churn.joblib"

    assert get_model_path(config) == Path("artifacts") / "churn.joblib"
