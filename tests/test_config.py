"""Unit tests for the configuration module."""

from src.config import DEFAULT_CONFIG_PATH, load_config


def test_load_config_returns_expected_keys() -> None:
    """The configuration file should expose the core data settings."""
    config = load_config(DEFAULT_CONFIG_PATH)

    assert "data" in config
    assert config["data"]["kaggle_dataset"] == "blastchar/telco-customer-churn"
    assert config["data"]["raw_filename"].endswith(".csv")
    assert config["data"]["raw_dir"] == "data/raw"
