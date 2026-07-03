"""Unit tests for the data ingestion module."""

from pathlib import Path

from src.data_ingestion import load_config


def test_load_config_returns_expected_keys() -> None:
    """The configuration file should expose the core data settings."""
    config = load_config(Path("config/config.yaml"))

    assert "data" in config
    assert config["data"]["kaggle_dataset"] == "blastchar/telco-customer-churn"
    assert config["data"]["raw_filename"].endswith(".csv")
    assert config["data"]["raw_dir"] == "data/raw"
