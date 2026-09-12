"""Unit tests for the data preprocessing module."""

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.data_preprocessing import clean_data, get_split_paths, run, split_data


def test_clean_data_drops_identifier(
    raw_df: pd.DataFrame, config: Dict[str, Any]
) -> None:
    """Identifier columns should be removed during cleaning."""
    cleaned = clean_data(raw_df, config)
    assert "customerID" not in cleaned.columns


def test_clean_data_encodes_target(
    raw_df: pd.DataFrame, config: Dict[str, Any]
) -> None:
    """The target must be encoded as integers in {0, 1}."""
    cleaned = clean_data(raw_df, config)
    target = config["preprocessing"]["target"]
    assert set(cleaned[target].unique()).issubset({0, 1})


def test_clean_data_coerces_blank_total_charges(
    raw_df: pd.DataFrame, config: Dict[str, Any]
) -> None:
    """The blank TotalCharges value must become a finite numeric value."""
    cleaned = clean_data(raw_df, config)
    assert pd.api.types.is_numeric_dtype(cleaned["TotalCharges"])
    assert cleaned["TotalCharges"].notna().all()


def test_split_data_preserves_rows_and_classes(
    raw_df: pd.DataFrame, config: Dict[str, Any]
) -> None:
    """The split should preserve all rows and keep both classes in training."""
    cleaned = clean_data(raw_df, config)
    train_df, test_df = split_data(cleaned, config)

    assert len(train_df) + len(test_df) == len(cleaned)
    assert not train_df.empty
    assert not test_df.empty
    assert set(train_df[config["preprocessing"]["target"]].unique()) == {0, 1}


def test_run_writes_processed_splits(
    tmp_path: Path, raw_df: pd.DataFrame, config: Dict[str, Any]
) -> None:
    """The entry point should persist both splits at the configured paths."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_df.to_csv(raw_dir / config["data"]["raw_filename"], index=False)
    config["data"]["raw_dir"] = str(raw_dir)
    config["data"]["processed_dir"] = str(tmp_path / "processed")

    train_path, test_path = run(config)

    assert (train_path, test_path) == get_split_paths(config)
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    assert len(train_df) + len(test_df) == len(raw_df)
    assert "customerID" not in train_df.columns
