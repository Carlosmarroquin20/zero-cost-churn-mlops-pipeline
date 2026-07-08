"""Data preprocessing module.

Clean the raw Telco Customer Churn dataset and split it into stratified
train/test sets ready for model training. Encoding and scaling are
intentionally deferred to the modeling stage (see :mod:`src.features`) so that
transformers are fitted on the training split only, preventing data leakage.

Usage:
    python -m src.data_preprocessing --config config/config.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_ingestion import load_config
from src.data_validation import validate_raw_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def clean_data(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Clean the raw dataframe into a model-ready (but unencoded) frame.

    Steps:
        1. Drop identifier columns with no predictive signal.
        2. Coerce text-encoded numeric columns; blanks become NaN.
        3. Impute the resulting missing values with zero (brand-new customers
           have no accumulated ``TotalCharges``).
        4. Encode the binary target as an integer (positive class -> 1).

    Args:
        df: The raw dataframe.
        config: Parsed project configuration.

    Returns:
        A cleaned copy of the dataframe.
    """
    prep = config["preprocessing"]
    df = df.copy()

    df = df.drop(columns=prep["drop_columns"], errors="ignore")

    for column in prep["numeric_coerce_columns"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.fillna({column: 0.0 for column in prep["numeric_coerce_columns"]})

    target = prep["target"]
    df[target] = (df[target] == prep["positive_class"]).astype(int)

    logger.info("Cleaned data: %d rows, %d columns.", df.shape[0], df.shape[1])
    return df


def split_data(
    df: pd.DataFrame, config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a cleaned dataframe into train and test sets.

    Args:
        df: The cleaned dataframe (target already encoded).
        config: Parsed project configuration.

    Returns:
        A ``(train_df, test_df)`` tuple.
    """
    prep = config["preprocessing"]
    target = prep["target"]
    stratify = df[target] if prep.get("stratify", True) else None

    train_df, test_df = train_test_split(
        df,
        test_size=prep["test_size"],
        random_state=prep["random_state"],
        stratify=stratify,
    )
    logger.info("Split data: %d train / %d test rows.", len(train_df), len(test_df))
    return train_df, test_df


def run(config: Dict[str, Any]) -> Tuple[Path, Path]:
    """Execute the full preprocessing step end to end.

    Loads the raw CSV, validates it, cleans it, splits it, and writes the
    train/test CSVs to the processed data directory.

    Args:
        config: Parsed project configuration.

    Returns:
        A ``(train_path, test_path)`` tuple of the written files.
    """
    data_cfg = config["data"]
    raw_path = Path(data_cfg["raw_dir"]) / data_cfg["raw_filename"]

    logger.info("Loading raw data from %s", raw_path)
    raw_df = pd.read_csv(raw_path)

    validate_raw_data(raw_df)
    cleaned_df = clean_data(raw_df, config)
    train_df, test_df = split_data(cleaned_df, config)

    processed_dir = Path(data_cfg["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_path = processed_dir / "train.csv"
    test_path = processed_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    logger.info("Saved processed data to %s and %s", train_path, test_path)
    return train_path, test_path


def main() -> None:
    """Command-line entry point for the preprocessing step."""
    parser = argparse.ArgumentParser(
        description="Clean and split the Telco Churn dataset."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config.yaml"),
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run(config)


if __name__ == "__main__":
    main()
