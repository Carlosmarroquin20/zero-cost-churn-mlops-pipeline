"""Unit tests for the data validation module."""

import pandas as pd
import pytest
from pandera.errors import SchemaError

from src.data_validation import validate_raw_data


def test_valid_raw_data_passes(raw_df: pd.DataFrame) -> None:
    """A schema-valid dataframe should pass validation unchanged."""
    result = validate_raw_data(raw_df)
    assert len(result) == len(raw_df)


def test_unexpected_category_is_rejected(raw_df: pd.DataFrame) -> None:
    """An out-of-domain categorical value must fail validation."""
    corrupted = raw_df.copy()
    corrupted.loc[0, "gender"] = "Unknown"

    with pytest.raises(SchemaError):
        validate_raw_data(corrupted)


def test_negative_tenure_is_rejected(raw_df: pd.DataFrame) -> None:
    """A negative tenure violates the non-negativity constraint."""
    corrupted = raw_df.copy()
    corrupted.loc[0, "tenure"] = -5

    with pytest.raises(SchemaError):
        validate_raw_data(corrupted)
