"""Unit tests for the feature engineering module."""

from typing import Any, Dict

import numpy as np
import pandas as pd

from src.data_preprocessing import clean_data
from src.features import build_preprocessor


def test_preprocessor_outputs_finite_numeric_matrix(
    raw_df: pd.DataFrame, config: Dict[str, Any]
) -> None:
    """Fitting the preprocessor yields one finite numeric row per sample."""
    cleaned = clean_data(raw_df, config)
    target = config["preprocessing"]["target"]
    features = cleaned.drop(columns=[target])

    preprocessor = build_preprocessor(config)
    transformed = preprocessor.fit_transform(features)

    dense = transformed.toarray() if hasattr(transformed, "toarray") else transformed
    assert dense.shape[0] == len(cleaned)
    assert np.isfinite(dense).all()
