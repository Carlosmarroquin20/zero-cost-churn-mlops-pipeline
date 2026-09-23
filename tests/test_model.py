"""Unit tests for the model definition module."""

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.model import ESTIMATORS, build_estimator, build_model


def test_build_model_chains_preprocessor_and_classifier(
    config: Dict[str, Any],
) -> None:
    """The pipeline must apply preprocessing before the configured estimator."""
    model = build_model(config)

    assert isinstance(model, Pipeline)
    assert list(model.named_steps) == ["preprocessor", "classifier"]
    expected_cls = ESTIMATORS[config["model"]["name"]]
    assert isinstance(model.named_steps["classifier"], expected_cls)


def test_build_estimator_forwards_params() -> None:
    """Configured keyword arguments must reach the estimator constructor."""
    estimator = build_estimator("random_forest", {"n_estimators": 7})
    assert estimator.n_estimators == 7


def test_build_estimator_rejects_unknown_model() -> None:
    """An unsupported model name should fail fast with a clear message."""
    with pytest.raises(ValueError, match="Unsupported model"):
        build_estimator("not_a_model")


@pytest.mark.parametrize("model_name", sorted(ESTIMATORS))
def test_every_registered_model_predicts_probabilities(
    model_name: str,
    config: Dict[str, Any],
    synthetic_splits: Tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Each supported estimator must fit and return valid churn probabilities."""
    train_df, test_df = synthetic_splits
    target = config["preprocessing"]["target"]
    config["model"] = {"name": model_name, "params": {}}

    model = build_model(config)
    model.fit(train_df.drop(columns=[target]), train_df[target])
    probabilities = model.predict_proba(test_df.drop(columns=[target]))

    assert probabilities.shape == (len(test_df), 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
