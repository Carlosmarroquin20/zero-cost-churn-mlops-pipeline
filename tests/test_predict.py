"""Unit tests for the prediction module."""

import dataclasses
import hashlib
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import pandas as pd
import pytest

from src.config import get_model_path
from src.predict import ChurnPredictor


def test_from_config_loads_model_with_fingerprint(
    trained_model_config: Dict[str, Any],
) -> None:
    """The predictor exposes the MD5 of the loaded artifact as its version."""
    predictor = ChurnPredictor.from_config(trained_model_config)

    model_bytes = get_model_path(trained_model_config).read_bytes()
    assert predictor.version == hashlib.md5(model_bytes).hexdigest()
    expected_threshold = trained_model_config["training"]["decision_threshold"]
    assert predictor.threshold == expected_threshold


def test_from_config_fails_when_model_is_missing(
    config: Dict[str, Any], tmp_path: Path
) -> None:
    """A missing artifact must raise a clear error instead of a pickle failure."""
    config["paths"]["models_dir"] = str(tmp_path / "missing")

    with pytest.raises(FileNotFoundError, match="Model artifact not found"):
        ChurnPredictor.from_config(config)


def test_from_config_rejects_artifacts_without_probabilities(
    config: Dict[str, Any], tmp_path: Path
) -> None:
    """Only artifacts that can produce churn probabilities are accepted."""
    config["paths"]["models_dir"] = str(tmp_path)
    joblib.dump({"not": "a model"}, get_model_path(config))

    with pytest.raises(TypeError, match="predict_proba"):
        ChurnPredictor.from_config(config)


def test_predict_matches_pipeline_probabilities(
    trained_model_config: Dict[str, Any],
    synthetic_splits: Tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """Predictions keep the row order and agree with the underlying pipeline."""
    _, test_df = synthetic_splits
    target = trained_model_config["preprocessing"]["target"]
    features = test_df.drop(columns=[target])
    predictor = ChurnPredictor.from_config(trained_model_config)

    predictions = predictor.predict(features)

    expected = predictor.pipeline.predict_proba(features)[:, 1]
    assert [p.churn_probability for p in predictions] == pytest.approx(
        expected.tolist()
    )
    assert all(
        p.will_churn == (p.churn_probability >= predictor.threshold)
        for p in predictions
    )


@pytest.mark.parametrize(("threshold", "expected_flag"), [(0.0, True), (1.0, False)])
def test_threshold_controls_churn_flag(
    threshold: float,
    expected_flag: bool,
    trained_model_config: Dict[str, Any],
    synthetic_splits: Tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """A zero threshold flags every customer; a threshold of one flags none."""
    _, test_df = synthetic_splits
    target = trained_model_config["preprocessing"]["target"]
    predictor = dataclasses.replace(
        ChurnPredictor.from_config(trained_model_config), threshold=threshold
    )

    predictions = predictor.predict(test_df.drop(columns=[target]))

    assert {p.will_churn for p in predictions} == {expected_flag}


def test_invalid_threshold_is_rejected(trained_model_config: Dict[str, Any]) -> None:
    """Thresholds outside [0, 1] are meaningless for probabilities."""
    predictor = ChurnPredictor.from_config(trained_model_config)

    with pytest.raises(ValueError, match="threshold"):
        dataclasses.replace(predictor, threshold=1.5)
