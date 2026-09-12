"""Unit tests for the model training module."""

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import mlflow
import pandas as pd
import pytest

from src.data_preprocessing import get_split_paths
from src.train import collect_run_params, run, split_features_target, train_and_evaluate


def test_split_features_target_separates_label(
    synthetic_splits: Tuple[pd.DataFrame, pd.DataFrame], config: Dict[str, Any]
) -> None:
    """The target column must be removed from the features and returned alone."""
    train_df, _ = synthetic_splits
    target = config["preprocessing"]["target"]

    features, labels = split_features_target(train_df, target)

    assert target not in features.columns
    assert labels.name == target
    assert len(features) == len(labels) == len(train_df)


def test_train_and_evaluate_reports_cv_and_test_metrics(
    synthetic_splits: Tuple[pd.DataFrame, pd.DataFrame], config: Dict[str, Any]
) -> None:
    """Training yields a fitted model plus cross-validation and test metrics."""
    train_df, test_df = synthetic_splits
    target = config["preprocessing"]["target"]

    model, metrics = train_and_evaluate(config, train_df, test_df)

    assert {"cv_roc_auc_mean", "cv_roc_auc_std", "test_roc_auc", "test_f1"} <= set(
        metrics
    )
    assert all(0.0 <= value <= 1.0 for value in metrics.values())
    # The synthetic data carries a real churn signal, so the model beats chance.
    assert metrics["test_roc_auc"] > 0.5
    assert len(model.predict(test_df.drop(columns=[target]))) == len(test_df)


def test_cross_validation_can_be_disabled(
    synthetic_splits: Tuple[pd.DataFrame, pd.DataFrame], config: Dict[str, Any]
) -> None:
    """Setting cv_folds below 2 skips cross-validation entirely."""
    config["training"]["cv_folds"] = 0
    train_df, test_df = synthetic_splits

    _, metrics = train_and_evaluate(config, train_df, test_df)

    assert not any(name.startswith("cv_") for name in metrics)


def test_collect_run_params_flattens_model_params(config: Dict[str, Any]) -> None:
    """Estimator parameters are logged with a ``model__`` prefix."""
    params = collect_run_params(config)

    assert params["model_name"] == config["model"]["name"]
    for name, value in config["model"]["params"].items():
        assert params[f"model__{name}"] == value


def test_run_saves_outputs_and_logs_to_mlflow(
    local_tracking_config: Dict[str, Any],
    synthetic_splits: Tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    """The entry point persists the model and metrics and records an MLflow run."""
    config = local_tracking_config
    train_df, test_df = synthetic_splits
    train_path, test_path = get_split_paths(config)
    train_path.parent.mkdir(parents=True)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    metrics = run(config)

    models_dir = Path(config["paths"]["models_dir"])
    model = joblib.load(models_dir / config["model"]["artifact_filename"])
    assert hasattr(model, "predict_proba")
    reports_dir = Path(config["paths"]["reports_dir"])
    metrics_file = reports_dir / config["training"]["metrics_filename"]
    assert json.loads(metrics_file.read_text(encoding="utf-8")) == pytest.approx(
        metrics
    )

    runs = mlflow.search_runs(experiment_names=[config["mlflow"]["experiment_name"]])
    assert len(runs) == 1
    assert runs.loc[0, "params.model_name"] == config["model"]["name"]
    assert runs.loc[0, "metrics.test_roc_auc"] == pytest.approx(metrics["test_roc_auc"])
    artifacts = mlflow.MlflowClient().list_artifacts(runs.loc[0, "run_id"], "model")
    assert [artifact.path for artifact in artifacts] == [
        f"model/{config['model']['artifact_filename']}"
    ]
