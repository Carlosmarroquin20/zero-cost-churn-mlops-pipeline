"""Model training module.

Fit the churn model on the processed training split, evaluate it on the
held-out test split, persist the fitted pipeline and its metrics, and record
the run (parameters, metrics, and artifacts) in MLflow.

Usage:
    python -m src.train --config config/config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import mlflow
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.data_ingestion import load_config
from src.data_preprocessing import get_split_paths
from src.evaluation import compute_classification_metrics
from src.model import build_model
from src.tracking import configure_tracking

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def split_features_target(
    df: pd.DataFrame, target: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate the feature matrix from the target column.

    Args:
        df: A processed dataframe that includes the target column.
        target: Name of the target column.

    Returns:
        A ``(features, labels)`` tuple.
    """
    return df.drop(columns=[target]), df[target]


def cross_validate(
    model: Pipeline,
    features: pd.DataFrame,
    labels: pd.Series,
    config: Dict[str, Any],
) -> Dict[str, float]:
    """Estimate generalization with stratified k-fold CV on the training split.

    Args:
        model: The unfitted model pipeline (cloned for every fold).
        features: Training features.
        labels: Training labels.
        config: Parsed project configuration.

    Returns:
        Mean and standard deviation of the ROC AUC across folds, or an empty
        mapping when cross-validation is disabled (``cv_folds`` below 2).
    """
    folds = config["training"]["cv_folds"]
    if folds < 2:
        return {}

    splitter = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=config["preprocessing"]["random_state"],
    )
    scores = cross_val_score(model, features, labels, cv=splitter, scoring="roc_auc")
    return {
        "cv_roc_auc_mean": float(scores.mean()),
        "cv_roc_auc_std": float(scores.std()),
    }


def train_and_evaluate(
    config: Dict[str, Any], train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[Pipeline, Dict[str, float]]:
    """Fit the model on the training split and score it on the test split.

    Args:
        config: Parsed project configuration.
        train_df: Processed training split, including the target column.
        test_df: Processed test split, including the target column.

    Returns:
        The fitted pipeline and a mapping of metric name to value. Test metrics
        are prefixed with ``test_`` and cross-validation metrics with ``cv_``.
    """
    target = config["preprocessing"]["target"]
    x_train, y_train = split_features_target(train_df, target)
    x_test, y_test = split_features_target(test_df, target)

    model = build_model(config)
    metrics = cross_validate(model, x_train, y_train, config)

    model.fit(x_train, y_train)
    # Labels are encoded as 0/1, so column 1 holds the churn probability.
    churn_probability = model.predict_proba(x_test)[:, 1]
    test_metrics = compute_classification_metrics(
        y_test,
        churn_probability,
        threshold=config["training"]["decision_threshold"],
    )
    metrics.update({f"test_{name}": value for name, value in test_metrics.items()})

    logger.info("Test ROC AUC: %.4f", metrics["test_roc_auc"])
    return model, metrics


def save_outputs(
    model: Pipeline, metrics: Dict[str, float], config: Dict[str, Any]
) -> Tuple[Path, Path]:
    """Persist the fitted pipeline and its metrics report.

    Args:
        model: The fitted model pipeline.
        metrics: Metrics computed for the model.
        config: Parsed project configuration.

    Returns:
        A ``(model_path, metrics_path)`` tuple of the written files.
    """
    model_path = (
        Path(config["paths"]["models_dir"]) / config["model"]["artifact_filename"]
    )
    metrics_path = (
        Path(config["paths"]["reports_dir"]) / config["training"]["metrics_filename"]
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    logger.info("Saved model to %s and metrics to %s", model_path, metrics_path)
    return model_path, metrics_path


def collect_run_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """Gather the parameters that define a training run.

    Args:
        config: Parsed project configuration.

    Returns:
        A flat mapping of parameter name to value, suitable for MLflow.
    """
    model_cfg = config["model"]
    params: Dict[str, Any] = {
        "model_name": model_cfg["name"],
        "test_size": config["preprocessing"]["test_size"],
        "random_state": config["preprocessing"]["random_state"],
        "cv_folds": config["training"]["cv_folds"],
        "decision_threshold": config["training"]["decision_threshold"],
    }
    estimator_params = model_cfg.get("params") or {}
    params.update({f"model__{name}": value for name, value in estimator_params.items()})
    return params


def run(config: Dict[str, Any]) -> Dict[str, float]:
    """Execute the full training step end to end.

    Loads the processed splits, trains and evaluates the model, saves the
    outputs, and records parameters, metrics, and artifacts in MLflow.

    Args:
        config: Parsed project configuration.

    Returns:
        The metrics computed for the trained model.
    """
    train_path, test_path = get_split_paths(config)
    logger.info("Loading processed splits from %s and %s", train_path, test_path)
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    configure_tracking(config)
    with mlflow.start_run():
        mlflow.log_params(collect_run_params(config))
        model, metrics = train_and_evaluate(config, train_df, test_df)
        model_path, metrics_path = save_outputs(model, metrics, config)

        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(model_path), artifact_path="model")
        mlflow.log_artifact(str(metrics_path))
    return metrics


def main() -> None:
    """Command-line entry point for the training step."""
    parser = argparse.ArgumentParser(description="Train and evaluate the churn model.")
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
