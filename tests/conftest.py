"""Shared pytest fixtures for the test suite."""

from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import joblib
import mlflow
import numpy as np
import pandas as pd
import pytest

from src.config import DEFAULT_CONFIG_PATH, get_model_path, load_config
from src.data_preprocessing import clean_data, split_data
from src.model import build_model

# Environment variables that could route test runs to a real MLflow server.
_MLFLOW_ENV_VARS: Tuple[str, ...] = (
    "MLFLOW_TRACKING_URI",
    "MLFLOW_TRACKING_USERNAME",
    "MLFLOW_TRACKING_PASSWORD",
)

# Allowed values per categorical column, mirroring the raw data contract.
_CATEGORY_DOMAINS: Dict[str, List[str]] = {
    "gender": ["Female", "Male"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}


@pytest.fixture(autouse=True)
def _isolate_mlflow_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep every test away from real MLflow servers and credentials."""
    for name in _MLFLOW_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _make_raw_row(customer_id: str, churn: str, total_charges: str = "100.5") -> dict:
    """Build a single schema-valid raw record for the Telco dataset."""
    return {
        "customerID": customer_id,
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": total_charges,
        "Churn": churn,
    }


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """A small but schema-valid raw dataframe with both target classes.

    Includes one brand-new customer whose ``TotalCharges`` is blank, mirroring
    the well-known data-quality quirk of the real dataset.
    """
    rows = [
        _make_raw_row(f"ID-{i:04d}", "Yes" if i % 2 == 0 else "No") for i in range(20)
    ]
    rows.append(_make_raw_row("ID-9999", "No", total_charges=" "))
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_raw_df() -> pd.DataFrame:
    """A larger, varied, schema-valid raw dataframe for model training tests.

    Churn is far more likely for month-to-month contracts, which gives models a
    learnable signal. The generator is seeded, so the data is deterministic.
    """
    rng = np.random.default_rng(seed=7)
    n_rows = 200

    df = pd.DataFrame(
        {
            column: rng.choice(values, size=n_rows)
            for column, values in _CATEGORY_DOMAINS.items()
        }
    )
    df.insert(0, "customerID", [f"SYN-{i:04d}" for i in range(n_rows)])
    df["SeniorCitizen"] = rng.integers(0, 2, size=n_rows)
    df["tenure"] = rng.integers(0, 73, size=n_rows)
    df["MonthlyCharges"] = rng.uniform(18.0, 120.0, size=n_rows).round(2)
    df["TotalCharges"] = (df["tenure"] * df["MonthlyCharges"]).round(2).astype(str)

    churn_probability = np.where(df["Contract"] == "Month-to-month", 0.7, 0.1)
    df["Churn"] = np.where(rng.random(n_rows) < churn_probability, "Yes", "No")
    return df


@pytest.fixture
def config() -> Dict[str, Any]:
    """The real project configuration, so tests validate it as well."""
    return load_config(DEFAULT_CONFIG_PATH)


@pytest.fixture
def local_tracking_config(
    config: Dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Dict[str, Any]]:
    """Configuration that records MLflow runs in a temporary local store.

    The working directory moves to ``tmp_path`` because MLflow stores local
    artifacts relative to it, which also keeps relative output paths from the
    configuration inside the temporary directory. MLflow exports the tracking
    URI to the environment, so its global state is reset afterwards.
    """
    monkeypatch.chdir(tmp_path)
    db_path = (tmp_path / "mlflow.db").as_posix()
    config["mlflow"]["tracking_uri"] = f"sqlite:///{db_path}"
    yield config
    mlflow.set_tracking_uri(None)


@pytest.fixture
def synthetic_splits(
    synthetic_raw_df: pd.DataFrame, config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Cleaned train/test splits built with the real preprocessing steps."""
    return split_data(clean_data(synthetic_raw_df, config), config)


@pytest.fixture
def trained_model_config(
    config: Dict[str, Any],
    tmp_path: Path,
    synthetic_splits: Tuple[pd.DataFrame, pd.DataFrame],
) -> Dict[str, Any]:
    """Configuration pointing to a model trained on the synthetic data.

    The model is built with the real model factory and saved under ``tmp_path``,
    so prediction and API tests never need the real dataset.
    """
    train_df, _ = synthetic_splits
    target = config["preprocessing"]["target"]
    model = build_model(config)
    model.fit(train_df.drop(columns=[target]), train_df[target])

    config["paths"]["models_dir"] = str(tmp_path / "models")
    model_path = get_model_path(config)
    model_path.parent.mkdir(parents=True)
    joblib.dump(model, model_path)
    return config
