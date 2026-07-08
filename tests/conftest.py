"""Shared pytest fixtures for the test suite."""

from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest

from src.data_ingestion import load_config


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
def config() -> Dict[str, Any]:
    """The real project configuration, so tests validate it as well."""
    return load_config(Path("config/config.yaml"))
