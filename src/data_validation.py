"""Data validation module.

Define the expected schema (a data contract) for the raw Telco Customer Churn
dataset and provide a helper to validate a DataFrame against it. Validation
fails fast on schema drift, unexpected categories, or out-of-range values,
which protects every downstream stage of the pipeline.
"""

from __future__ import annotations

import logging
from typing import Final, List

import pandas as pd
from pandera.pandas import Check, Column, DataFrameSchema

logger = logging.getLogger(__name__)

# Categorical domains kept here as the single source of truth for the contract.
_YES_NO: Final[List[str]] = ["Yes", "No"]
_YES_NO_NO_INTERNET: Final[List[str]] = ["Yes", "No", "No internet service"]
_PAYMENT_METHODS: Final[List[str]] = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]

# Schema for the raw dataset exactly as published on Kaggle. Note that
# ``TotalCharges`` is stored as text (blank for brand-new customers), so it is
# validated as a string here and coerced to numeric during preprocessing.
RAW_SCHEMA: Final[DataFrameSchema] = DataFrameSchema(
    {
        "customerID": Column(str, nullable=False, unique=True),
        "gender": Column(str, Check.isin(["Female", "Male"])),
        "SeniorCitizen": Column(int, Check.isin([0, 1])),
        "Partner": Column(str, Check.isin(_YES_NO)),
        "Dependents": Column(str, Check.isin(_YES_NO)),
        "tenure": Column(int, Check.ge(0)),
        "PhoneService": Column(str, Check.isin(_YES_NO)),
        "MultipleLines": Column(str, Check.isin(["Yes", "No", "No phone service"])),
        "InternetService": Column(str, Check.isin(["DSL", "Fiber optic", "No"])),
        "OnlineSecurity": Column(str, Check.isin(_YES_NO_NO_INTERNET)),
        "OnlineBackup": Column(str, Check.isin(_YES_NO_NO_INTERNET)),
        "DeviceProtection": Column(str, Check.isin(_YES_NO_NO_INTERNET)),
        "TechSupport": Column(str, Check.isin(_YES_NO_NO_INTERNET)),
        "StreamingTV": Column(str, Check.isin(_YES_NO_NO_INTERNET)),
        "StreamingMovies": Column(str, Check.isin(_YES_NO_NO_INTERNET)),
        "Contract": Column(str, Check.isin(["Month-to-month", "One year", "Two year"])),
        "PaperlessBilling": Column(str, Check.isin(_YES_NO)),
        "PaymentMethod": Column(str, Check.isin(_PAYMENT_METHODS)),
        "MonthlyCharges": Column(float, Check.gt(0)),
        "TotalCharges": Column(str, nullable=False),
        "Churn": Column(str, Check.isin(_YES_NO)),
    },
    strict=False,
    coerce=False,
)


def validate_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a raw dataframe against the Telco Churn data contract.

    Args:
        df: The raw dataframe to validate.

    Returns:
        The validated dataframe (unchanged when validation succeeds).

    Raises:
        pandera.errors.SchemaError: If the dataframe violates the schema.
    """
    validated = RAW_SCHEMA.validate(df, lazy=False)
    logger.info(
        "Raw data validation passed: %d rows, %d columns.",
        validated.shape[0],
        validated.shape[1],
    )
    return validated
