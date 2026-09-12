"""Request and response schemas for the churn prediction API.

Request fields use snake_case names for API consumers. Each field declares the
original dataset column as its serialization alias, so a validated request
converts directly into the feature frame the model was trained on. The allowed
categorical values mirror the raw data contract in :mod:`src.data_validation`.
"""

from __future__ import annotations

from typing import Any, Dict, Final, List, Literal, Optional, Sequence

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

YesNo = Literal["Yes", "No"]
InternetAddOn = Literal["Yes", "No", "No internet service"]

# A representative customer record, shown in the interactive API docs.
EXAMPLE_CUSTOMER: Final[Dict[str, Any]] = {
    "gender": "Female",
    "senior_citizen": 0,
    "partner": "Yes",
    "dependents": "No",
    "tenure": 1,
    "phone_service": "No",
    "multiple_lines": "No phone service",
    "internet_service": "DSL",
    "online_security": "No",
    "online_backup": "Yes",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "No",
    "streaming_movies": "No",
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
    "monthly_charges": 29.85,
    "total_charges": 29.85,
}


class CustomerFeatures(BaseModel):
    """Attributes of a single customer, as used by the churn model."""

    model_config = ConfigDict(
        extra="forbid", json_schema_extra={"examples": [EXAMPLE_CUSTOMER]}
    )

    gender: Literal["Female", "Male"] = Field(serialization_alias="gender")
    senior_citizen: Literal[0, 1] = Field(
        serialization_alias="SeniorCitizen",
        description="1 if the customer is a senior citizen, otherwise 0.",
    )
    partner: YesNo = Field(serialization_alias="Partner")
    dependents: YesNo = Field(serialization_alias="Dependents")
    tenure: int = Field(
        ge=0,
        serialization_alias="tenure",
        description="Number of months the customer has stayed with the company.",
    )
    phone_service: YesNo = Field(serialization_alias="PhoneService")
    multiple_lines: Literal["Yes", "No", "No phone service"] = Field(
        serialization_alias="MultipleLines"
    )
    internet_service: Literal["DSL", "Fiber optic", "No"] = Field(
        serialization_alias="InternetService"
    )
    online_security: InternetAddOn = Field(serialization_alias="OnlineSecurity")
    online_backup: InternetAddOn = Field(serialization_alias="OnlineBackup")
    device_protection: InternetAddOn = Field(serialization_alias="DeviceProtection")
    tech_support: InternetAddOn = Field(serialization_alias="TechSupport")
    streaming_tv: InternetAddOn = Field(serialization_alias="StreamingTV")
    streaming_movies: InternetAddOn = Field(serialization_alias="StreamingMovies")
    contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        serialization_alias="Contract"
    )
    paperless_billing: YesNo = Field(serialization_alias="PaperlessBilling")
    payment_method: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = Field(serialization_alias="PaymentMethod")
    monthly_charges: float = Field(
        gt=0,
        serialization_alias="MonthlyCharges",
        description="Amount currently charged to the customer every month.",
    )
    total_charges: float = Field(
        ge=0,
        serialization_alias="TotalCharges",
        description="Total amount charged to the customer so far.",
    )


class BatchPredictionRequest(BaseModel):
    """Request body of the batch prediction endpoint."""

    model_config = ConfigDict(extra="forbid")

    customers: List[CustomerFeatures] = Field(
        min_length=1,
        description="Customers to score. The API enforces a maximum batch size.",
    )


class PredictionResult(BaseModel):
    """Churn score for one customer."""

    churn_probability: float = Field(
        ge=0.0,
        le=1.0,
        description="Predicted probability that the customer churns.",
    )
    will_churn: bool = Field(
        description="True when the probability reaches the decision threshold."
    )


class PredictionResponse(PredictionResult):
    """Response of the single-customer prediction endpoint."""

    threshold: float = Field(description="Decision threshold applied to the score.")
    model_version: str = Field(
        description="MD5 fingerprint of the model that produced the score."
    )


class BatchPredictionResponse(BaseModel):
    """Response of the batch prediction endpoint."""

    predictions: List[PredictionResult] = Field(
        description="One result per customer, in request order."
    )
    threshold: float = Field(description="Decision threshold applied to the scores.")
    model_version: str = Field(
        description="MD5 fingerprint of the model that produced the scores."
    )


class HealthResponse(BaseModel):
    """Service health, including whether a model is ready to serve."""

    status: Literal["ok", "unavailable"]
    model_loaded: bool
    model_version: Optional[str] = None


def to_feature_frame(customers: Sequence[CustomerFeatures]) -> pd.DataFrame:
    """Convert validated customers into the model's feature frame.

    Args:
        customers: Validated customer records.

    Returns:
        One row per customer, with the dataset column names used in training.
    """
    return pd.DataFrame([customer.model_dump(by_alias=True) for customer in customers])
