"""Unit tests for the prediction API schemas."""

from typing import Any, Dict, Literal, get_args, get_origin

import pytest
from pydantic import ValidationError

from src.api.schemas import (
    EXAMPLE_CUSTOMER,
    BatchPredictionRequest,
    CustomerFeatures,
    to_feature_frame,
)
from src.data_validation import RAW_SCHEMA


def test_feature_aliases_match_model_features(config: Dict[str, Any]) -> None:
    """Every request field must map to exactly one feature the model uses."""
    aliases = {
        field.serialization_alias for field in CustomerFeatures.model_fields.values()
    }
    features = config["features"]

    assert aliases == set(features["numeric"]) | set(features["categorical"])


def test_allowed_values_match_raw_data_contract() -> None:
    """Categorical domains must not drift from the Pandera data contract."""
    for field in CustomerFeatures.model_fields.values():
        if get_origin(field.annotation) is not Literal:
            continue
        column = RAW_SCHEMA.columns[field.serialization_alias]
        contract_values = column.checks[0].statistics["allowed_values"]

        assert set(get_args(field.annotation)) == set(contract_values), column.name


def test_example_converts_to_feature_frame() -> None:
    """A valid request becomes a one-row frame with the training column names."""
    customer = CustomerFeatures.model_validate(EXAMPLE_CUSTOMER)

    frame = to_feature_frame([customer])

    assert len(frame) == 1
    assert frame.loc[0, "MonthlyCharges"] == EXAMPLE_CUSTOMER["monthly_charges"]
    assert frame.loc[0, "SeniorCitizen"] == EXAMPLE_CUSTOMER["senior_citizen"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"contract": "Weekly"},
        {"tenure": -1},
        {"monthly_charges": 0},
        {"senior_citizen": 2},
        {"customerID": "0000-TEST"},
    ],
    ids=["unknown-category", "negative-tenure", "zero-charges", "bad-flag", "extra"],
)
def test_invalid_customer_is_rejected(overrides: Dict[str, Any]) -> None:
    """Out-of-contract values and unexpected fields must fail validation."""
    with pytest.raises(ValidationError):
        CustomerFeatures.model_validate({**EXAMPLE_CUSTOMER, **overrides})


def test_missing_field_is_rejected() -> None:
    """All features are required; the model cannot score partial records."""
    payload = {k: v for k, v in EXAMPLE_CUSTOMER.items() if k != "tenure"}

    with pytest.raises(ValidationError):
        CustomerFeatures.model_validate(payload)


def test_batch_request_requires_at_least_one_customer() -> None:
    """An empty batch is a client error, not an empty prediction list."""
    with pytest.raises(ValidationError):
        BatchPredictionRequest.model_validate({"customers": []})
