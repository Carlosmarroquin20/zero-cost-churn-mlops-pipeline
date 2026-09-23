"""Tests for the FastAPI inference service."""

from pathlib import Path
from typing import Any, Dict, Iterator

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.schemas import EXAMPLE_CUSTOMER, CustomerFeatures, to_feature_frame
from src.predict import ChurnPredictor

LOYAL_CUSTOMER: Dict[str, Any] = {
    **EXAMPLE_CUSTOMER,
    "contract": "Two year",
    "tenure": 70,
    "total_charges": 2089.5,
}
FIBER_CUSTOMER: Dict[str, Any] = {
    **EXAMPLE_CUSTOMER,
    "internet_service": "Fiber optic",
    "monthly_charges": 104.8,
    "total_charges": 104.8,
}


@pytest.fixture
def client(trained_model_config: Dict[str, Any]) -> Iterator[TestClient]:
    """API client backed by a model trained on the synthetic data."""
    with TestClient(create_app(trained_model_config)) as test_client:
        yield test_client


@pytest.fixture
def client_without_model(
    config: Dict[str, Any], tmp_path: Path
) -> Iterator[TestClient]:
    """API client whose configuration points to a missing model artifact."""
    config["paths"]["models_dir"] = str(tmp_path / "missing")
    with TestClient(create_app(config)) as test_client:
        yield test_client


def test_root_redirects_to_docs(client: TestClient) -> None:
    """The base URL leads visitors to the interactive documentation."""
    response = client.get("/", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


def test_health_reports_loaded_model(
    client: TestClient, trained_model_config: Dict[str, Any]
) -> None:
    """A loaded model makes the service healthy and exposes its version."""
    response = client.get("/health")

    assert response.status_code == 200
    expected_version = ChurnPredictor.from_config(trained_model_config).version
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
        "model_version": expected_version,
    }


def test_health_reports_missing_model(client_without_model: TestClient) -> None:
    """Without a model the service is up but reports itself as unavailable."""
    response = client_without_model.get("/health")

    assert response.status_code == 503
    assert response.json() == {
        "status": "unavailable",
        "model_loaded": False,
        "model_version": None,
    }


def test_predict_matches_model_output(
    client: TestClient, trained_model_config: Dict[str, Any]
) -> None:
    """The endpoint returns exactly what the underlying predictor computes."""
    response = client.post("/predict", json=EXAMPLE_CUSTOMER)

    assert response.status_code == 200
    body = response.json()
    predictor = ChurnPredictor.from_config(trained_model_config)
    customer = CustomerFeatures.model_validate(EXAMPLE_CUSTOMER)
    (expected,) = predictor.predict(to_feature_frame([customer]))
    assert body["churn_probability"] == pytest.approx(expected.churn_probability)
    assert body["will_churn"] == expected.will_churn
    assert body["threshold"] == predictor.threshold
    assert body["model_version"] == predictor.version


def test_predict_rejects_invalid_payload(client: TestClient) -> None:
    """Payloads that break the data contract are rejected before scoring."""
    response = client.post("/predict", json={**EXAMPLE_CUSTOMER, "contract": "Weekly"})

    assert response.status_code == 422


def test_predict_is_unavailable_without_model(
    client_without_model: TestClient,
) -> None:
    """Prediction requests fail fast with 503 when no model is loaded."""
    response = client_without_model.post("/predict", json=EXAMPLE_CUSTOMER)

    assert response.status_code == 503
    assert response.json() == {"detail": "Model is not loaded."}


def test_batch_predict_preserves_request_order(client: TestClient) -> None:
    """Batch results line up with the customers in the request."""
    customers = [EXAMPLE_CUSTOMER, LOYAL_CUSTOMER, FIBER_CUSTOMER]

    response = client.post("/predict/batch", json={"customers": customers})

    assert response.status_code == 200
    batch_scores = [p["churn_probability"] for p in response.json()["predictions"]]
    single_scores = [
        client.post("/predict", json=customer).json()["churn_probability"]
        for customer in customers
    ]
    assert batch_scores == pytest.approx(single_scores)


def test_batch_predict_enforces_size_limit(
    trained_model_config: Dict[str, Any],
) -> None:
    """Batches larger than the configured limit are refused."""
    trained_model_config["api"]["max_batch_size"] = 2
    payload = {"customers": [EXAMPLE_CUSTOMER] * 3}

    with TestClient(create_app(trained_model_config)) as limited_client:
        response = limited_client.post("/predict/batch", json=payload)

    assert response.status_code == 413


def test_openapi_schema_lists_endpoints(client: TestClient) -> None:
    """The OpenAPI document exposes every public endpoint."""
    response = client.get("/openapi.json")

    assert response.status_code == 200
    assert {"/health", "/predict", "/predict/batch"} <= set(response.json()["paths"])
