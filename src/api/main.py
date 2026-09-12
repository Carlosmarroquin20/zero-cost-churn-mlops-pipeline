"""FastAPI application that serves churn predictions.

The model is loaded once at startup. When no model artifact is available the
service still starts, reports itself as unavailable on ``/health``, and answers
prediction requests with HTTP 503 instead of crashing.

Run locally from the repository root with:
    uvicorn src.api.main:app --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Annotated, Any, AsyncIterator, Dict, Optional

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import RedirectResponse

from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerFeatures,
    HealthResponse,
    PredictionResponse,
    PredictionResult,
    to_feature_frame,
)
from src.config import load_config
from src.predict import ChurnPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

router = APIRouter()


def get_predictor(request: Request) -> ChurnPredictor:
    """Return the loaded predictor, failing with HTTP 503 when there is none.

    Args:
        request: The incoming request, used to reach the application state.

    Returns:
        The predictor loaded at startup.

    Raises:
        HTTPException: If no model is loaded.
    """
    predictor: Optional[ChurnPredictor] = request.app.state.predictor
    if predictor is None:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE, detail="Model is not loaded."
        )
    return predictor


PredictorDependency = Annotated[ChurnPredictor, Depends(get_predictor)]


@router.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Send visitors of the base URL to the interactive documentation."""
    return RedirectResponse(url="/docs")


@router.get("/health", response_model=HealthResponse)
def health(request: Request, response: Response) -> HealthResponse:
    """Report whether the service is ready to serve predictions."""
    predictor: Optional[ChurnPredictor] = request.app.state.predictor
    if predictor is None:
        response.status_code = HTTPStatus.SERVICE_UNAVAILABLE
        return HealthResponse(status="unavailable", model_loaded=False)
    return HealthResponse(
        status="ok", model_loaded=True, model_version=predictor.version
    )


@router.post("/predict", response_model=PredictionResponse)
def predict(
    customer: CustomerFeatures, predictor: PredictorDependency
) -> PredictionResponse:
    """Score a single customer."""
    (prediction,) = predictor.predict(to_feature_frame([customer]))
    return PredictionResponse(
        churn_probability=prediction.churn_probability,
        will_churn=prediction.will_churn,
        threshold=predictor.threshold,
        model_version=predictor.version,
    )


@router.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(
    payload: BatchPredictionRequest,
    request: Request,
    predictor: PredictorDependency,
) -> BatchPredictionResponse:
    """Score several customers in a single request."""
    max_batch_size: int = request.app.state.max_batch_size
    if len(payload.customers) > max_batch_size:
        raise HTTPException(
            status_code=HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            detail=f"Batch size exceeds the limit of {max_batch_size} customers.",
        )

    predictions = predictor.predict(to_feature_frame(payload.customers))
    return BatchPredictionResponse(
        predictions=[
            PredictionResult(
                churn_probability=prediction.churn_probability,
                will_churn=prediction.will_churn,
            )
            for prediction in predictions
        ],
        threshold=predictor.threshold,
        model_version=predictor.version,
    )


def load_predictor(config: Dict[str, Any]) -> Optional[ChurnPredictor]:
    """Load the model, or return ``None`` when no artifact has been deployed.

    Args:
        config: Parsed project configuration.

    Returns:
        The loaded predictor, or ``None`` if the model file does not exist.
    """
    try:
        return ChurnPredictor.from_config(config)
    except FileNotFoundError as error:
        logger.error("%s Predictions are disabled until a model is deployed.", error)
        return None


def create_app(config: Optional[Dict[str, Any]] = None) -> FastAPI:
    """Build the inference API application.

    Args:
        config: Parsed project configuration. Defaults to the configuration
            file at the default path.

    Returns:
        The FastAPI application. The model is loaded when the app starts.
    """
    app_config = config if config is not None else load_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.predictor = load_predictor(app_config)
        yield
        app.state.predictor = None

    app = FastAPI(
        title="Telco Customer Churn Prediction API",
        description="Scores how likely a telecom customer is to churn.",
        lifespan=lifespan,
    )
    app.state.predictor = None
    app.state.max_batch_size = app_config["api"]["max_batch_size"]
    app.include_router(router)
    return app


app = create_app()
