"""Prediction module.

Load the trained churn pipeline and score customers with it. This module only
depends on the serialized scikit-learn pipeline, not on training, tracking, or
data versioning libraries, so the inference API and batch jobs can reuse it.

Model files are deserialized with joblib (pickle), which can execute arbitrary
code: only load artifacts produced by this project's own training pipeline.
"""

from __future__ import annotations

import hashlib
import io
import logging
from dataclasses import dataclass
from typing import Any, Dict, List

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import get_model_path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Prediction:
    """Churn score for a single customer.

    Attributes:
        churn_probability: Predicted probability that the customer churns.
        will_churn: Whether the probability reaches the decision threshold.
    """

    churn_probability: float
    will_churn: bool


@dataclass(frozen=True)
class ChurnPredictor:
    """A trained churn pipeline together with its decision settings.

    Attributes:
        pipeline: Fitted preprocessing and classifier pipeline.
        threshold: Probability at or above which a customer is flagged as churn.
        version: MD5 fingerprint of the loaded model file. It matches the hash
            DVC records for the artifact, which makes served models traceable.
    """

    pipeline: Pipeline
    threshold: float
    version: str

    def __post_init__(self) -> None:
        """Validate the decision threshold.

        Raises:
            ValueError: If the threshold is outside the ``[0, 1]`` interval.
        """
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be within [0, 1], got {self.threshold}.")

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> ChurnPredictor:
        """Load the model artifact referenced by the configuration.

        The file is read once, so the fingerprint always describes exactly the
        bytes that were deserialized.

        Args:
            config: Parsed project configuration.

        Returns:
            A ready-to-use predictor.

        Raises:
            FileNotFoundError: If the model artifact does not exist.
            TypeError: If the artifact cannot produce probabilities.
        """
        model_path = get_model_path(config)
        if not model_path.is_file():
            raise FileNotFoundError(f"Model artifact not found: {model_path}")

        payload = model_path.read_bytes()
        pipeline = joblib.load(io.BytesIO(payload))
        if not hasattr(pipeline, "predict_proba"):
            raise TypeError(
                f"Model artifact at {model_path} does not support predict_proba."
            )

        version = hashlib.md5(payload, usedforsecurity=False).hexdigest()
        logger.info("Loaded model %s (md5 %s).", model_path, version)
        return cls(
            pipeline=pipeline,
            threshold=config["training"]["decision_threshold"],
            version=version,
        )

    def predict(self, features: pd.DataFrame) -> List[Prediction]:
        """Score customers and flag likely churners.

        Args:
            features: One row per customer with the feature columns used in
                training.

        Returns:
            One prediction per input row, in the same order.
        """
        # Labels are encoded as 0/1, so column 1 holds the churn probability.
        probabilities = self.pipeline.predict_proba(features)[:, 1]
        return [
            Prediction(
                churn_probability=float(probability),
                will_churn=bool(probability >= self.threshold),
            )
            for probability in probabilities
        ]
