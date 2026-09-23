"""Model definition module.

Assemble the full, servable scikit-learn pipeline: the feature preprocessor from
:mod:`src.features` followed by a classifier selected from a small registry of
supported estimators. Keeping both steps in a single
:class:`~sklearn.pipeline.Pipeline` guarantees that exactly the same
transformations are applied at training and inference time.
"""

from __future__ import annotations

from typing import Any, Dict, Final, Optional, Type

from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.features import build_preprocessor

# Supported estimators, keyed by the name used in ``config.yaml``.
ESTIMATORS: Final[Dict[str, Type[ClassifierMixin]]] = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
}


def build_estimator(
    name: str, params: Optional[Dict[str, Any]] = None
) -> ClassifierMixin:
    """Instantiate a registered classifier with the given parameters.

    Args:
        name: Registry key of the estimator (see :data:`ESTIMATORS`).
        params: Keyword arguments forwarded to the estimator constructor.

    Returns:
        An unfitted classifier instance.

    Raises:
        ValueError: If ``name`` is not a registered estimator.
    """
    try:
        estimator_cls = ESTIMATORS[name]
    except KeyError as error:
        supported = ", ".join(sorted(ESTIMATORS))
        raise ValueError(
            f"Unsupported model '{name}'. Supported models: {supported}."
        ) from error
    return estimator_cls(**(params or {}))


def build_model(config: Dict[str, Any]) -> Pipeline:
    """Build the unfitted end-to-end churn model.

    Args:
        config: Parsed project configuration containing the ``features`` and
            ``model`` sections.

    Returns:
        A :class:`~sklearn.pipeline.Pipeline` with a ``preprocessor`` step
        followed by a ``classifier`` step.
    """
    model_cfg = config["model"]
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(config)),
            (
                "classifier",
                build_estimator(model_cfg["name"], model_cfg.get("params")),
            ),
        ]
    )
