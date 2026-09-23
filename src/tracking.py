"""Experiment tracking module.

Configure where MLflow records runs. When ``MLFLOW_TRACKING_URI`` is set in the
environment (for example, pointing to a DagsHub repository, with credentials in
``MLFLOW_TRACKING_USERNAME`` and ``MLFLOW_TRACKING_PASSWORD``), runs are sent to
that remote server. Otherwise they fall back to the local store declared in
``config.yaml``, so training works offline and without any secrets.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Final
from urllib.parse import urlsplit, urlunsplit

import mlflow

logger = logging.getLogger(__name__)

TRACKING_URI_ENV_VAR: Final[str] = "MLFLOW_TRACKING_URI"


def resolve_tracking_uri(config: Dict[str, Any]) -> str:
    """Return the MLflow tracking URI, preferring the environment.

    Args:
        config: Parsed project configuration containing the ``mlflow`` section.

    Returns:
        The value of ``MLFLOW_TRACKING_URI`` when it is set and non-empty,
        otherwise the local fallback URI from the configuration.
    """
    return os.environ.get(TRACKING_URI_ENV_VAR) or config["mlflow"]["tracking_uri"]


def redact_credentials(uri: str) -> str:
    """Remove any user information embedded in a URI so it is safe to log.

    Args:
        uri: A URI that may contain ``user:password@`` credentials.

    Returns:
        The same URI without its user information.
    """
    parts = urlsplit(uri)
    if "@" not in parts.netloc:
        return uri
    return urlunsplit(parts._replace(netloc=parts.netloc.rsplit("@", 1)[1]))


def configure_tracking(config: Dict[str, Any]) -> str:
    """Point MLflow at the resolved tracking store and activate the experiment.

    Args:
        config: Parsed project configuration containing the ``mlflow`` section.

    Returns:
        The tracking URI in use.
    """
    tracking_uri = resolve_tracking_uri(config)
    experiment_name = config["mlflow"]["experiment_name"]

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    logger.info(
        "Tracking MLflow runs at %s (experiment '%s').",
        redact_credentials(tracking_uri),
        experiment_name,
    )
    return tracking_uri
