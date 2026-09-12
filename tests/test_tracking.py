"""Unit tests for the experiment tracking module."""

from typing import Any, Dict

import mlflow
import pytest

from src.tracking import (
    TRACKING_URI_ENV_VAR,
    configure_tracking,
    redact_credentials,
    resolve_tracking_uri,
)


def test_resolve_tracking_uri_falls_back_to_config(config: Dict[str, Any]) -> None:
    """Without the environment variable, the local store from config is used."""
    assert resolve_tracking_uri(config) == config["mlflow"]["tracking_uri"]


def test_resolve_tracking_uri_prefers_environment(
    config: Dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A remote server configured in the environment takes precedence."""
    remote_uri = "https://dagshub.com/user/repo.mlflow"
    monkeypatch.setenv(TRACKING_URI_ENV_VAR, remote_uri)

    assert resolve_tracking_uri(config) == remote_uri


def test_resolve_tracking_uri_ignores_empty_environment_value(
    config: Dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty variable, as left by a blank .env entry, must not win."""
    monkeypatch.setenv(TRACKING_URI_ENV_VAR, "")

    assert resolve_tracking_uri(config) == config["mlflow"]["tracking_uri"]


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        (
            "https://user:secret@dagshub.com/user/repo.mlflow",
            "https://dagshub.com/user/repo.mlflow",
        ),
        (
            "https://token@dagshub.com:443/user/repo.mlflow",
            "https://dagshub.com:443/user/repo.mlflow",
        ),
        ("sqlite:///mlflow.db", "sqlite:///mlflow.db"),
    ],
)
def test_redact_credentials_strips_user_info(uri: str, expected: str) -> None:
    """Credentials embedded in a URI must never reach the logs."""
    assert redact_credentials(uri) == expected


def test_configure_tracking_activates_experiment(
    local_tracking_config: Dict[str, Any],
) -> None:
    """Configuring tracking selects the store and creates the experiment."""
    tracking_uri = configure_tracking(local_tracking_config)

    assert mlflow.get_tracking_uri() == tracking_uri
    experiment_name = local_tracking_config["mlflow"]["experiment_name"]
    assert mlflow.get_experiment_by_name(experiment_name) is not None
