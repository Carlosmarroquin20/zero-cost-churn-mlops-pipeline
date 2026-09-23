"""Tests that keep the development environment aligned with the serving image."""

import re
from importlib.metadata import version
from pathlib import Path
from typing import Dict, List

import pytest

REQUIREMENTS_FILE = Path(__file__).resolve().parents[1] / "requirements-api.txt"
# Packages that produce or consume the model pickle. A mismatch between the
# environment that trains the model and the image that serves it is unsupported.
MODEL_RUNTIME_PACKAGES = ("scikit-learn", "numpy", "pandas", "joblib")


def _requirement_lines() -> List[str]:
    """Return the requirement entries of the serving file, without comments."""
    lines = REQUIREMENTS_FILE.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def _pinned_versions() -> Dict[str, str]:
    """Parse ``name==version`` pins, ignoring any extras in the requirement."""
    pattern = re.compile(r"^([A-Za-z0-9_.\-]+)(?:\[[^\]]+\])?==(.+)$")
    matches = (pattern.match(line) for line in _requirement_lines())
    return {
        match.group(1).lower(): match.group(2).strip() for match in matches if match
    }


def test_every_serving_dependency_is_pinned() -> None:
    """Unpinned dependencies would make the serving image irreproducible."""
    assert len(_pinned_versions()) == len(_requirement_lines())


@pytest.mark.parametrize("package", MODEL_RUNTIME_PACKAGES)
def test_installed_version_matches_serving_pin(package: str) -> None:
    """Models are trained in this environment and loaded in the image."""
    assert version(package) == _pinned_versions()[package]
