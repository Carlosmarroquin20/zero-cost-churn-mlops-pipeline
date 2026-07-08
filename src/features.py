"""Feature engineering module.

Build the scikit-learn preprocessing transformer used by the model: standard
scaling for numeric features and one-hot encoding for categorical features.

The returned transformer is *unfitted* by design. It is meant to be embedded in
a modeling :class:`~sklearn.pipeline.Pipeline` and fitted on the training split
only, which keeps preprocessing and the estimator together in a single,
leak-free, servable artifact.
"""

from __future__ import annotations

from typing import Any, Dict

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(config: Dict[str, Any]) -> ColumnTransformer:
    """Build the unfitted feature-preprocessing transformer.

    Args:
        config: Parsed project configuration containing the ``features``
            section with ``numeric`` and ``categorical`` column lists.

    Returns:
        A :class:`~sklearn.compose.ColumnTransformer` that scales numeric
        features and one-hot encodes categorical features.
    """
    features = config["features"]

    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), features["numeric"]),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", drop="if_binary"),
                features["categorical"],
            ),
        ],
        remainder="drop",
    )
