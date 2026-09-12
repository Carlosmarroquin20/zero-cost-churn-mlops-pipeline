"""Model evaluation module.

Compute the classification metrics used to compare churn models. Metrics are
derived from predicted churn probabilities, so threshold-independent scores
(ROC AUC, average precision) and threshold-dependent scores (precision, recall,
F1, accuracy) are always reported together.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: ArrayLike, y_score: ArrayLike, threshold: float = 0.5
) -> Dict[str, float]:
    """Compute churn classification metrics from predicted probabilities.

    Args:
        y_true: Ground-truth binary labels (1 means churn).
        y_score: Predicted probability of the positive (churn) class.
        threshold: Probability cut-off used to derive class labels.

    Returns:
        A mapping of metric name to value.

    Raises:
        ValueError: If ``threshold`` is outside the ``[0, 1]`` interval.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be within [0, 1], got {threshold}.")

    labels = np.asarray(y_true)
    scores = np.asarray(y_score, dtype=float)
    predictions = (scores >= threshold).astype(int)

    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(labels, predictions)),
    }
