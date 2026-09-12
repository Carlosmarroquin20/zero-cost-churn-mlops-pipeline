"""Unit tests for the model evaluation module."""

import pytest

from src.evaluation import compute_classification_metrics

EXPECTED_METRICS = {
    "roc_auc",
    "average_precision",
    "precision",
    "recall",
    "f1",
    "accuracy",
}


def test_perfect_predictions_score_one() -> None:
    """A model that ranks and labels every sample correctly scores 1.0."""
    metrics = compute_classification_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])

    assert set(metrics) == EXPECTED_METRICS
    assert all(value == pytest.approx(1.0) for value in metrics.values())


def test_threshold_changes_labels_but_not_ranking_metrics() -> None:
    """Lowering the threshold trades precision for recall; ROC AUC is unchanged."""
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.6, 0.4, 0.9]

    default = compute_classification_metrics(y_true, y_score, threshold=0.5)
    lenient = compute_classification_metrics(y_true, y_score, threshold=0.35)

    assert default["precision"] == pytest.approx(0.5)
    assert default["recall"] == pytest.approx(0.5)
    assert lenient["precision"] == pytest.approx(2 / 3)
    assert lenient["recall"] == pytest.approx(1.0)
    assert default["roc_auc"] == pytest.approx(0.75)
    assert lenient["roc_auc"] == pytest.approx(default["roc_auc"])


@pytest.mark.parametrize("threshold", [-0.1, 1.5])
def test_invalid_threshold_is_rejected(threshold: float) -> None:
    """Thresholds outside [0, 1] are meaningless for probabilities."""
    with pytest.raises(ValueError, match="threshold"):
        compute_classification_metrics([0, 1], [0.2, 0.8], threshold=threshold)
