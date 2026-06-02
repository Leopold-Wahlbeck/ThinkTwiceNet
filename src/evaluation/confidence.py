from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score


def evaluate_confidence_thresholds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
    thresholds: list[float],
) -> list[dict[str, float]]:
    """
    Evaluate how a classifier behaves under different confidence thresholds.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        confidence: Confidence score for each prediction.
        thresholds: Confidence thresholds to evaluate.

    Returns:
        A list of dictionaries containing coverage, accepted accuracy,
        and deferred rate for each threshold.
    """
    results = []

    for threshold in thresholds:
        accepted = confidence >= threshold
        coverage = accepted.mean()
        deferred_rate = 1.0 - coverage

        if accepted.sum() > 0:
            accepted_accuracy = accuracy_score(y_true[accepted], y_pred[accepted])
        else:
            accepted_accuracy = 0.0

        results.append(
            {
                "threshold": threshold,
                "coverage": coverage,
                "accepted_accuracy": accepted_accuracy,
                "deferred_rate": deferred_rate,
                "num_accepted": int(accepted.sum()),
                "num_deferred": int((~accepted).sum()),
            }
        )

    return results


def print_confidence_threshold_results(results: list[dict[str, float]]) -> None:
    """
    Print confidence-threshold results as a readable table.
    """
    print("\nConfidence threshold analysis:")
    print(
        "Threshold | Coverage | Accepted accuracy | Deferred | Accepted | Deferred count"
    )
    print("-" * 78)

    for row in results:
        print(
            f"{row['threshold']:9.2f} | "
            f"{row['coverage']:8.3f} | "
            f"{row['accepted_accuracy']:17.3f} | "
            f"{row['deferred_rate']:8.3f} | "
            f"{int(row['num_accepted']):8d} | "
            f"{int(row['num_deferred']):14d}"
        )