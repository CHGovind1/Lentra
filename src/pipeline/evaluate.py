"""Model  evaluation utilities."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
) -> dict[str, Any]:
    """Compute classification metrics for validation data."""
    metrics: dict[str, Any] = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    if y_prob is not None:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))
    return metrics


def print_metrics(metrics: dict[str, Any]) -> None:
    """Print metrics in a compact format."""
    print("[evaluate] Validation metrics")
    for key, value in metrics.items():
        print(f"  - {key}: {value:.4f}")
