"""
ML Evaluator — tính toán metrics chuẩn cho classification & regression.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from agent.schemas import ExperimentMetrics, TaskType


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    task_type: TaskType,
) -> ExperimentMetrics:
    """
    Compute standard evaluation metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities (needed for AUC). Can be None.
        task_type: TaskType enum value.

    Returns:
        ExperimentMetrics object with computed values.
    """
    metrics = ExperimentMetrics()

    if task_type == TaskType.BINARY_CLASSIFICATION:
        avg = "binary"
        metrics.accuracy = round(float(accuracy_score(y_true, y_pred)), 4)
        metrics.precision = round(float(precision_score(y_true, y_pred, zero_division=0)), 4)
        metrics.recall = round(float(recall_score(y_true, y_pred, zero_division=0)), 4)
        metrics.f1 = round(float(f1_score(y_true, y_pred, zero_division=0)), 4)
        if y_prob is not None:
            probs = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
            metrics.roc_auc = round(float(roc_auc_score(y_true, probs)), 4)

    elif task_type == TaskType.MULTICLASS_CLASSIFICATION:
        metrics.accuracy = round(float(accuracy_score(y_true, y_pred)), 4)
        metrics.precision = round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4)
        metrics.recall = round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4)
        metrics.f1 = round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4)
        if y_prob is not None:
            try:
                metrics.roc_auc = round(
                    float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")), 4
                )
            except Exception:
                pass

    return metrics


def get_full_classification_report(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, Any]:
    """Return sklearn's full classification_report as a dict."""
    return classification_report(y_true, y_pred, output_dict=True, zero_division=0)


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Return confusion matrix as a nested list."""
    cm = confusion_matrix(y_true, y_pred)
    return {"confusion_matrix": cm.tolist()}
