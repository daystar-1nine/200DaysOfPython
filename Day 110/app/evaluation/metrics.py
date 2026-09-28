"""
Evaluation metrics computation module for binary classification.
"""
from typing import Dict, Any, List
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


def compute_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Computes standard evaluation metrics:
    Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC.
    """
    y_true = np.asarray(y_true).ravel()
    y_probs = np.asarray(y_probs).ravel()
    y_pred = (y_probs >= threshold).astype(int)

    # In case a split has only 1 class or edge cases
    try:
        roc_auc = float(roc_auc_score(y_true, y_probs))
    except ValueError:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_true, y_probs))
    except ValueError:
        pr_auc = 0.0

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "threshold": float(threshold)
    }
