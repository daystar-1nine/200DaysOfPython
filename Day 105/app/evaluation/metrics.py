"""
Metrics module for Day 105: Neural NLP & Text Classification.
Calculates Accuracy, Precision, Recall, F1, ROC-AUC, and Average Precision.
"""

from typing import Dict, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


def compute_classification_metrics(
    y_true: Union[np.ndarray, list],
    y_probs: Union[np.ndarray, list],
    threshold: float = 0.5
) -> Dict[str, float]:
    """Calculate comprehensive binary classification metrics.
    
    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_probs: Predicted probability estimates for class 1.
        threshold: Decision threshold for classification (default 0.5).
        
    Returns:
        Dictionary of computed metric values.
    """
    y_t = np.asarray(y_true, dtype=int).flatten()
    y_p = np.asarray(y_probs, dtype=float).flatten()

    y_pred = (y_p >= threshold).astype(int)

    acc = float(accuracy_score(y_t, y_pred))
    prec = float(precision_score(y_t, y_pred, zero_division=0))
    rec = float(recall_score(y_t, y_pred, zero_division=0))
    f1 = float(f1_score(y_t, y_pred, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_t, y_p))
    except ValueError:
        roc_auc = 0.5

    try:
        avg_prec = float(average_precision_score(y_t, y_p))
    except ValueError:
        avg_prec = 0.0

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "average_precision": avg_prec
    }
