"""
Confusion matrix calculation and formatting for Day 105.
"""

from typing import Dict, Union
import numpy as np
from sklearn.metrics import confusion_matrix


def compute_confusion_matrix(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list]
) -> Dict[str, int]:
    """Calculate binary confusion matrix values (TN, FP, FN, TP)."""
    y_t = np.asarray(y_true, dtype=int).flatten()
    y_p = np.asarray(y_pred, dtype=int).flatten()

    cm = confusion_matrix(y_t, y_p, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "matrix": cm
    }


def format_confusion_matrix(cm_dict: Dict[str, int]) -> str:
    """Format confusion matrix as human-readable ASCII table."""
    tn = cm_dict["true_negatives"]
    fp = cm_dict["false_positives"]
    fn = cm_dict["false_negatives"]
    tp = cm_dict["true_positives"]

    lines = [
        "------------------------------------",
        "         | Predicted HAM | Predicted SPAM",
        "------------------------------------",
        f"Actual HAM  | {tn:<13} | {fp:<14}",
        f"Actual SPAM | {fn:<13} | {tp:<14}",
        "------------------------------------"
    ]
    return "\n".join(lines)
