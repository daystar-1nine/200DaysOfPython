"""Confusion matrix extraction and metrics."""
from typing import Dict
import numpy as np
from sklearn.metrics import confusion_matrix

def get_confusion_matrix_breakdown(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    """
    Computes TP, TN, FP, FN for binary classification.
    Assumes positive class = 1 (spam), negative class = 0 (ham).
    """
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = int(cm[0, 0]), 0, 0, 0
        
    return {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }
