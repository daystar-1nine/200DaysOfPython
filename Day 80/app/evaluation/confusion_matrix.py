from typing import Dict
import numpy as np
from sklearn.metrics import confusion_matrix

def compute_confusion_breakdown(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    """Breakdown confusion matrix into TP, FP, FN, TN."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn),
        'TP': int(tp),
        'Total': int(len(y_true))
    }
