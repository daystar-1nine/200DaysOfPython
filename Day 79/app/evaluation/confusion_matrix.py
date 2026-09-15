from typing import Dict
import numpy as np
from sklearn.metrics import confusion_matrix

def compute_confusion_matrix_breakdown(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn),
        'TP': int(tp),
        'total': int(len(y_true))
    }
