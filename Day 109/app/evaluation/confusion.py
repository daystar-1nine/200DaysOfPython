import numpy as np
from sklearn.metrics import confusion_matrix
from typing import Dict, Tuple

def compute_confusion_matrix_details(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Tuple[np.ndarray, Dict[str, int]]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    details = {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp)
    }
    return cm, details
