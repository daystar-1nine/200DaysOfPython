from sklearn.metrics import roc_curve, auc
import numpy as np
from typing import Tuple

def compute_roc_curve_data(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Computes false positive rate, true positive rate, and AUC.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = float(auc(fpr, tpr))
    return fpr, tpr, roc_auc
