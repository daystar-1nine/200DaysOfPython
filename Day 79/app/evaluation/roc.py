from typing import Tuple
import numpy as np
from sklearn.metrics import roc_curve, auc

def compute_roc_curve_data(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = float(auc(fpr, tpr))
    return fpr, tpr, thresholds, roc_auc
