from typing import Tuple
import numpy as np
from sklearn.metrics import roc_curve, auc

def calculate_roc_curve(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute ROC curve and AUC score."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = float(auc(fpr, tpr))
    return fpr, tpr, thresholds, roc_auc
