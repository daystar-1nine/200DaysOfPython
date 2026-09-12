import numpy as np
from sklearn.metrics import roc_curve
from typing import Tuple

def compute_roc_curve(y_true, probabilities) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fpr, tpr, thresholds = roc_curve(y_true, probabilities)
    return fpr, tpr, thresholds

def find_optimal_roc_threshold(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray) -> float:
    j_statistic = tpr - fpr
    optimal_idx = np.argmax(j_statistic)
    return thresholds[optimal_idx]
