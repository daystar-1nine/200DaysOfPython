from typing import Tuple
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score

def compute_pr_curve_data(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc = float(average_precision_score(y_true, y_prob))
    return precision, recall, thresholds, pr_auc
