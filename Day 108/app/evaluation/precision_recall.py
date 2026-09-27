from sklearn.metrics import precision_recall_curve, average_precision_score
import numpy as np
from typing import Tuple

def compute_pr_curve_data(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Computes precision-recall curve data and Average Precision (AP).
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = float(average_precision_score(y_true, y_prob))
    return precision, recall, ap
