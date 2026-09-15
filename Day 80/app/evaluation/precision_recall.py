from typing import Tuple
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score

def calculate_pr_curve(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute Precision-Recall curve and Average Precision."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    ap = float(average_precision_score(y_true, y_prob))
    return precision, recall, thresholds, ap
