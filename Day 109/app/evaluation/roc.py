from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import numpy as np
from typing import Tuple

def compute_roc_curve_data(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = float(auc(fpr, tpr))
    return fpr, tpr, roc_auc

def compute_pr_curve_data(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = float(average_precision_score(y_true, y_prob))
    return precision, recall, ap
