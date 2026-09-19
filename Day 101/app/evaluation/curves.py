"""ROC and Precision-Recall curve extraction."""
from typing import Tuple, Dict, Any
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve

def compute_roc_data(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thresholds.tolist()}

def compute_pr_data(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    return {"precision": precision.tolist(), "recall": recall.tolist(), "thresholds": thresholds.tolist()}
