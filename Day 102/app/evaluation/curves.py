"""ROC and Precision-Recall curve data extraction."""
from typing import Dict, Any
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve

def extract_roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, Any]:
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thresholds.tolist()}

def extract_pr_curve(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, Any]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    return {"precision": precision.tolist(), "recall": recall.tolist(), "thresholds": thresholds.tolist()}
