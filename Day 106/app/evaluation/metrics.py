"""
Evaluation metrics module for Day 106: RNNs & Sequential Text Learning.
Computes Accuracy, Precision, Recall, F1, ROC-AUC, and Average Precision.
"""

from typing import Dict, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


class ClassificationMetrics:
    """Computes standard binary classification performance indicators."""

    @staticmethod
    def compute(
        y_true: Union[list, np.ndarray],
        y_probs: Union[list, np.ndarray],
        threshold: float = 0.50
    ) -> Dict[str, float]:
        """Compute all primary classification metrics.
        
        Args:
            y_true: Ground truth binary labels (0 or 1).
            y_probs: Predicted probabilities for class 1.
            threshold: Classification decision boundary.
            
        Returns:
            Dictionary of calculated performance metrics.
        """
        y_t = np.asarray(y_true, dtype=int).flatten()
        y_p = np.asarray(y_probs, dtype=float).flatten()
        y_pred = (y_p >= threshold).astype(int)

        acc = float(accuracy_score(y_t, y_pred))
        prec = float(precision_score(y_t, y_pred, zero_division=0))
        rec = float(recall_score(y_t, y_pred, zero_division=0))
        f1 = float(f1_score(y_t, y_pred, zero_division=0))

        # Handle edge cases for ROC-AUC and Average Precision
        unique_classes = len(np.unique(y_t))
        if unique_classes > 1:
            roc_auc = float(roc_auc_score(y_t, y_p))
            ap = float(average_precision_score(y_t, y_p))
        else:
            roc_auc = 0.50
            ap = float(np.mean(y_t))

        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc_auc,
            "average_precision": ap,
            "threshold": threshold,
        }
