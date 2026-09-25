"""
ROC Curve and AUC computation module for Day 106: RNNs & Sequential Text Learning.
"""

from typing import Dict, Tuple, Union
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score


class ROCEvaluator:
    """Computes Receiver Operating Characteristic curve coordinates and AUC."""

    @staticmethod
    def compute_curve(
        y_true: Union[list, np.ndarray],
        y_probs: Union[list, np.ndarray]
    ) -> Dict[str, Union[float, np.ndarray]]:
        """Compute FPR, TPR, thresholds, and ROC-AUC.
        
        Args:
            y_true: Ground truth binary labels.
            y_probs: Predicted probabilities for class 1.
            
        Returns:
            Dictionary with 'fpr', 'tpr', 'thresholds', and 'auc'.
        """
        y_t = np.asarray(y_true, dtype=int).flatten()
        y_p = np.asarray(y_probs, dtype=float).flatten()

        if len(np.unique(y_t)) < 2:
            return {
                "fpr": np.array([0.0, 1.0]),
                "tpr": np.array([0.0, 1.0]),
                "thresholds": np.array([0.5, 0.5]),
                "auc": 0.5,
            }

        fpr, tpr, thresholds = roc_curve(y_t, y_p)
        auc_val = float(roc_auc_score(y_t, y_p))

        return {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
            "auc": auc_val,
        }
