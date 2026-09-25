"""
Precision-Recall Curve and Average Precision module for Day 106: RNNs & Sequential Text Learning.
"""

from typing import Dict, Union
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score


class PrecisionRecallEvaluator:
    """Computes Precision-Recall curve coordinates and Average Precision."""

    @staticmethod
    def compute_curve(
        y_true: Union[list, np.ndarray],
        y_probs: Union[list, np.ndarray]
    ) -> Dict[str, Union[float, np.ndarray]]:
        """Compute precision, recall, thresholds, and Average Precision.
        
        Args:
            y_true: Ground truth binary labels.
            y_probs: Predicted probabilities for class 1.
            
        Returns:
            Dictionary with 'precision', 'recall', 'thresholds', and 'average_precision'.
        """
        y_t = np.asarray(y_true, dtype=int).flatten()
        y_p = np.asarray(y_probs, dtype=float).flatten()

        if len(np.unique(y_t)) < 2:
            return {
                "precision": np.array([1.0, 0.0]),
                "recall": np.array([0.0, 1.0]),
                "thresholds": np.array([0.5]),
                "average_precision": 0.5,
            }

        precision, recall, thresholds = precision_recall_curve(y_t, y_p)
        ap = float(average_precision_score(y_t, y_p))

        return {
            "precision": precision,
            "recall": recall,
            "thresholds": thresholds,
            "average_precision": ap,
        }
