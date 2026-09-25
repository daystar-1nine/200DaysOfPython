"""
Confusion matrix computation and diagnostics for Day 106.
"""

from typing import Dict, Union
import numpy as np
from sklearn.metrics import confusion_matrix


class ConfusionMatrixCalculator:
    """Computes confusion matrix values and rates for binary classification."""

    @staticmethod
    def compute(
        y_true: Union[list, np.ndarray],
        y_pred: Union[list, np.ndarray]
    ) -> Dict[str, Union[int, float]]:
        """Calculate TN, FP, FN, TP and derived error rates.
        
        Args:
            y_true: True binary labels (0=ham, 1=spam).
            y_pred: Predicted binary labels (0=ham, 1=spam).
            
        Returns:
            Dictionary with counts and rates.
        """
        y_t = np.asarray(y_true, dtype=int).flatten()
        y_p = np.asarray(y_pred, dtype=int).flatten()

        cm = confusion_matrix(y_t, y_p, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        return {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "false_positive_rate": float(fpr),
            "false_negative_rate": float(fnr),
            "matrix": cm.tolist(),
        }

    @staticmethod
    def format_table(results: Dict[str, Union[int, float]]) -> str:
        """Produce a formatted ASCII confusion matrix table."""
        tn = results["tn"]
        fp = results["fp"]
        fn = results["fn"]
        tp = results["tp"]
        lines = [
            "+--------------------+-------------------+-------------------+",
            "| Actual \\ Predicted | Predicted HAM (0) | Predicted SPAM (1)|",
            "+--------------------+-------------------+-------------------+",
            f"| Actual HAM (0)     | {tn:<17} | {fp:<17} |",
            f"| Actual SPAM (1)    | {fn:<17} | {tp:<17} |",
            "+--------------------+-------------------+-------------------+",
        ]
        return "\n".join(lines)
