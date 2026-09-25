"""
Threshold experimentation and operational trade-off module for Day 106.
Sweeps decision boundaries from 0.10 to 0.90 and evaluates FP vs FN cost implications.
"""

from typing import List, Union
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score


class ThresholdAnalyzer:
    """Evaluates classification performance across multiple probability thresholds."""

    @staticmethod
    def evaluate_thresholds(
        y_true: Union[list, np.ndarray],
        y_probs: Union[list, np.ndarray],
        thresholds: List[float] = None
    ) -> pd.DataFrame:
        """Sweep decision boundaries and compute performance metrics.
        
        Args:
            y_true: Ground truth binary labels.
            y_probs: Predicted probabilities.
            thresholds: List of thresholds to evaluate.
            
        Returns:
            DataFrame containing metrics for each threshold.
        """
        if thresholds is None:
            thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

        y_t = np.asarray(y_true, dtype=int).flatten()
        y_p = np.asarray(y_probs, dtype=float).flatten()

        rows = []
        for th in thresholds:
            y_pred = (y_p >= th).astype(int)
            prec = float(precision_score(y_t, y_pred, zero_division=0))
            rec = float(recall_score(y_t, y_pred, zero_division=0))
            f1 = float(f1_score(y_t, y_pred, zero_division=0))
            fp = int(np.sum((y_pred == 1) & (y_t == 0)))
            fn = int(np.sum((y_pred == 0) & (y_t == 1)))
            tp = int(np.sum((y_pred == 1) & (y_t == 1)))
            tn = int(np.sum((y_pred == 0) & (y_t == 0)))

            rows.append({
                "threshold": round(th, 2),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "tp": tp,
                "tn": tn,
                "false_positives": fp,
                "false_negatives": fn,
            })

        return pd.DataFrame(rows)
