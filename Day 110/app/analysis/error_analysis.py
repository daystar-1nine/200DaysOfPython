"""
Error analysis module investigating false positives and false negatives in Transformer predictions.
"""
from typing import List, Dict, Any
import numpy as np
import pandas as pd


def perform_error_analysis(
    texts: List[str],
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    Categorizes predictions into TP, TN, FP, FN and flags misclassified examples.
    """
    y_true = np.asarray(y_true).ravel()
    y_probs = np.asarray(y_probs).ravel()
    y_pred = (y_probs >= threshold).astype(int)

    records = []
    for text, true_lbl, prob, pred in zip(texts, y_true, y_probs, y_pred):
        if true_lbl == 1 and pred == 1:
            cat = "TP"
        elif true_lbl == 0 and pred == 0:
            cat = "TN"
        elif true_lbl == 0 and pred == 1:
            cat = "FP"
        else:
            cat = "FN"

        is_error = int(cat in ["FP", "FN"])
        records.append({
            "text": text,
            "true_label": int(true_lbl),
            "predicted_prob": float(prob),
            "predicted_label": int(pred),
            "category": cat,
            "is_error": is_error
        })

    return pd.DataFrame(records)
