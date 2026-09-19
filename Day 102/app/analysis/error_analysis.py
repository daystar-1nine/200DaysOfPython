"""Error analysis and diagnostic classification."""
import pandas as pd
from typing import List, Dict, Any

def extract_classification_errors(
    texts: List[str],
    y_true: List[int],
    y_pred: List[int],
    y_score: List[float],
    model_name: str
) -> pd.DataFrame:
    """Extracts False Positives and False Negatives into structured dataframe."""
    errors = []
    for text, true_lbl, pred_lbl, score in zip(texts, y_true, y_pred, y_score):
        if true_lbl != pred_lbl:
            err_type = "False Positive" if pred_lbl == 1 else "False Negative"
            errors.append({
                "text": text,
                "actual": "spam" if true_lbl == 1 else "ham",
                "predicted": "spam" if pred_lbl == 1 else "ham",
                "model": model_name,
                "confidence_or_score": float(score),
                "error_type": err_type
            })
    return pd.DataFrame(errors)
