"""Error analysis module for NLP predictions."""
import pandas as pd
from typing import Dict, List, Any

def extract_prediction_errors(texts: List[str], y_true: List[int], y_pred: List[int], y_proba: List[float]) -> pd.DataFrame:
    """Extracts False Positives (legitimate ham flagged as spam) and False Negatives (spam missed)."""
    records = []
    for text, true_lbl, pred_lbl, prob in zip(texts, y_true, y_pred, y_proba):
        if true_lbl != pred_lbl:
            error_type = "False Positive (Ham as Spam)" if pred_lbl == 1 else "False Negative (Spam as Ham)"
            records.append({
                "text": text,
                "true_label": "spam" if true_lbl == 1 else "ham",
                "pred_label": "spam" if pred_lbl == 1 else "ham",
                "spam_probability": float(prob),
                "error_type": error_type,
                "length": len(text),
                "word_count": len(text.split())
            })
    return pd.DataFrame(records)

def diagnose_errors(error_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Analyzes reasons for misclassification."""
    diagnostics = []
    for _, row in error_df.iterrows():
        reasons = []
        text_lower = row["text"].lower()
        if any(w in text_lower for w in ["call", "win", "prize", "free", "urgent", "claim", "reward", "http"]):
            reasons.append("Contains trigger keywords associated with promotional spam")
        if any(char.isdigit() for char in row["text"]):
            reasons.append("Contains phone numbers, codes, or numeric amounts")
        if row["length"] < 25:
            reasons.append("Very short text with insufficient contextual cues")
        if not reasons:
            reasons.append("Atypical vocabulary or colloquial phrasing")
            
        diagnostics.append({
            "text": row["text"],
            "error_type": row["error_type"],
            "probability": row["spam_probability"],
            "reasons": reasons
        })
    return diagnostics
