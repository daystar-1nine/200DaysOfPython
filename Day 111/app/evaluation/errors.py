"""
Error analysis module for detecting, categorizing, and exporting classification mispredictions.
"""
from pathlib import Path
from typing import List, Dict, Any, Union
import numpy as np
import pandas as pd


def analyze_errors(
    texts: List[str],
    y_true: Union[np.ndarray, List[int]],
    y_pred: Union[np.ndarray, List[int]],
    y_probs: Union[np.ndarray, List[float]],
    output_path: Path = None
) -> pd.DataFrame:
    """
    Identifies False Positives and False Negatives, calculates error margin/confidence,
    and categorizes the failure mode.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    y_probs_arr = np.asarray(y_probs, dtype=float)

    error_records = []
    for idx, (text, true_lbl, pred_lbl, prob) in enumerate(zip(texts, y_true_arr, y_pred_arr, y_probs_arr)):
        if true_lbl != pred_lbl:
            err_type = "False Positive" if pred_lbl == 1 else "False Negative"
            confidence = prob if pred_lbl == 1 else (1.0 - prob)

            # Heuristic failure categorization
            lower_text = text.lower()
            if err_type == "False Positive":
                if any(w in lower_text for w in ["call", "free", "win", "urgent", "won", "prize", "cash", "txt"]):
                    reason = "Promotional or urgent trigger words in legitimate message"
                elif len(text.split()) < 5:
                    reason = "Extremely short text lacking context"
                else:
                    reason = "Conversational structure mimicking solicitation"
            else:  # False Negative
                if any(w in lower_text for w in ["link", "http", "click", "url", "www"]):
                    reason = "Spam relies heavily on external URL without typical vocabulary"
                elif len(text.split()) < 6:
                    reason = "Subtle short spam message"
                else:
                    reason = "Low keyword frequency or colloquial obfuscation"

            error_records.append({
                "sample_index": idx,
                "error_type": err_type,
                "true_label": int(true_lbl),
                "predicted_label": int(pred_lbl),
                "predicted_probability": round(float(prob), 4),
                "error_confidence": round(float(confidence), 4),
                "text_length": len(text),
                "word_count": len(text.split()),
                "hypothesized_cause": reason,
                "text": text
            })

    df = pd.DataFrame(error_records)
    if not df.empty:
        df = df.sort_values(by="error_confidence", ascending=False).reset_index(drop=True)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Error analysis saved to: {output_path} | Total errors found: {len(df)}")

    return df
