"""
Error analysis module for Day 106: RNNs & Sequential Text Learning.
Diagnoses misclassified messages and identifies systematic linguistic failure modes.
"""

import re
from typing import List, Union
import numpy as np
import pandas as pd


class ErrorAnalyzer:
    """Diagnoses false positives and false negatives at the message level."""

    @staticmethod
    def analyze(
        messages: List[str],
        y_true: Union[list, np.ndarray],
        y_probs: Union[list, np.ndarray],
        threshold: float = 0.50
    ) -> pd.DataFrame:
        """Analyze prediction errors and extract linguistic characteristics.
        
        Args:
            messages: List of raw SMS text strings.
            y_true: True binary labels (0=ham, 1=spam).
            y_probs: Predicted probability estimates for spam (class 1).
            threshold: Decision boundary.
            
        Returns:
            DataFrame containing diagnostic attributes for misclassified samples.
        """
        y_t = np.asarray(y_true, dtype=int).flatten()
        y_p = np.asarray(y_probs, dtype=float).flatten()
        y_pred = (y_p >= threshold).astype(int)

        rows = []
        for i, text in enumerate(messages):
            actual = int(y_t[i])
            pred = int(y_pred[i])
            prob = float(y_p[i])

            if actual != pred:
                error_type = "False Positive" if pred == 1 and actual == 0 else "False Negative"
                has_url = bool(re.search(r"https?://\S+|www\.\S+", text))
                has_phone = bool(re.search(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b|\b0800\S+|\b\d{10,11}\b", text))
                has_currency = bool(re.search(r"[£$€]|pkr|inr|usd", text.lower()))
                char_len = len(text)
                word_count = len(text.split())

                rows.append({
                    "message": text,
                    "actual_label": "spam" if actual == 1 else "ham",
                    "predicted_label": "spam" if pred == 1 else "ham",
                    "probability": round(prob, 4),
                    "error_type": error_type,
                    "char_length": char_len,
                    "word_count": word_count,
                    "has_url": has_url,
                    "has_phone": has_phone,
                    "has_currency": has_currency,
                })

        columns = [
            "message", "actual_label", "predicted_label", "probability",
            "error_type", "char_length", "word_count", "has_url",
            "has_phone", "has_currency"
        ]
        return pd.DataFrame(rows, columns=columns)
