"""
Error analysis module for Day 105: Neural NLP & Text Classification.
Diagnoses false positive and false negative predictions to uncover systematic model blindspots.
"""

import re
from typing import List, Union
import numpy as np
import pandas as pd


class ErrorAnalyzer:
    """Diagnoses prediction discrepancies and categorizes error patterns."""

    @staticmethod
    def analyze(
        texts: List[str],
        y_true: Union[List[int], np.ndarray],
        y_probs: Union[List[float], np.ndarray],
        threshold: float = 0.5
    ) -> pd.DataFrame:
        """Produce an error analysis DataFrame for misclassified samples.
        
        Args:
            texts: List of raw input strings.
            y_true: Ground truth binary labels.
            y_probs: Model predicted probabilities for class 1.
            threshold: Classification decision boundary.
            
        Returns:
            DataFrame containing detailed diagnostics for all misclassified samples.
        """
        y_t = np.asarray(y_true, dtype=int).flatten()
        y_p = np.asarray(y_probs, dtype=float).flatten()
        y_pred = (y_p >= threshold).astype(int)

        rows = []
        for i, text in enumerate(texts):
            actual = int(y_t[i])
            pred = int(y_pred[i])
            prob = float(y_p[i])

            if actual != pred:
                error_type = "False Positive" if pred == 1 and actual == 0 else "False Negative"
                has_url = bool(re.search(r"https?://\S+|www\.\S+", text))
                has_phone = bool(re.search(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b|\b0800\S+|\b\d{10,11}\b", text))
                char_len = len(text)
                word_len = len(text.split())
                rows.append({
                    "text": text,
                    "actual": actual,
                    "predicted": pred,
                    "probability": prob,
                    "error_type": error_type,
                    "char_length": char_len,
                    "word_count": word_len,
                    "has_url": has_url,
                    "has_phone": has_phone
                })

        columns = [
            "text", "actual", "predicted", "probability",
            "error_type", "char_length", "word_count",
            "has_url", "has_phone"
        ]
        return pd.DataFrame(rows, columns=columns)
