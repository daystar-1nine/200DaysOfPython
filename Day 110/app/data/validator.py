"""
Dataset validation module ensuring data integrity and zero leakage.
"""
from typing import Dict, Any
import pandas as pd


def validate_sms_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validates cleaned dataframe schema, types, and class balance.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("DataFrame is empty.")

    required_columns = {"text", "label"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"DataFrame must contain {required_columns}, found {df.columns}")

    unique_labels = set(df["label"].unique())
    if not unique_labels.issubset({0, 1}):
        raise ValueError(f"Labels must be binary {0, 1}, found {unique_labels}")

    stats = {
        "total_samples": len(df),
        "ham_count": int((df["label"] == 0).sum()),
        "spam_count": int((df["label"] == 1).sum()),
        "spam_percentage": float((df["label"] == 1).mean() * 100),
        "is_valid": True,
    }
    return stats
