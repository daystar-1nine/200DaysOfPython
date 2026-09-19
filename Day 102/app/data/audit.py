"""Comprehensive data audit module."""
import pandas as pd
from typing import Dict, Any

def audit_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes real dataset statistics:
    - Number of records
    - Number of classes & distribution
    - Duplicate count
    - Missing values
    - Average, median, min, and max message length (characters and words)
    """
    total_records = len(df)
    class_dist = df["label"].value_counts().to_dict()
    dup_count = int(df.duplicated(subset=["text"]).sum())
    missing_vals = df[["label", "text"]].isnull().sum().to_dict()
    
    char_lengths = df["text"].astype(str).apply(len)
    word_counts = df["text"].astype(str).apply(lambda s: len(s.split()))
    
    return {
        "total_records": total_records,
        "num_classes": len(class_dist),
        "class_distribution": class_dist,
        "duplicate_count": dup_count,
        "missing_values": missing_vals,
        "char_length": {
            "mean": float(char_lengths.mean()),
            "median": float(char_lengths.median()),
            "min": int(char_lengths.min()),
            "max": int(char_lengths.max())
        },
        "word_count": {
            "mean": float(word_counts.mean()),
            "median": float(word_counts.median()),
            "min": int(word_counts.min()),
            "max": int(word_counts.max())
        }
    }
