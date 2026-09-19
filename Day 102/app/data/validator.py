"""Data validation utilities."""
import pandas as pd
from typing import Dict, Any

def validate_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Validates schema, null counts, and class values."""
    if df.empty:
        raise ValueError("Dataset is empty.")
    
    issues = []
    if "label" not in df.columns or "text" not in df.columns:
        issues.append("Missing required columns ('label', 'text').")
        
    null_counts = {}
    classes = []
    if "label" in df.columns and "text" in df.columns:
        null_counts = df[["label", "text"]].isnull().sum().to_dict()
        for col, count in null_counts.items():
            if count > 0:
                issues.append(f"Column '{col}' contains {count} null values.")
        classes = list(set(df["label"].dropna().unique()))
        if not set(classes).issubset({"ham", "spam"}):
            issues.append(f"Unexpected classes found: {set(classes) - {'ham', 'spam'}}")
            
    return {
        "is_valid": len(issues) == 0,
        "issues": issues,
        "null_counts": null_counts,
        "classes": classes
    }
