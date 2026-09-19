"""Data validation module."""
import pandas as pd

def validate_dataset(df: pd.DataFrame) -> dict:
    """Validates dataset integrity and returns diagnostic metrics."""
    if df.empty:
        raise ValueError("Dataset is empty.")
    
    issues = []
    if "label" not in df.columns or "text" not in df.columns:
        issues.append("Missing required columns ('label', 'text').")
        
    null_counts = df[["label", "text"]].isnull().sum().to_dict()
    class_counts = df["label"].value_counts().to_dict() if "label" in df.columns else {}
    
    has_imbalance = False
    if class_counts:
        min_class = min(class_counts.values())
        max_class = max(class_counts.values())
        if max_class / (min_class + 1e-9) > 3.0:
            has_imbalance = True
            
    return {
        "total_rows": len(df),
        "null_counts": null_counts,
        "class_distribution": class_counts,
        "class_imbalance_detected": has_imbalance,
        "issues": issues,
        "is_valid": len(issues) == 0
    }
