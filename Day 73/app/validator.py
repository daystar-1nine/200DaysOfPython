"""
Day 73 - Dataset Validator
Performs structural integrity, column presence, and variance checks.
"""
from typing import Dict, Any
import numpy as np
import pandas as pd

def validate_data(df: pd.DataFrame, feature_col: str = "Advertising_Spend", target_col: str = "Sales", min_rows: int = 10) -> Dict[str, Any]:
    if feature_col not in df.columns:
        raise KeyError(f"Feature column '{feature_col}' not found in dataset.")
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataset.")
        
    n_rows = len(df)
    if n_rows < min_rows:
        raise ValueError(f"Dataset must contain at least {min_rows} rows. Found {n_rows}.")
        
    var_x = float(np.var(df[feature_col], ddof=1))
    var_y = float(np.var(df[target_col], ddof=1))
    
    if np.isclose(var_x, 0.0):
        raise ValueError(f"Feature '{feature_col}' has zero variance; slope cannot be determined.")
    if np.isclose(var_y, 0.0):
        raise ValueError(f"Target '{target_col}' has zero variance.")
        
    return {
        "valid": True,
        "n_rows": n_rows,
        "feature_variance": var_x,
        "target_variance": var_y,
        "min_feature": float(df[feature_col].min()),
        "max_feature": float(df[feature_col].max())
    }
