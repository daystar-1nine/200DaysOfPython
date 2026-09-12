"""
Day 72 — Dataset Validator
Validates numerical dimensions and identifies constant columns with zero variance.
"""
from typing import Dict, Any, List
import numpy as np
import pandas as pd

def validate_numeric_data(df: pd.DataFrame, min_rows: int = 10, min_cols: int = 2) -> Dict[str, Any]:
    n_rows, n_cols = df.shape
    if n_rows < min_rows:
        raise ValueError(f"Dataset must have at least {min_rows} rows. Found {n_rows}.")
    if n_cols < min_cols:
        raise ValueError(f"Dataset must have at least {min_cols} columns. Found {n_cols}.")
        
    # Check for constant columns (zero standard deviation)
    constant_cols = []
    for col in df.columns:
        if np.isclose(df[col].std(ddof=1), 0.0):
            constant_cols.append(col)
            
    return {
        "valid": len(constant_cols) == 0,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "columns": list(df.columns),
        "constant_columns": constant_cols
    }
