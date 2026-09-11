"""
Input validation module for A/B testing and comparative inference.
"""

from typing import Sequence, Union
import numpy as np
import pandas as pd

def validate_groups(df: pd.DataFrame, group_col: str = "group") -> tuple[pd.DataFrame, pd.DataFrame]:
    if df is None or len(df) == 0:
        raise ValueError("DataFrame cannot be empty or None.")
    if group_col not in df.columns:
        raise ValueError(f"Missing required group column: '{group_col}'")
        
    unique_groups = set(df[group_col].dropna().unique())
    if not ({"Control", "Treatment"}.issubset(unique_groups) or len(unique_groups) == 2):
        raise ValueError(f"DataFrame must contain exactly two groups (e.g. 'Control' and 'Treatment'), found: {unique_groups}")
        
    ctrl = df[df[group_col] == "Control"]
    trt = df[df[group_col] == "Treatment"]
    if len(ctrl) < 2 or len(trt) < 2:
        raise ValueError(f"Both groups must contain at least 2 observations (Control={len(ctrl)}, Treatment={len(trt)})")
    return ctrl, trt

def validate_binary_vector(vec: Sequence[Union[int, float]]) -> np.ndarray:
    arr = np.asarray(vec, dtype=float)
    arr = arr[~np.isnan(arr)]
    unique_vals = set(arr)
    if not unique_vals.issubset({0.0, 1.0}):
        raise ValueError(f"Binary outcome vector must contain only 0 and 1, found values: {unique_vals}")
    return arr.astype(int)

def validate_proportions(successes: int, total: int) -> float:
    if total <= 0:
        raise ValueError("Total trials must be strictly positive.")
    if not (0 <= successes <= total):
        raise ValueError(f"Successes ({successes}) must be between 0 and total trials ({total}).")
    return successes / total

def validate_alpha(alpha: float) -> float:
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"Significance level alpha must be strictly between 0 and 1, got {alpha}")
    return float(alpha)
