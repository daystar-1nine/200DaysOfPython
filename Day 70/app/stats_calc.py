"""
Descriptive and inferential statistical calculations.
"""

import math
from typing import Sequence, Union
import numpy as np

try:
    from app.validator import validate_numeric_array
except (ImportError, ModuleNotFoundError):
    from validator import validate_numeric_array

def compute_sample_stats(data: Sequence[Union[int, float]]) -> dict:
    arr = validate_numeric_array(data, min_len=2)
    n = len(arr)
    mean_val = float(np.mean(arr))
    var_val = float(np.var(arr, ddof=1))
    std_val = float(np.std(arr, ddof=1))
    se_val = std_val / math.sqrt(n)
    
    return {
        "n": n,
        "mean": mean_val,
        "variance": var_val,
        "std": std_val,
        "se": se_val,
        "df": n - 1,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr))
    }
