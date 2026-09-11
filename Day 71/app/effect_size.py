"""
Standardized effect size calculations: Cohen's d for independent and paired designs.
"""

import math
from typing import Sequence, Union
import numpy as np

def compute_cohens_d_independent(
    group_a: Sequence[Union[int, float]],
    group_b: Sequence[Union[int, float]]
) -> dict:
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    n_a, n_b = len(a), len(b)
    
    mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
    var_a, var_b = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    
    s_pooled = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    diff = mean_b - mean_a
    d = diff / s_pooled if s_pooled > 0 else 0.0
    abs_d = abs(d)
    
    if abs_d < 0.20:
        tier = "Negligible"
    elif abs_d < 0.50:
        tier = "Small"
    elif abs_d < 0.80:
        tier = "Medium"
    else:
        tier = "Large"
        
    return {
        "mean_difference": round(diff, 4),
        "pooled_std": round(s_pooled, 4),
        "cohens_d": round(d, 4),
        "abs_cohens_d": round(abs_d, 4),
        "magnitude": tier
    }

def compute_cohens_d_paired(
    before: Sequence[Union[int, float]],
    after: Sequence[Union[int, float]]
) -> dict:
    b = np.asarray(before, dtype=float)
    a = np.asarray(after, dtype=float)
    diffs = a - b
    mean_d = float(np.mean(diffs))
    std_d = float(np.std(diffs, ddof=1))
    d = mean_d / std_d if std_d > 0 else 0.0
    abs_d = abs(d)
    
    if abs_d < 0.20:
        tier = "Negligible"
    elif abs_d < 0.50:
        tier = "Small"
    elif abs_d < 0.80:
        tier = "Medium"
    else:
        tier = "Large"
        
    return {
        "mean_difference": round(mean_d, 4),
        "std_difference": round(std_d, 4),
        "cohens_d": round(d, 4),
        "abs_cohens_d": round(abs_d, 4),
        "magnitude": tier
    }
