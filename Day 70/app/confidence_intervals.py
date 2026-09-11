"""
Confidence interval calculators linked to hypothesis tests.
"""

import math
from typing import Sequence, Union
import numpy as np
from scipy import stats

try:
    from app.validator import validate_numeric_array, validate_alpha
    from app.stats_calc import compute_sample_stats
except (ImportError, ModuleNotFoundError):
    from validator import validate_numeric_array, validate_alpha
    from stats_calc import compute_sample_stats

def compute_mean_ci(data: Sequence[Union[int, float]], confidence_level: float = 0.95) -> dict:
    arr = validate_numeric_array(data)
    alpha = 1.0 - confidence_level
    alpha = validate_alpha(alpha)
    
    st = compute_sample_stats(arr)
    t_crit = float(stats.t.ppf(1 - alpha / 2, df=st["df"]))
    me = t_crit * st["se"]
    
    return {
        "confidence_level": confidence_level,
        "mean": st["mean"],
        "margin_of_error": me,
        "ci_lower": st["mean"] - me,
        "ci_upper": st["mean"] + me
    }

def check_ci_contains_null(ci_lower: float, ci_upper: float, null_val: float) -> bool:
    return ci_lower <= null_val <= ci_upper
