"""
Two-sample mean hypothesis tests: Independent Student's t-test, Welch's t-test, and Paired t-test.
"""

import math
from typing import Sequence, Union
import numpy as np
from scipy import stats

try:
    from app.validator import validate_alpha
except (ImportError, ModuleNotFoundError):
    from validator import validate_alpha

def independent_ttest(
    group_a: Sequence[Union[int, float]],
    group_b: Sequence[Union[int, float]],
    equal_var: bool = False,
    alpha: float = 0.05
) -> dict:
    alpha = validate_alpha(alpha)
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        raise ValueError("Both groups must have at least 2 valid numeric observations.")
        
    mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
    var_a, var_b = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    diff = mean_b - mean_a
    
    if equal_var:
        df = n_a + n_b - 2
        sp2 = ((n_a - 1) * var_a + (n_b - 1) * var_b) / df
        se = math.sqrt(sp2 * (1.0 / n_a + 1.0 / n_b))
    else:
        se = math.sqrt((var_a / n_a) + (var_b / n_b))
        va, vb = var_a / n_a, var_b / n_b
        df = ((va + vb)**2) / ((va**2) / (n_a - 1) + (vb**2) / (n_b - 1))
        
    t_stat = diff / se if se > 0 else 0.0
    p_val = float(2 * (1 - stats.t.cdf(abs(t_stat), df=df)))
    t_crit = float(stats.t.ppf(1 - alpha / 2, df=df))
    
    return {
        "test_type": "Student's Independent t-test" if equal_var else "Welch's Two-Sample t-test",
        "n_control": n_a,
        "n_treatment": n_b,
        "mean_control": round(mean_a, 4),
        "mean_treatment": round(mean_b, 4),
        "difference": round(diff, 4),
        "standard_error": round(se, 4),
        "degrees_of_freedom": round(df, 2),
        "t_statistic": float(t_stat),
        "critical_value": round(t_crit, 4),
        "p_value": p_val,
        "reject_null": bool(p_val < alpha)
    }

def paired_ttest(
    before: Sequence[Union[int, float]],
    after: Sequence[Union[int, float]],
    alpha: float = 0.05
) -> dict:
    alpha = validate_alpha(alpha)
    b = np.asarray(before, dtype=float)
    a = np.asarray(after, dtype=float)
    if len(b) != len(a):
        raise ValueError(f"Paired measurements must have identical length (Before={len(b)}, After={len(a)}).")
    if len(b) < 2:
        raise ValueError("Must have at least 2 pairs.")
        
    diffs = a - b
    n = len(diffs)
    mean_d = float(np.mean(diffs))
    var_d = float(np.var(diffs, ddof=1))
    std_d = math.sqrt(var_d)
    se_d = std_d / math.sqrt(n)
    df = n - 1
    
    t_stat = mean_d / se_d if se_d > 0 else 0.0
    p_val = float(2 * (1 - stats.t.cdf(abs(t_stat), df=df)))
    t_crit = float(stats.t.ppf(1 - alpha / 2, df=df))
    
    return {
        "test_type": "Paired Two-Sample t-test",
        "pairs_count": n,
        "mean_before": round(float(np.mean(b)), 4),
        "mean_after": round(float(np.mean(a)), 4),
        "mean_difference": round(mean_d, 4),
        "std_difference": round(std_d, 4),
        "standard_error": round(se_d, 4),
        "degrees_of_freedom": df,
        "t_statistic": float(t_stat),
        "critical_value": round(t_crit, 4),
        "p_value": p_val,
        "reject_null": bool(p_val < alpha)
    }
