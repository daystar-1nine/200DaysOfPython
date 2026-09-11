"""
Confidence intervals for difference between two independent means, paired differences, and proportions.
"""

import math
from typing import Sequence, Union
import numpy as np
from scipy import stats

try:
    from app.validator import validate_alpha
except (ImportError, ModuleNotFoundError):
    from validator import validate_alpha

def mean_difference_ci(
    group_a: Sequence[Union[int, float]],
    group_b: Sequence[Union[int, float]],
    confidence: float = 0.95
) -> dict:
    alpha = 1.0 - confidence
    alpha = validate_alpha(alpha)
    
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    n_a, n_b = len(a), len(b)
    
    mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
    var_a, var_b = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    diff = mean_b - mean_a
    
    se = math.sqrt((var_a / n_a) + (var_b / n_b))
    va, vb = var_a / n_a, var_b / n_b
    df = ((va + vb)**2) / ((va**2) / (n_a - 1) + (vb**2) / (n_b - 1))
    
    t_crit = float(stats.t.ppf(1 - alpha / 2, df=df))
    margin = t_crit * se
    
    return {
        "difference": round(diff, 4),
        "standard_error": round(se, 4),
        "margin_of_error": round(margin, 4),
        "ci_lower": round(diff - margin, 4),
        "ci_upper": round(diff + margin, 4),
        "confidence_level": confidence,
        "excludes_zero": bool((diff - margin > 0) or (diff + margin < 0))
    }

def proportion_difference_ci(
    p_control: float,
    n_control: int,
    p_treatment: float,
    n_treatment: int,
    confidence: float = 0.95
) -> dict:
    alpha = 1.0 - confidence
    alpha = validate_alpha(alpha)
    
    diff = p_treatment - p_control
    # Unpooled standard error for confidence interval
    se_unpool = math.sqrt((p_control * (1.0 - p_control) / n_control) + (p_treatment * (1.0 - p_treatment) / n_treatment))
    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    margin = z_crit * se_unpool
    
    return {
        "difference": round(diff, 4),
        "standard_error": round(se_unpool, 6),
        "margin_of_error": round(margin, 4),
        "ci_lower": round(diff - margin, 4),
        "ci_upper": round(diff + margin, 4),
        "confidence_level": confidence,
        "excludes_zero": bool((diff - margin > 0) or (diff + margin < 0))
    }
