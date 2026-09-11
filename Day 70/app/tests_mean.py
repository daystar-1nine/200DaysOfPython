"""
Parametric hypothesis testing for population means: 1-sample Z-test and 1-sample t-test.
"""

import math
from typing import Sequence, Union
import numpy as np
from scipy import stats

try:
    from app.validator import validate_numeric_array, validate_alpha, validate_alternative
    from app.stats_calc import compute_sample_stats
except (ImportError, ModuleNotFoundError):
    from validator import validate_numeric_array, validate_alpha, validate_alternative
    from stats_calc import compute_sample_stats

def one_sample_t_test(
    data: Sequence[Union[int, float]],
    mu_0: float,
    alpha: float = 0.05,
    alternative: str = "two-sided"
) -> dict:
    arr = validate_numeric_array(data, min_len=2)
    alpha = validate_alpha(alpha)
    alt = validate_alternative(alternative)
    
    st = compute_sample_stats(arr)
    n = st["n"]
    x_bar = st["mean"]
    s = st["std"]
    se = st["se"]
    df = st["df"]
    
    if se == 0:
        raise ValueError("Sample standard deviation is zero; cannot conduct t-test.")
        
    t_stat = (x_bar - mu_0) / se
    
    if alt == "two-sided":
        t_crit = float(stats.t.ppf(1 - alpha / 2, df=df))
        p_val = float(2 * (1 - stats.t.cdf(abs(t_stat), df=df)))
        reject = abs(t_stat) >= t_crit
    elif alt == "greater":
        t_crit = float(stats.t.ppf(1 - alpha, df=df))
        p_val = float(1 - stats.t.cdf(t_stat, df=df))
        reject = t_stat >= t_crit
    else:  # less
        t_crit = float(stats.t.ppf(alpha, df=df))
        p_val = float(stats.t.cdf(t_stat, df=df))
        reject = t_stat <= t_crit
        
    # (1 - alpha) CI
    t_ci_crit = float(stats.t.ppf(1 - alpha / 2, df=df))
    ci_lower = x_bar - t_ci_crit * se
    ci_upper = x_bar + t_ci_crit * se
    
    return {
        "n": n,
        "sample_mean": x_bar,
        "sample_std": s,
        "standard_error": se,
        "degrees_of_freedom": df,
        "hypothesized_mean": mu_0,
        "test_statistic": t_stat,
        "critical_value": t_crit,
        "p_value": p_val,
        "alpha": alpha,
        "alternative": alt,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "reject_null": reject
    }

def one_sample_z_test(
    sample_mean: float,
    n: int,
    mu_0: float,
    sigma: float,
    alpha: float = 0.05,
    alternative: str = "two-sided"
) -> dict:
    if n <= 0 or sigma <= 0:
        raise ValueError("Sample size n and sigma must be positive.")
    alpha = validate_alpha(alpha)
    alt = validate_alternative(alternative)
    
    se = sigma / math.sqrt(n)
    z_stat = (sample_mean - mu_0) / se
    
    if alt == "two-sided":
        z_crit = float(stats.norm.ppf(1 - alpha / 2))
        p_val = float(2 * (1 - stats.norm.cdf(abs(z_stat))))
        reject = abs(z_stat) >= z_crit
    elif alt == "greater":
        z_crit = float(stats.norm.ppf(1 - alpha))
        p_val = float(1 - stats.norm.cdf(z_stat))
        reject = z_stat >= z_crit
    else:  # less
        z_crit = float(stats.norm.ppf(alpha))
        p_val = float(stats.norm.cdf(z_stat))
        reject = z_stat <= z_crit
        
    z_ci_crit = float(stats.norm.ppf(1 - alpha / 2))
    ci_lower = sample_mean - z_ci_crit * se
    ci_upper = sample_mean + z_ci_crit * se
    
    return {
        "n": n,
        "sample_mean": sample_mean,
        "known_sigma": sigma,
        "standard_error": se,
        "hypothesized_mean": mu_0,
        "test_statistic": z_stat,
        "critical_value": z_crit,
        "p_value": p_val,
        "alpha": alpha,
        "alternative": alt,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "reject_null": reject
    }
