"""
One-sample proportion Z-tests and binomial approximation checks.
"""

import math
from scipy import stats

try:
    from app.validator import validate_counts, validate_proportion, validate_alpha, validate_alternative
except (ImportError, ModuleNotFoundError):
    from validator import validate_counts, validate_proportion, validate_alpha, validate_alternative

def one_sample_proportion_test(
    successes: int,
    trials: int,
    p_0: float,
    alpha: float = 0.05,
    alternative: str = "two-sided"
) -> dict:
    successes, trials = validate_counts(successes, trials)
    p_0 = validate_proportion(p_0, "Null proportion p_0")
    alpha = validate_alpha(alpha)
    alt = validate_alternative(alternative)
    
    p_hat = successes / trials
    expected_success = trials * p_0
    expected_failure = trials * (1.0 - p_0)
    normality_valid = (expected_success >= 10.0 and expected_failure >= 10.0)
    
    # Null standard error (using p_0)
    se_null = math.sqrt((p_0 * (1.0 - p_0)) / trials)
    z_stat = (p_hat - p_0) / se_null
    
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
        
    # Wald Confidence Interval for proportion using sample p_hat
    se_sample = math.sqrt((p_hat * (1.0 - p_hat)) / trials) if 0 < p_hat < 1 else se_null
    z_ci_crit = float(stats.norm.ppf(1 - alpha / 2))
    ci_lower = max(0.0, p_hat - z_ci_crit * se_sample)
    ci_upper = min(1.0, p_hat + z_ci_crit * se_sample)
    
    return {
        "trials": trials,
        "successes": successes,
        "sample_proportion": p_hat,
        "hypothesized_proportion": p_0,
        "standard_error": se_null,
        "test_statistic": z_stat,
        "critical_value": z_crit,
        "p_value": p_val,
        "alpha": alpha,
        "alternative": alt,
        "normality_assumption_met": normality_valid,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "reject_null": reject
    }
