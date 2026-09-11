"""
Challenge 3: Robust One-Sample Proportion Hypothesis Test
Includes success-failure checks, standard error under H0, Wald CI, and decision.
"""

import math
from scipy import stats

def one_sample_proportion_engine(
    successes: int,
    trials: int,
    p_0: float,
    alpha: float = 0.05,
    alternative: str = "two-sided"
) -> dict:
    if trials <= 0:
        raise ValueError("Trials must be positive integer.")
    if not (0 <= successes <= trials):
        raise ValueError("Successes must be between 0 and trials.")
    if not (0 < p_0 < 1):
        raise ValueError("Hypothesized proportion p_0 must be strictly between 0 and 1.")
    if not (0 < alpha < 1):
        raise ValueError("Alpha must be between 0 and 1.")
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError("Alternative must be 'two-sided', 'greater', or 'less'.")

    p_hat = successes / trials
    exp_succ = trials * p_0
    exp_fail = trials * (1 - p_0)
    norm_valid = (exp_succ >= 10.0 and exp_fail >= 10.0)
    
    # Null standard error
    se_null = math.sqrt((p_0 * (1 - p_0)) / trials)
    z_stat = (p_hat - p_0) / se_null
    
    if alternative == "two-sided":
        z_crit = float(stats.norm.ppf(1 - alpha / 2))
        p_val = float(2 * (1 - stats.norm.cdf(abs(z_stat))))
        reject = abs(z_stat) >= z_crit
    elif alternative == "greater":
        z_crit = float(stats.norm.ppf(1 - alpha))
        p_val = float(1 - stats.norm.cdf(z_stat))
        reject = z_stat >= z_crit
    else:
        z_crit = float(stats.norm.ppf(alpha))
        p_val = float(stats.norm.cdf(z_stat))
        reject = z_stat <= z_crit

    # Sample standard error for Wald CI
    se_sample = math.sqrt((p_hat * (1 - p_hat)) / trials) if 0 < p_hat < 1 else se_null
    z_ci_crit = float(stats.norm.ppf(1 - alpha / 2))
    ci_lower = max(0.0, p_hat - z_ci_crit * se_sample)
    ci_upper = min(1.0, p_hat + z_ci_crit * se_sample)
    
    return {
        "successes": successes,
        "trials": trials,
        "sample_proportion": round(p_hat, 4),
        "hypothesized_proportion": p_0,
        "expected_successes": round(exp_succ, 1),
        "expected_failures": round(exp_fail, 1),
        "normality_assumption_met": norm_valid,
        "z_statistic": round(z_stat, 4),
        "critical_value": round(z_crit, 4),
        "p_value": p_val,
        "alpha": alpha,
        "alternative": alternative,
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "reject_null": reject,
        "decision": "Reject Null Hypothesis" if reject else "Fail to Reject Null Hypothesis"
    }

if __name__ == "__main__":
    res = one_sample_proportion_engine(successes=145, trials=1000, p_0=0.12, alpha=0.05, alternative="greater")
    print("=== Challenge 3: Proportion Test Engine ===")
    for k, v in res.items():
        print(f"  {k}: {v}")
