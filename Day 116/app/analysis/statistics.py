"""
Statistical significance testing and bootstrap confidence interval computation for LLM evaluation.
"""
from typing import List, Dict, Any, Tuple
import math
import random
import numpy as np


def bootstrap_confidence_interval(
    scores: List[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42
) -> Dict[str, float]:
    """
    Computes a non-parametric bootstrap confidence interval for a metric:
    Resamples scores with replacement n_bootstrap times and computes percentiles.
    Returns:
      {"mean": float, "ci_lower": float, "ci_upper": float, "std_err": float}
    """
    if not scores:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "std_err": 0.0}

    n = len(scores)
    if n == 1:
        val = float(scores[0])
        return {"mean": val, "ci_lower": val, "ci_upper": val, "std_err": 0.0}

    rng = np.random.RandomState(seed)
    arr = np.array(scores, dtype=np.float64)

    # Vectorized bootstrap resampling
    sample_indices = rng.randint(0, n, size=(n_bootstrap, n))
    bootstrap_means = np.mean(arr[sample_indices], axis=1)

    alpha = 1.0 - ci
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    ci_lower = np.percentile(bootstrap_means, lower_pct)
    ci_upper = np.percentile(bootstrap_means, upper_pct)
    mean_val = float(np.mean(arr))
    std_err = float(np.std(bootstrap_means))

    return {
        "mean": round(mean_val, 4),
        "ci_lower": round(float(ci_lower), 4),
        "ci_upper": round(float(ci_upper), 4),
        "std_err": round(std_err, 4)
    }


def paired_permutation_test(
    scores_a: List[float],
    scores_b: List[float],
    n_permutations: int = 2000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Two-sided paired permutation test assessing whether the difference between
    Model A and Model B is statistically significant (H0: diff == 0).
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(f"Score arrays must have equal length: {len(scores_a)} vs {len(scores_b)}")

    n = len(scores_a)
    if n == 0:
        return {"observed_diff": 0.0, "p_value": 1.0, "is_significant": False}

    a = np.array(scores_a, dtype=np.float64)
    b = np.array(scores_b, dtype=np.float64)
    diffs = a - b
    observed_diff = float(np.mean(diffs))

    if abs(observed_diff) < 1e-9:
        return {"observed_diff": 0.0, "p_value": 1.0, "is_significant": False}

    rng = np.random.RandomState(seed)
    # Random sign flips: +1 or -1 for each pair
    flips = rng.choice([-1.0, 1.0], size=(n_permutations, n))
    permuted_diffs = np.mean(flips * diffs, axis=1)

    # Two-sided empirical p-value
    extreme_count = np.sum(np.abs(permuted_diffs) >= abs(observed_diff))
    p_value = float((extreme_count + 1) / (n_permutations + 1))

    return {
        "observed_diff": round(observed_diff, 4),
        "p_value": round(p_value, 5),
        "is_significant": bool(p_value < 0.05)
    }


def paired_t_test(
    scores_a: List[float],
    scores_b: List[float]
) -> Dict[str, Any]:
    """
    Computes standard paired student's t-test statistic and p-value.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("Score lengths must match.")

    n = len(scores_a)
    if n < 2:
        return {"t_stat": 0.0, "p_value": 1.0, "is_significant": False}

    a = np.array(scores_a, dtype=np.float64)
    b = np.array(scores_b, dtype=np.float64)
    d = a - b
    mean_d = np.mean(d)
    std_d = np.std(d, ddof=1)

    if std_d == 0.0:
        return {"t_stat": 0.0, "p_value": 1.0 if mean_d == 0 else 0.0, "is_significant": mean_d != 0}

    t_stat = mean_d / (std_d / math.sqrt(n))

    # Approximate two-tailed p-value using standard normal for moderate n
    z = abs(t_stat)
    # Complementary error function approximation for normal CDF
    p_value = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))

    return {
        "t_stat": round(float(t_stat), 4),
        "p_value": round(float(p_value), 5),
        "is_significant": bool(p_value < 0.05)
    }


def compute_cohens_d(scores_a: List[float], scores_b: List[float]) -> float:
    """
    Computes paired Cohen's d effect size: mean(diff) / std(diff).
    """
    if len(scores_a) != len(scores_b) or len(scores_a) < 2:
        return 0.0
    a = np.array(scores_a, dtype=np.float64)
    b = np.array(scores_b, dtype=np.float64)
    d = a - b
    std_d = np.std(d, ddof=1)
    if std_d == 0.0:
        return 0.0
    return round(float(np.mean(d) / std_d), 4)
