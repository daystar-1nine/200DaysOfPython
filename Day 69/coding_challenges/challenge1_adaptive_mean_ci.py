"""
Day 69 Challenge 1: Adaptive Mean Confidence Interval
Automatically uses Z-distribution if population standard deviation is provided,
or Student's T-distribution if population standard deviation is unknown.
"""
from typing import Optional
import numpy as np
from scipy import stats

def confidence_interval_mean(
    data: np.ndarray | list[float],
    confidence: float = 0.95,
    population_std: Optional[float] = None
) -> dict:
    """
    Compute confidence interval for population mean, automatically selecting
    Z-distribution (known sigma) or T-distribution (unknown sigma).
    """
    arr = np.asarray(data, dtype=float)
    if len(arr) == 0:
        raise ValueError("Data array cannot be empty.")
    if not (0.0 < confidence < 1.0):
        raise ValueError("Confidence level must be strictly between 0 and 1.")
        
    n = len(arr)
    mean_val = float(np.mean(arr))
    alpha = 1.0 - confidence
    
    if population_std is not None:
        if population_std < 0:
            raise ValueError("Population standard deviation cannot be negative.")
        method = "Z-Distribution (Known Sigma)"
        df = None
        crit_val = float(stats.norm.ppf(1.0 - alpha / 2.0))
        se = float(population_std / np.sqrt(n))
    else:
        if n < 2:
            raise ValueError("At least 2 observations required for t-distribution estimation.")
        method = "T-Distribution (Unknown Sigma)"
        df = n - 1
        crit_val = float(stats.t.ppf(1.0 - alpha / 2.0, df=df))
        s = float(np.std(arr, ddof=1))
        se = float(s / np.sqrt(n))
        
    me = float(crit_val * se)
    lower = float(mean_val - me)
    upper = float(mean_val + me)
    
    return {
        "sample_size": n,
        "sample_mean": round(mean_val, 4),
        "method": method,
        "degrees_of_freedom": df,
        "critical_value": round(crit_val, 4),
        "standard_error": round(se, 4),
        "margin_of_error": round(me, 4),
        "lower_bound": round(lower, 4),
        "upper_bound": round(upper, 4),
        "confidence_level": confidence
    }

def main():
    print("=" * 65)
    print("  CHALLENGE 1: ADAPTIVE MEAN CONFIDENCE INTERVAL")
    print("=" * 65)
    sample = [24.5, 27.0, 26.2, 29.1, 23.8, 28.4, 25.9, 27.5, 26.8, 30.2]
    
    # 1. Unknown sigma -> Auto selects T
    res_t = confidence_interval_mean(sample, confidence=0.95)
    print(f"1. Unknown Sigma: {res_t['method']}")
    print(f"   Mean: {res_t['sample_mean']} | ME: +/- {res_t['margin_of_error']} | CI: [{res_t['lower_bound']}, {res_t['upper_bound']}]")
    
    # 2. Known sigma -> Auto selects Z
    res_z = confidence_interval_mean(sample, confidence=0.95, population_std=2.0)
    print(f"\n2. Known Sigma:   {res_z['method']}")
    print(f"   Mean: {res_z['sample_mean']} | ME: +/- {res_z['margin_of_error']} | CI: [{res_z['lower_bound']}, {res_z['upper_bound']}]")

if __name__ == "__main__":
    main()
