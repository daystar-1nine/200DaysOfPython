"""
Day 69 Challenge 5: Comprehensive Population Parameter Estimation
Computes complete estimation profile: point estimate, sample std, SE, critical t, ME, and bounds.
"""
import numpy as np
from scipy import stats

def estimate_population_mean(data: np.ndarray | list[float], confidence: float = 0.95) -> dict:
    """
    Calculate comprehensive estimation metrics for an unknown population mean.
    """
    arr = np.asarray(data, dtype=float)
    if len(arr) < 2:
        raise ValueError("At least 2 observations required for estimation.")
    if not (0.0 < confidence < 1.0):
        raise ValueError("Confidence must be strictly between 0 and 1.")
        
    n = len(arr)
    df = n - 1
    mean_val = float(np.mean(arr))
    sample_std = float(np.std(arr, ddof=1))
    se = float(sample_std / np.sqrt(n))
    
    alpha = 1.0 - confidence
    crit_t = float(stats.t.ppf(1.0 - alpha / 2.0, df=df))
    me = float(crit_t * se)
    
    lower = float(mean_val - me)
    upper = float(mean_val + me)
    
    return {
        "sample_size": n,
        "degrees_of_freedom": df,
        "sample_mean": round(mean_val, 4),
        "sample_std": round(sample_std, 4),
        "critical_value": round(crit_t, 4),
        "standard_error": round(se, 4),
        "margin_of_error": round(me, 4),
        "lower_bound": round(lower, 4),
        "upper_bound": round(upper, 4),
        "confidence_level": confidence,
        "relative_margin_of_error_pct": round((me / mean_val) * 100.0, 2)
    }

def main():
    print("=" * 70)
    print("  CHALLENGE 5: COMPREHENSIVE POPULATION PARAMETER ESTIMATION")
    print("=" * 70)
    orders = [2480.0, 2350.0, 2510.0, 2420.0, 2600.0, 2390.0, 2550.0, 2470.0, 2430.0, 2580.0]
    res = estimate_population_mean(orders, confidence=0.95)
    
    for k, v in res.items():
        print(f"  {k:<32}: {v}")
    print("-" * 70)
    print(f"Business Summary: We are 95% confident that population AOV is Rs. {res['sample_mean']:.2f} +/- Rs. {res['margin_of_error']:.2f}")

if __name__ == "__main__":
    main()
