"""
Day 69 Task 3: T-Distribution Confidence Interval (Unknown Population Sigma)
Implements calculate_t_interval(data, confidence).
"""
import numpy as np
from scipy import stats

def calculate_t_interval(data: np.ndarray, confidence: float = 0.95) -> dict:
    """
    Calculate confidence interval for mean with unknown sigma using Student's t-distribution.
    """
    if len(data) < 2:
        raise ValueError("At least 2 data points are required for a t-interval.")
    if not (0.0 < confidence < 1.0):
        raise ValueError("Confidence level must be strictly between 0 and 1.")
        
    n = len(data)
    df = n - 1
    mean_val = float(np.mean(data))
    sample_std = float(np.std(data, ddof=1))
    
    alpha = 1.0 - confidence
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=df))
    standard_error = sample_std / np.sqrt(n)
    margin_of_error = float(t_crit * standard_error)
    
    lower_bound = float(mean_val - margin_of_error)
    upper_bound = float(mean_val + margin_of_error)
    
    return {
        "sample_size": n,
        "degrees_of_freedom": df,
        "sample_mean": round(mean_val, 4),
        "sample_std": round(sample_std, 4),
        "critical_t": round(t_crit, 4),
        "standard_error": round(standard_error, 4),
        "margin_of_error": round(margin_of_error, 4),
        "lower_bound": round(lower_bound, 4),
        "upper_bound": round(upper_bound, 4)
    }

def main():
    print("=" * 60)
    print("  TASK 3: T-DISTRIBUTION CONFIDENCE INTERVAL")
    print("=" * 60)
    data = np.array([65.0, 72.0, 70.0, 68.0, 75.0, 71.0, 69.0, 73.0, 76.0, 67.0])
    res = calculate_t_interval(data, confidence=0.95)
    
    print(f"Sample Size (n):          {res['sample_size']} (df = {res['degrees_of_freedom']})")
    print(f"Sample Mean:              {res['sample_mean']:.2f}")
    print(f"Sample Std (s):           {res['sample_std']:.2f}")
    print(f"Critical t (df={res['degrees_of_freedom']}):       {res['critical_t']:.4f}")
    print(f"Standard Error:           {res['standard_error']:.4f}")
    print(f"Margin of Error:          +/- {res['margin_of_error']:.4f}")
    print(f"95% Confidence Interval:  [{res['lower_bound']:.2f}, {res['upper_bound']:.2f}]")

if __name__ == "__main__":
    main()
