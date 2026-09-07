"""
Day 69 Task 2: Z-Distribution Confidence Interval (Known Population Sigma)
Implements calculate_z_interval() returning lower_bound, upper_bound, and margin_of_error.
"""
import numpy as np
from scipy import stats

def calculate_z_interval(
    mean: float,
    population_std: float,
    sample_size: int,
    confidence: float = 0.95
) -> tuple[float, float, float]:
    """
    Calculate confidence interval for population mean when sigma is known.
    Returns: (lower_bound, upper_bound, margin_of_error)
    """
    if sample_size <= 0:
        raise ValueError("Sample size must be strictly positive.")
    if population_std < 0:
        raise ValueError("Population standard deviation cannot be negative.")
    if not (0.0 < confidence < 1.0):
        raise ValueError("Confidence level must be strictly between 0 and 1.")
        
    alpha = 1.0 - confidence
    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    standard_error = population_std / np.sqrt(sample_size)
    margin_of_error = float(z_crit * standard_error)
    
    lower_bound = float(mean - margin_of_error)
    upper_bound = float(mean + margin_of_error)
    
    return round(lower_bound, 4), round(upper_bound, 4), round(margin_of_error, 4)

def main():
    print("=" * 60)
    print("  TASK 2: Z-DISTRIBUTION CONFIDENCE INTERVAL")
    print("=" * 60)
    mean = 72.0
    pop_sd = 10.0
    n = 100
    conf = 0.95
    
    lower, upper, me = calculate_z_interval(mean, pop_sd, n, conf)
    print(f"Parameters: Mean = {mean}, Sigma = {pop_sd}, n = {n}, Conf = {conf*100:.0f}%")
    print(f"Margin of Error (ME): +/- {me:.4f}")
    print(f"95% Confidence Interval: [{lower:.4f}, {upper:.4f}]")
    print("-" * 60)
    print("Interpretation: Using a 95% procedure, we estimate true mu is in this span.")

if __name__ == "__main__":
    main()
