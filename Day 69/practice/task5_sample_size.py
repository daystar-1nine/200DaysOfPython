"""
Day 69 Task 5: Sample Size Determination for Targeted Precision
Implements required_sample_size_mean(population_std, margin_of_error, confidence).
"""
import math
from scipy import stats

def required_sample_size_mean(
    population_std: float,
    margin_of_error: float,
    confidence: float = 0.95
) -> int:
    """
    Calculate minimum required sample size to guarantee Margin of Error <= E.
    Formula: n = ceil((z * sigma / E)^2)
    """
    if population_std <= 0:
        raise ValueError("Population standard deviation must be strictly positive.")
    if margin_of_error <= 0:
        raise ValueError("Margin of error must be strictly positive.")
    if not (0.0 < confidence < 1.0):
        raise ValueError("Confidence level must be strictly between 0 and 1.")
        
    alpha = 1.0 - confidence
    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    n_exact = (z_crit * population_std / margin_of_error) ** 2
    return math.ceil(n_exact)

def main():
    print("=" * 60)
    print("  TASK 5: SAMPLE SIZE DETERMINATION")
    print("=" * 60)
    sigma = 20.0
    desired_error = 2.0
    conf = 0.95
    
    n_req = required_sample_size_mean(sigma, desired_error, conf)
    print(f"Parameters: Sigma = {sigma}, Desired Margin of Error = {desired_error}, Conf = {conf*100:.0f}%")
    print(f"Exact n required:   {((1.960 * sigma) / desired_error)**2:.2f}")
    print(f"Ceiling n required: {n_req}")
    print("-" * 60)
    print("Insight: Always round upward to ensure error stays strictly within contract.")

if __name__ == "__main__":
    main()
