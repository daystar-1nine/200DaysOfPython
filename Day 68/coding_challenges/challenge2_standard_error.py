"""
Day 68 Coding Challenge 2: Standard Error & Finite Population Correction (FPC)
Calculates standard error with optional finite population correction factor.
"""
import numpy as np

def calculate_standard_error(
    population_std: float,
    sample_size: int,
    finite_population_size: int | None = None
) -> float:
    """
    Calculate standard error of the mean: SE = sigma / sqrt(n).
    If finite_population_size N is provided and n/N > 0.05, applies FPC: sqrt((N - n) / (N - 1)).
    """
    if population_std < 0:
        raise ValueError("Population standard deviation cannot be negative.")
    if sample_size <= 0:
        raise ValueError("Sample size must be strictly positive.")
        
    base_se = population_std / np.sqrt(sample_size)
    
    if finite_population_size is not None:
        if finite_population_size < sample_size:
            raise ValueError("Population size N cannot be smaller than sample size n.")
        if finite_population_size == 1:
            return 0.0
        # FPC factor
        fpc = np.sqrt((finite_population_size - sample_size) / (finite_population_size - 1))
        return float(base_se * fpc)
        
    return float(base_se)

def main():
    print("=" * 65)
    print("  CHALLENGE 2: STANDARD ERROR WITH FPC")
    print("=" * 65)
    sigma = 20.0
    n = 100
    
    se_inf = calculate_standard_error(sigma, n)
    se_fpc_large = calculate_standard_error(sigma, n, finite_population_size=10_000)
    se_fpc_small = calculate_standard_error(sigma, n, finite_population_size=200)
    
    print(f"Parameters: sigma = {sigma}, sample size n = {n}")
    print(f"1. Standard Error (Infinite Population):       {se_inf:.4f}")
    print(f"2. Standard Error (N = 10,000, n/N = 1%):      {se_fpc_large:.4f}")
    print(f"3. Standard Error (N = 200, n/N = 50% [FPC]):   {se_fpc_small:.4f}")
    print("Insight: When sampling a large fraction (>5%) of a finite population,")
    print("FPC reduces the standard error because less uncertainty remains.")

if __name__ == "__main__":
    main()
