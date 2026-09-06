"""
Day 67 Task 6: Normal Monte Carlo Sampling & Theoretical Comparison
Generates 10,000 samples from N(70, 10) and compares empirical vs theoretical statistics.
"""
import numpy as np
from scipy.stats import norm

def main():
    mu = 70.0
    sigma = 10.0
    n_samples = 10_000
    
    # Deterministic seed for reproducible simulation
    samples = norm.rvs(loc=mu, scale=sigma, size=n_samples, random_state=42)
    
    # Empirical Metrics
    emp_mean = float(np.mean(samples))
    emp_std = float(np.std(samples, ddof=1))
    emp_median = float(np.median(samples))
    emp_p10 = float(np.percentile(samples, 10))
    emp_p90 = float(np.percentile(samples, 90))
    
    # Theoretical Metrics
    theo_mean = mu
    theo_std = sigma
    theo_median = mu
    theo_p10 = float(norm.ppf(0.10, loc=mu, scale=sigma))
    theo_p90 = float(norm.ppf(0.90, loc=mu, scale=sigma))
    
    print("=" * 70)
    print(f"  TASK 6: NORMAL SIMULATION (N = {n_samples:,} Samples)")
    print("=" * 70)
    print(f"{'Metric':<18} | {'Theoretical':<14} | {'Simulated':<14} | {'Absolute Error':<14}")
    print("-" * 70)
    print(f"{'Mean':<18} | {theo_mean:<14.4f} | {emp_mean:<14.4f} | {abs(emp_mean - theo_mean):<14.4f}")
    print(f"{'Std Deviation':<18} | {theo_std:<14.4f} | {emp_std:<14.4f} | {abs(emp_std - theo_std):<14.4f}")
    print(f"{'Median':<18} | {theo_median:<14.4f} | {emp_median:<14.4f} | {abs(emp_median - theo_median):<14.4f}")
    print(f"{'10th Percentile':<18} | {theo_p10:<14.4f} | {emp_p10:<14.4f} | {abs(emp_p10 - theo_p10):<14.4f}")
    print(f"{'90th Percentile':<18} | {theo_p90:<14.4f} | {emp_p90:<14.4f} | {abs(emp_p90 - theo_p90):<14.4f}")
    print("-" * 70)
    print("Insight: Empirical metrics converge tightly to theoretical parameters")
    print(f"within standard error bound SE = sigma / sqrt(N) = {sigma / np.sqrt(n_samples):.4f}.")

if __name__ == "__main__":
    main()
