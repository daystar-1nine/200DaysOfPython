"""
Day 67 Task 4: Normal Distribution Quantiles / Percentiles
Computes percentiles for exam scores N(mu=70, sigma=10) via norm.ppf().
"""
from scipy.stats import norm

def main():
    mu = 70
    sigma = 10
    rv = norm(loc=mu, scale=sigma)
    
    p50 = float(rv.ppf(0.50))
    p90 = float(rv.ppf(0.90))
    p95 = float(rv.ppf(0.95))
    p99 = float(rv.ppf(0.99))
    
    print("=" * 60)
    print("  TASK 4: NORMAL PERCENTILES (mu = 70, sigma = 10)")
    print("=" * 60)
    print(f"50th Percentile (Median): {p50:.2f}")
    print(f"90th Percentile (Top 10%):{p90:.2f}")
    print(f"95th Percentile (Top 5%): {p95:.2f}")
    print(f"99th Percentile (Top 1%): {p99:.2f}")

if __name__ == "__main__":
    main()
