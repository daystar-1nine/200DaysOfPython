"""
Day 68 Task 2: Take One Random Sample
Draws a single random sample (n = 100) from the population,
comparing sample statistics (x_bar, s) with population parameters (mu, sigma).
"""
import numpy as np

def main():
    rng = np.random.default_rng(42)
    population = rng.normal(loc=50.0, scale=10.0, size=100_000)
    
    # Draw a single sample of size n = 100 without replacement
    sample = rng.choice(population, size=100, replace=False)
    
    sample_mean = float(np.mean(sample))
    sample_std = float(np.std(sample, ddof=1))
    
    print("=" * 60)
    print("  TASK 2: SINGLE SAMPLE COMPARISON (n = 100)")
    print("=" * 60)
    print(f"Population Mean (mu):       {np.mean(population):.4f}")
    print(f"Sample Mean (x_bar):        {sample_mean:.4f} (Diff: {abs(sample_mean - np.mean(population)):.4f})")
    print(f"Population Std Dev (sigma): {np.std(population):.4f}")
    print(f"Sample Std Dev (s):         {sample_std:.4f} (Diff: {abs(sample_std - np.std(population)):.4f})")

if __name__ == "__main__":
    main()
