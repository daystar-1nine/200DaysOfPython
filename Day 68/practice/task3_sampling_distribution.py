"""
Day 68 Task 3: Generate Sampling Distribution of the Mean
Draws 5,000 repeated samples of size n = 30, computing the mean of sample
means and comparing empirical standard error with theoretical sigma / sqrt(n).
"""
import numpy as np

def main():
    rng = np.random.default_rng(42)
    population = rng.normal(loc=50.0, scale=10.0, size=100_000)
    
    n_sample_size = 30
    n_repetitions = 5_000
    
    # Draw 5,000 samples and record their means
    sample_means = np.array([
        np.mean(rng.choice(population, size=n_sample_size, replace=False))
        for _ in range(n_repetitions)
    ])
    
    emp_mean = float(np.mean(sample_means))
    emp_se = float(np.std(sample_means, ddof=1))
    theo_se = float(np.std(population) / np.sqrt(n_sample_size))
    
    print("=" * 65)
    print(f"  TASK 3: SAMPLING DISTRIBUTION (K = {n_repetitions:,}, n = {n_sample_size})")
    print("=" * 65)
    print(f"Population Mean (mu):           {np.mean(population):.4f}")
    print(f"Mean of Sample Means (mu_xbar): {emp_mean:.4f}")
    print(f"Theoretical SE (sigma/sqrt(n)): {theo_se:.4f}")
    print(f"Experimental SE (s_xbar):       {emp_se:.4f}")
    print(f"Absolute SE Difference:         {abs(emp_se - theo_se):.4f}")

if __name__ == "__main__":
    main()
