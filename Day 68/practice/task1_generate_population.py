"""
Day 68 Task 1: Generate Population
Generates a large population (N = 100,000) from Normal(mu=50, sigma=10),
computing the exact population parameters mu and sigma.
"""
import numpy as np

def main():
    rng = np.random.default_rng(42)
    
    # Generate population of N = 100,000 observations
    population = rng.normal(loc=50.0, scale=10.0, size=100_000)
    
    pop_mean = float(np.mean(population))
    pop_std = float(np.std(population))
    
    print("=" * 60)
    print("  TASK 1: POPULATION GENERATION (N = 100,000)")
    print("=" * 60)
    print(f"Population Mean (mu):     {pop_mean:.4f}")
    print(f"Population Std Dev (sigma): {pop_std:.4f}")

if __name__ == "__main__":
    main()
