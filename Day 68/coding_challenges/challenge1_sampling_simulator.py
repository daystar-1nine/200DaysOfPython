"""
Day 68 Coding Challenge 1: Sampling Simulator
Implements simulate_sampling() to draw repeated samples and evaluate
the empirical sampling distribution of the sample mean.
"""
import numpy as np

def simulate_sampling(
    population: np.ndarray,
    sample_size: int,
    n_samples: int,
    replace: bool = False,
    seed: int = 42
) -> dict:
    """
    Simulate drawing repeated random samples from a population.
    Returns array of sample means, empirical statistics, and theoretical standard error.
    """
    if sample_size <= 0 or sample_size > len(population) and not replace:
        raise ValueError(f"Invalid sample_size {sample_size} for population size {len(population)} (replace={replace}).")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
        
    rng = np.random.default_rng(seed)
    
    # Generate repeated samples
    sample_means = np.empty(n_samples, dtype=float)
    for i in range(n_samples):
        sample = rng.choice(population, size=sample_size, replace=replace)
        sample_means[i] = np.mean(sample)
        
    pop_mean = float(np.mean(population))
    pop_std = float(np.std(population))
    emp_mean = float(np.mean(sample_means))
    emp_se = float(np.std(sample_means, ddof=1))
    theo_se = pop_std / np.sqrt(sample_size)
    
    return {
        "sample_size": sample_size,
        "n_samples": n_samples,
        "sample_means": sample_means,
        "population_mean": round(pop_mean, 4),
        "population_std": round(pop_std, 4),
        "empirical_mean": round(emp_mean, 4),
        "empirical_se": round(emp_se, 4),
        "theoretical_se": round(theo_se, 4),
        "abs_error_mean": round(abs(emp_mean - pop_mean), 4),
        "abs_error_se": round(abs(emp_se - theo_se), 4)
    }

def main():
    print("=" * 65)
    print("  CHALLENGE 1: MODULAR SAMPLING SIMULATOR")
    print("=" * 65)
    rng = np.random.default_rng(42)
    pop = rng.normal(loc=100.0, scale=15.0, size=50_000)
    
    res = simulate_sampling(pop, sample_size=50, n_samples=5_000)
    print(f"Population Mean:  {res['population_mean']} | Std Dev: {res['population_std']}")
    print(f"Sample Size (n):  {res['sample_size']} | Repetitions: {res['n_samples']:,}")
    print(f"Empirical Mean:   {res['empirical_mean']} (Error: {res['abs_error_mean']})")
    print(f"Theoretical SE:   {res['theoretical_se']}")
    print(f"Empirical SE:     {res['empirical_se']} (Error: {res['abs_error_se']})")

if __name__ == "__main__":
    main()
