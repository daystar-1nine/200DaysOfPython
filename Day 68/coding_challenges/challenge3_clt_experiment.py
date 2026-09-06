"""
Day 68 Coding Challenge 3: Multi-Scale CLT Experiment
Executes CLT across varying sample sizes from a non-normal (Uniform) population,
evaluating empirical standard error and distribution skewness.
"""
import numpy as np
import pandas as pd
from scipy import stats

def run_clt_experiment(
    population: np.ndarray,
    sample_sizes: list[int],
    n_samples: int = 5_000,
    seed: int = 42
) -> pd.DataFrame:
    """
    Run Central Limit Theorem experiment across multiple sample sizes.
    Tracks empirical mean, empirical SE, theoretical SE, skewness, and normality test.
    """
    rng = np.random.default_rng(seed)
    pop_mean = float(np.mean(population))
    pop_std = float(np.std(population))
    
    records = []
    for n in sample_sizes:
        means = np.array([np.mean(rng.choice(population, size=n, replace=False)) for _ in range(n_samples)])
        emp_mean = float(np.mean(means))
        emp_se = float(np.std(means, ddof=1))
        theo_se = pop_std / np.sqrt(n)
        sample_skew = float(stats.skew(means))
        # Normaltest p-value
        _, p_norm = stats.normaltest(means)
        
        records.append({
            "Sample Size (n)": n,
            "Empirical Mean": round(emp_mean, 4),
            "Theoretical SE": round(theo_se, 4),
            "Empirical SE": round(emp_se, 4),
            "Skewness": round(sample_skew, 4),
            "Is Normal (Skew ~ 0)": bool(abs(sample_skew) < 0.10)
        })
        
    return pd.DataFrame(records)

def main():
    print("=" * 70)
    print("  CHALLENGE 3: CLT CONVERGENCE EXPERIMENT (UNIFORM POPULATION)")
    print("=" * 70)
    rng = np.random.default_rng(42)
    # Continuous Uniform [0, 100]
    pop = rng.uniform(0.0, 100.0, size=50_000)
    
    sample_sizes = [2, 5, 10, 30, 50, 100]
    df_results = run_clt_experiment(pop, sample_sizes, n_samples=5_000)
    print(df_results.to_string(index=False))

if __name__ == "__main__":
    main()
