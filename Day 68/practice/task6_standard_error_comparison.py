"""
Day 68 Task 6: Compare Standard Errors Across Sample Sizes
Demonstrates the 1 / sqrt(n) scaling law by comparing theoretical
and experimental standard errors for n = [10, 25, 50, 100, 500].
"""
import numpy as np
import pandas as pd

def main():
    rng = np.random.default_rng(42)
    population = rng.normal(loc=50.0, scale=10.0, size=100_000)
    pop_std = float(np.std(population))
    
    sample_sizes = [10, 25, 50, 100, 500]
    n_reps = 3_000
    records = []
    
    for n in sample_sizes:
        means = np.array([np.mean(rng.choice(population, size=n, replace=False)) for _ in range(n_reps)])
        theo_se = pop_std / np.sqrt(n)
        exp_se = float(np.std(means, ddof=1))
        
        records.append({
            "Sample Size (n)": n,
            "Theoretical SE": round(theo_se, 4),
            "Experimental SE": round(exp_se, 4),
            "Abs Difference": round(abs(exp_se - theo_se), 4),
            "Rel Error (%)": round(abs(exp_se - theo_se) / theo_se * 100, 2)
        })
        
    df_se = pd.DataFrame(records)
    print("=" * 70)
    print("  TASK 6: STANDARD ERROR VS SAMPLE SIZE TABLE")
    print("=" * 70)
    print(df_se.to_string(index=False))

if __name__ == "__main__":
    main()
