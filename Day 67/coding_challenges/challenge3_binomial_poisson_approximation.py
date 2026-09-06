"""
Day 67 Challenge 3: Poisson Approximation of the Binomial Distribution
Evaluates the Law of Rare Events by comparing Binomial(n=100, p=0.05)
against Poisson(lambda = np = 5.0).
"""
import numpy as np
import pandas as pd
from scipy.stats import binom, poisson

def compare_binomial_poisson(n: int = 100, p: float = 0.05, max_k: int = 15) -> pd.DataFrame:
    lam = n * p
    k_vals = np.arange(0, max_k + 1)
    
    binom_probs = binom.pmf(k_vals, n, p)
    poiss_probs = poisson.pmf(k_vals, mu=lam)
    abs_errors = np.abs(binom_probs - poiss_probs)
    
    df = pd.DataFrame({
        "k": k_vals,
        "Binomial PMF": np.round(binom_probs, 5),
        "Poisson PMF": np.round(poiss_probs, 5),
        "Absolute Error": np.round(abs_errors, 5)
    })
    return df

def main():
    print("=" * 65)
    print("  CHALLENGE 3: POISSON APPROXIMATION OF BINOMIAL (n=100, p=0.05)")
    print("=" * 65)
    df_comp = compare_binomial_poisson(n=100, p=0.05, max_k=12)
    print(df_comp.to_string(index=False))
    
    max_err = df_comp["Absolute Error"].max()
    tvd = 0.5 * df_comp["Absolute Error"].sum()
    print("-" * 65)
    print(f"Maximum Pointwise Error:      {max_err:.5f}")
    print(f"Total Variation Distance TVD: {tvd:.5f}")
    print("Conclusion: The Poisson distribution provides an exceptionally close")
    print("approximation when n is large and p is small (lambda = np = 5.0).")

if __name__ == "__main__":
    main()
