"""
Day 67 Challenge 4: De Moivre-Laplace Normal Approximation of the Binomial
Examines convergence of Binomial(n, p=0.5) to a Normal bell curve as n increases (10, 50, 100).
"""
import numpy as np
from scipy.stats import binom, norm

def evaluate_convergence(n_trials_list: list[int], p: float = 0.5):
    print("=" * 70)
    print(f"  CHALLENGE 4: NORMAL APPROXIMATION TO BINOMIAL (p = {p})")
    print("=" * 70)
    
    for n in n_trials_list:
        mu = n * p
        sigma = np.sqrt(n * p * (1 - p))
        
        # Test central probability P(mu - 1 <= X <= mu + 1)
        k_low = int(np.floor(mu - 1))
        k_high = int(np.ceil(mu + 1))
        
        exact_binom = float(binom.cdf(k_high, n, p) - binom.cdf(k_low - 1, n, p))
        # With continuity correction
        approx_norm = float(norm.cdf(k_high + 0.5, loc=mu, scale=sigma) - norm.cdf(k_low - 0.5, loc=mu, scale=sigma))
        diff = abs(exact_binom - approx_norm)
        
        print(f"n = {n:<3} | mu = {mu:<5.1f} | sigma = {sigma:<5.2f} | "
              f"Binomial: {exact_binom:.4f} | Normal (w/ CC): {approx_norm:.4f} | Diff: {diff:.5f}")

def main():
    evaluate_convergence([10, 30, 50, 100, 500], p=0.5)
    print("-" * 70)
    print("Insight: As n increases, the discrete Binomial probability mass converges")
    print("smoothly to the continuous Normal density according to the Central Limit Theorem.")

if __name__ == "__main__":
    main()
