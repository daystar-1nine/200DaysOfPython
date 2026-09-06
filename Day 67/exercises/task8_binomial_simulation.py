"""
Day 67 Task 8: Binomial Simulation vs Theoretical PMF
Simulates 10,000 experiments of n = 20, p = 0.5, comparing observed
success frequencies directly against theoretical Binomial probabilities.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binom

def main():
    n = 20
    p = 0.5
    n_sims = 10_000
    
    # 1. Theoretical PMF
    k_vals = np.arange(0, n + 1)
    theo_probs = binom.pmf(k_vals, n, p)
    
    # 2. Stochastic Simulation
    sim_counts = binom.rvs(n=n, p=p, size=n_sims, random_state=42)
    emp_freqs = np.bincount(sim_counts, minlength=n + 1) / n_sims
    
    # 3. Visualization
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.35
    
    ax.bar(k_vals - width/2, theo_probs, width=width, label="Theoretical PMF", color="#2563EB", alpha=0.85)
    ax.bar(k_vals + width/2, emp_freqs, width=width, label=f"Simulated (N={n_sims:,})", color="#10B981", alpha=0.85)
    
    ax.set_title(f"Binomial Theory vs Simulation (n={n}, p={p})", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Successes (k)", fontsize=11)
    ax.set_ylabel("Probability / Relative Frequency", fontsize=11)
    ax.set_xticks(k_vals)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True)
    
    plt.tight_layout()
    chart_path = Path(__file__).resolve().parent.parent / "output" / "charts" / "binomial_simulation.png"
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(chart_path, dpi=200)
    plt.close(fig)
    
    print("=" * 60)
    print("  TASK 8: BINOMIAL SIMULATION VS THEORETICAL PMF")
    print("=" * 60)
    print(f"Peak Success Count (k = 10):")
    print(f"  Theoretical P(X=10): {theo_probs[10]:.5f}")
    print(f"  Simulated Relative:  {emp_freqs[10]:.5f}")
    print(f"  Absolute Difference: {abs(emp_freqs[10] - theo_probs[10]):.5f}")
    print(f"Chart saved to: {chart_path}")

if __name__ == "__main__":
    main()
