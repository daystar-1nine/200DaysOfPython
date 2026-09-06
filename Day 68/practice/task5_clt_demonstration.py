"""
Day 68 Task 5: Demonstrate Central Limit Theorem
Compares sampling distributions across sample sizes n = 5, 10, 30, 100
drawn from a heavily skewed Exponential population.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    rng = np.random.default_rng(42)
    # Heavily skewed population: Exponential(scale=10)
    population = rng.exponential(scale=10.0, size=100_000)
    
    sample_sizes = [5, 10, 30, 100]
    n_reps = 5_000
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, n in enumerate(sample_sizes):
        means = np.array([np.mean(rng.choice(population, size=n, replace=False)) for _ in range(n_reps)])
        ax = axes[i]
        ax.hist(means, bins=40, color="#2563EB", edgecolor="#1E3A8A", alpha=0.7, density=True)
        ax.axvline(np.mean(population), color="#DC2626", linestyle="--", linewidth=1.8, label=f"mu = {np.mean(population):.1f}")
        ax.set_title(f"Sample Size n = {n} (SE = {np.std(means):.2f})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Sample Mean")
        ax.set_ylabel("Density")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", frameon=True)
        
    plt.suptitle("Central Limit Theorem: Skewed Population -> Normal Sampling Distribution", fontsize=13, fontweight="bold")
    plt.tight_layout()
    chart_path = Path(__file__).resolve().parent.parent / "output" / "charts" / "clt_comparison.png"
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(chart_path, dpi=200)
    plt.close(fig)
    
    print("=" * 60)
    print("  TASK 5: CENTRAL LIMIT THEOREM DEMONSTRATION")
    print("=" * 60)
    print(f"Chart saved to: {chart_path}")
    print("Observation: At n=5, the distribution retains right-skewness.")
    print("At n=30 and n=100, the sampling distribution becomes smoothly Gaussian.")

if __name__ == "__main__":
    main()
