"""
Day 68 Task 4: Visualize Sampling Distribution of the Mean
Plots the histogram of sample means against the true population mean.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    rng = np.random.default_rng(42)
    population = rng.normal(loc=50.0, scale=10.0, size=100_000)
    
    n_sample_size = 30
    n_repetitions = 5_000
    sample_means = np.array([
        np.mean(rng.choice(population, size=n_sample_size, replace=False))
        for _ in range(n_repetitions)
    ])
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(sample_means, bins=40, color="#3B82F6", edgecolor="#1D4ED8", alpha=0.75, density=True, label="Sample Means (n=30)")
    ax.axvline(np.mean(population), color="#DC2626", linestyle="--", linewidth=2.5, label=f"Population Mean mu = {np.mean(population):.2f}")
    ax.axvline(np.mean(sample_means), color="#059669", linestyle=":", linewidth=2, label=f"Mean of Means = {np.mean(sample_means):.2f}")
    
    ax.set_title("Sampling Distribution of the Sample Mean (K=5,000, n=30)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Sample Mean (x_bar)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True)
    
    plt.tight_layout()
    chart_path = Path(__file__).resolve().parent.parent / "output" / "charts" / "sampling_distribution.png"
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(chart_path, dpi=200)
    plt.close(fig)
    
    print("=" * 60)
    print("  TASK 4: VISUALIZE SAMPLING DISTRIBUTION")
    print("=" * 60)
    print(f"Chart saved successfully to: {chart_path}")

if __name__ == "__main__":
    main()
