"""
Publication-grade visualizer for sampling, CLT, standard error, and bootstrap.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

# Style setup
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "figure.titlesize": 14,
    "figure.titleweight": "bold"
})

class Visualizer:
    def __init__(self, charts_dir: Path):
        self.charts_dir = Path(charts_dir)
        self.charts_dir.mkdir(parents=True, exist_ok=True)

    def plot_population_distributions(self, populations: dict[str, np.ndarray], filename: str = "population.png"):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        colors = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77"]

        for ax, (name, data), color in zip(axes, populations.items(), colors):
            sns.histplot(data, kde=True, ax=ax, color=color, stat="density", bins=40, alpha=0.6)
            mean_val = np.mean(data)
            median_val = np.median(data)
            ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.1f}")
            ax.axvline(median_val, color="black", linestyle=":", linewidth=1.5, label=f"Median: {median_val:.1f}")
            ax.set_title(f"Population: {name}")
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
            ax.legend()

        plt.suptitle("Day 68: Population Distributions (Underlying Realities)", y=1.02)
        plt.tight_layout()
        plt.savefig(self.charts_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_sample_vs_population(self, population: np.ndarray, sample: np.ndarray, filename: str = "sample.png"):
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.kdeplot(population, ax=ax, color="#2b5c8f", linewidth=2.5, label=f"Population (N={len(population):,})")
        sns.histplot(sample, stat="density", ax=ax, color="#e7298a", bins=25, alpha=0.5, label=f"Single Sample (n={len(sample)})")

        ax.axvline(np.mean(population), color="#2b5c8f", linestyle="--", label=f"True Mean: {np.mean(population):.2f}")
        ax.axvline(np.mean(sample), color="#e7298a", linestyle="-.", label=f"Sample Mean: {np.mean(sample):.2f}")

        ax.set_title("Single Sample vs Full Population Distribution")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.charts_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_sampling_distribution(self, pop: np.ndarray, sample_sizes: tuple[int, ...], n_samples: int = 5000, filename: str = "sampling_distribution.png"):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        rng = np.random.default_rng(42)
        pop_mean = np.mean(pop)

        for ax, n in zip(axes, sample_sizes):
            means = np.array([np.mean(rng.choice(pop, size=n, replace=False)) for _ in range(n_samples)])
            sns.histplot(means, kde=True, ax=ax, color="#386cb0", stat="density", bins=35)
            ax.axvline(pop_mean, color="red", linestyle="--", label=f"True Mean ({pop_mean:.1f})")
            ax.set_title(f"Sampling Distribution of Mean (n = {n})")
            ax.set_xlabel("Sample Mean")
            ax.set_ylabel("Density")
            ax.legend()

        plt.suptitle("Sampling Distributions of the Sample Mean Across Sample Sizes", y=1.02)
        plt.tight_layout()
        plt.savefig(self.charts_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_clt_comparison(self, skewed_pop: np.ndarray, sample_sizes: list[int], n_samples: int = 5000, filename: str = "clt_comparison.png"):
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        rng = np.random.default_rng(42)

        for ax, n in zip(axes, sample_sizes):
            means = np.array([np.mean(rng.choice(skewed_pop, size=n, replace=False)) for _ in range(n_samples)])
            skew = stats.skew(means)
            sns.histplot(means, kde=True, ax=ax, color="#1b9e77", stat="density", bins=30)
            ax.set_title(f"n = {n} | Skewness: {skew:.3f}")
            ax.set_xlabel("Sample Mean")
            ax.set_ylabel("Density")

        plt.suptitle("CLT Demonstration: Highly Skewed Population -> Gaussian Means", y=1.03)
        plt.tight_layout()
        plt.savefig(self.charts_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_standard_error(self, pop_std: float, sample_sizes: list[int], empirical_ses: list[float], filename: str = "standard_error.png"):
        fig, ax = plt.subplots(figsize=(9, 5))
        n_continuous = np.linspace(min(sample_sizes), max(sample_sizes) + 50, 200)
        theoretical_curve = pop_std / np.sqrt(n_continuous)

        ax.plot(n_continuous, theoretical_curve, color="#e41a1c", linewidth=2, label=r"Theoretical $SE = \sigma / \sqrt{n}$")
        ax.scatter(sample_sizes, empirical_ses, color="#377eb8", s=60, zorder=5, label="Empirical SE (Simulation)")

        ax.set_title("Standard Error vs Sample Size (Inverse Square Root Law)")
        ax.set_xlabel("Sample Size (n)")
        ax.set_ylabel("Standard Error (SE)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.charts_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_theoretical_vs_experimental(self, empirical_means: np.ndarray, pop_mean: float, theo_se: float, filename: str = "theoretical_vs_experimental.png"):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        # 1. Overlay Density vs Theoretical Gaussian
        sns.histplot(empirical_means, stat="density", ax=ax1, color="#4daf4a", bins=40, alpha=0.5, label="Empirical Draws")
        x = np.linspace(pop_mean - 4 * theo_se, pop_mean + 4 * theo_se, 300)
        ax1.plot(x, stats.norm.pdf(x, loc=pop_mean, scale=theo_se), color="red", linewidth=2, label="Theoretical N(mu, SE)")
        ax1.set_title("Empirical Distribution vs Theoretical Normal PDF")
        ax1.set_xlabel("Sample Mean")
        ax1.set_ylabel("Density")
        ax1.legend()

        # 2. Q-Q Plot
        stats.probplot(empirical_means, dist="norm", plot=ax2)
        ax2.set_title("Normal Q-Q Plot (Empirical Means)")
        ax2.get_lines()[0].set_color("#4daf4a")
        ax2.get_lines()[0].set_markersize(4)
        ax2.get_lines()[1].set_color("red")

        plt.tight_layout()
        plt.savefig(self.charts_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_bootstrap_ci(self, boot_dict: dict, filename: str = "bootstrap.png"):
        fig, ax = plt.subplots(figsize=(9, 5))
        dist = boot_dict["distribution"]
        ci_l = boot_dict["ci_lower"]
        ci_u = boot_dict["ci_upper"]
        obs = boot_dict["observed"]
        true_pop = boot_dict.get("true_population_mean")

        sns.histplot(dist, stat="density", kde=True, ax=ax, color="#984ea3", bins=40, alpha=0.6)
        ax.axvline(obs, color="blue", linestyle="-", linewidth=2, label=f"Sample Estimate: {obs:.2f}")
        ax.axvline(ci_l, color="black", linestyle="--", linewidth=1.5, label=f"95% CI Lower: {ci_l:.2f}")
        ax.axvline(ci_u, color="black", linestyle="--", linewidth=1.5, label=f"95% CI Upper: {ci_u:.2f}")
        if true_pop is not None:
            ax.axvline(true_pop, color="red", linestyle=":", linewidth=2, label=f"True Pop Mean: {true_pop:.2f}")

        ax.set_title(f"Bootstrap Resampling Distribution ({boot_dict['ci_level']*100:.0f}% Percentile CI)")
        ax.set_xlabel("Resampled Statistic")
        ax.set_ylabel("Density")
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.charts_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()
