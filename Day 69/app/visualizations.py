"""
Publication-grade visualizer generating 7 high-resolution diagnostic charts.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# Global styling
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

    def plot_sample_distribution(self, data: np.ndarray, filename: str = "sample_distribution.png"):
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.histplot(data, kde=True, ax=ax, color="#2b5c8f", bins=35, stat="density", alpha=0.6)
        
        mean_val = float(np.mean(data))
        median_val = float(np.median(data))
        ax.axvline(mean_val, color="red", linestyle="--", linewidth=2, label=f"Mean: Rs. {mean_val:.2f}")
        ax.axvline(median_val, color="black", linestyle=":", linewidth=2, label=f"Median: Rs. {median_val:.2f}")
        
        ax.set_title(f"Sample Observations Distribution (n = {len(data):,})")
        ax.set_xlabel("Order Value (Rs.)")
        ax.set_ylabel("Probability Density")
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(self.charts_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_confidence_interval(self, mean: float, lower: float, upper: float, filename: str = "confidence_interval.png"):
        fig, ax = plt.subplots(figsize=(8, 3.5))
        me = upper - mean
        ax.errorbar(mean, 0, xerr=me, fmt="o", color="#d95f02", ecolor="#d95f02",
                    elinewidth=3, capsize=8, capthick=2, markersize=10, label="95% Confidence Interval")
        
        ax.text(mean, 0.12, f"Point Estimate: Rs. {mean:.2f}", ha="center", fontweight="bold", color="#d95f02")
        ax.text(lower, -0.15, f"Lower: Rs. {lower:.2f}", ha="center", fontsize=9, fontweight="bold")
        ax.text(upper, -0.15, f"Upper: Rs. {upper:.2f}", ha="center", fontsize=9, fontweight="bold")
        
        ax.set_ylim(-0.4, 0.4)
        ax.set_yticks([])
        ax.set_xlabel("Order Value (Rs.)", fontweight="bold")
        ax.set_title("Point Estimate & 95% Confidence Interval Span")
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(self.charts_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_confidence_levels(self, mean: float, levels_dict: dict, filename: str = "confidence_levels.png"):
        fig, ax = plt.subplots(figsize=(9, 4.5))
        colors = ["#1b9e77", "#2b5c8f", "#7570b3"]
        y_pos = np.arange(len(levels_dict))
        labels = list(levels_dict.keys())
        
        for y, (lvl, d), col in zip(y_pos, levels_dict.items(), colors):
            me = d["margin_of_error"]
            ax.errorbar(mean, y, xerr=me, fmt="s", color=col, ecolor=col,
                        elinewidth=3, capsize=6, capthick=2, markersize=8, label=f"{lvl} (Width: Rs. {d['interval_width']:.2f})")
            ax.text(d["lower_bound"], y + 0.18, f"{d['lower_bound']:.1f}", ha="center", fontsize=9, color=col, fontweight="bold")
            ax.text(d["upper_bound"], y + 0.18, f"{d['upper_bound']:.1f}", ha="center", fontsize=9, color=col, fontweight="bold")
            
        ax.axvline(mean, color="black", linestyle="--", alpha=0.7, label=f"Mean: {mean:.1f}")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontweight="bold")
        ax.set_xlabel("Order Value (Rs.)")
        ax.set_title("Confidence Level Trade-off: 90% vs 95% vs 99%")
        ax.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(self.charts_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_sample_size_vs_width(self, sample_sizes: list[int], widths: list[float], filename: str = "ci_width_vs_sample_size.png"):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(sample_sizes, widths, marker="o", color="#e7298a", linewidth=2.5, markersize=7, label=r"Interval Width $\propto 1/\sqrt{n}$")
        
        for n, w in zip(sample_sizes, widths):
            ax.annotate(f"Rs. {w:.1f}", (n, w), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)
            
        ax.set_title("Sample Size vs. 95% Confidence Interval Width (Diminishing Returns)")
        ax.set_xlabel("Sample Size (n)")
        ax.set_ylabel("Confidence Interval Width (Rs.)")
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(self.charts_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_bootstrap_distribution(self, boot_dist: np.ndarray, obs_mean: float, lower: float, upper: float, filename: str = "bootstrap_distribution.png"):
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.histplot(boot_dist, kde=True, ax=ax, color="#7570b3", stat="density", bins=40, alpha=0.6)
        
        ax.axvline(obs_mean, color="blue", linestyle="-", linewidth=2, label=f"Observed Mean: Rs. {obs_mean:.2f}")
        ax.axvline(lower, color="red", linestyle="--", linewidth=1.8, label=f"2.5% Cutoff: Rs. {lower:.2f}")
        ax.axvline(upper, color="red", linestyle="--", linewidth=1.8, label=f"97.5% Cutoff: Rs. {upper:.2f}")
        
        ax.set_title(f"Bootstrap Resampling Distribution ({len(boot_dist):,} Iterations)")
        ax.set_xlabel("Bootstrap Sample Mean (Rs.)")
        ax.set_ylabel("Density")
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(self.charts_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_bootstrap_vs_t(self, t_res: dict, boot_res: dict, filename: str = "bootstrap_confidence_interval.png"):
        fig, ax = plt.subplots(figsize=(9, 4))
        y_pos = [0, 1]
        labels = ["Analytical T-Interval", "Bootstrap Percentile CI"]
        
        t_mean = t_res["sample_mean"]
        t_me = t_res["margin_of_error"]
        ax.errorbar(t_mean, 0, xerr=t_me, fmt="o", color="#2b5c8f", ecolor="#2b5c8f",
                    elinewidth=3, capsize=6, capthick=2, markersize=8, label=f"T-Interval [{t_res['lower_bound']:.1f}, {t_res['upper_bound']:.1f}]")
        
        boot_mean = boot_res["bootstrap_mean"]
        b_lower_err = boot_mean - boot_res["lower_bound"]
        b_upper_err = boot_res["upper_bound"] - boot_mean
        ax.errorbar(boot_mean, 1, xerr=[[b_lower_err], [b_upper_err]], fmt="s", color="#1b9e77", ecolor="#1b9e77",
                    elinewidth=3, capsize=6, capthick=2, markersize=8, label=f"Bootstrap CI [{boot_res['lower_bound']:.1f}, {boot_res['upper_bound']:.1f}]")
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontweight="bold")
        ax.set_xlabel("Order Value (Rs.)")
        ax.set_title("Parametric (Student's T) vs. Non-Parametric (Bootstrap) Comparison")
        ax.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(self.charts_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_parameter_estimation_dashboard(self, data: np.ndarray, t_res: dict, boot_res: dict, filename: str = "parameter_estimation.png"):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Panel 1: Sample Histogram with CI Shade
        sns.histplot(data, kde=True, ax=ax1, color="#2b5c8f", bins=30, alpha=0.5, stat="density")
        ax1.axvspan(t_res["lower_bound"], t_res["upper_bound"], color="orange", alpha=0.25, label="95% CI Region")
        ax1.axvline(t_res["sample_mean"], color="red", linestyle="--", linewidth=2, label=f"Sample Mean: Rs. {t_res['sample_mean']:.1f}")
        ax1.set_title("Customer Order Value Distribution & 95% CI Span")
        ax1.set_xlabel("Order Value (Rs.)")
        ax1.set_ylabel("Density")
        ax1.legend(loc="upper right")
        
        # Panel 2: Comparative Estimator Spans
        labels = ["T-Interval (Parametric)", "Bootstrap (Percentile)"]
        means = [t_res["sample_mean"], boot_res["bootstrap_mean"]]
        errs_low = [t_res["sample_mean"] - t_res["lower_bound"], boot_res["bootstrap_mean"] - boot_res["lower_bound"]]
        errs_high = [t_res["upper_bound"] - t_res["sample_mean"], boot_res["upper_bound"] - boot_res["bootstrap_mean"]]
        
        colors = ["#d95f02", "#7570b3"]
        y_pos = [0, 1]
        for y, m, el, eh, col, lbl in zip(y_pos, means, errs_low, errs_high, colors, labels):
            ax2.errorbar(m, y, xerr=[[el], [eh]], fmt="o", color=col, ecolor=col,
                         elinewidth=3, capsize=6, markersize=8, label=lbl)
            ax2.text(m, y + 0.15, f"Rs. {m:.2f}", ha="center", fontweight="bold", color=col)
            
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(labels, fontweight="bold")
        ax2.set_xlabel("Estimated AOV (Rs.)")
        ax2.set_title("Estimation Methodology Comparison")
        ax2.legend(loc="lower right")
        
        plt.suptitle("Executive Statistical Estimation Dashboard — E-Commerce AOV", y=1.02)
        plt.tight_layout()
        plt.savefig(self.charts_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()
