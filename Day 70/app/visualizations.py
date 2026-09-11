"""
Publication-grade diagnostic charts for hypothesis testing engine.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

try:
    from app.config import config
    from app.loader import DataLoader
    from app.hypotheses import TestResult
except (ImportError, ModuleNotFoundError):
    from config import config
    from loader import DataLoader
    from hypotheses import TestResult

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14
})

class Visualizer:
    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or config.CHARTS_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        self.loader = DataLoader()

    def plot_sample_distributions(self) -> str:
        """Chart 1: Sample Distributions with Null Reference Marks"""
        df_del = self.loader.load_delivery_times()
        df_mfg = self.loader.load_manufacturing()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        
        # Delivery times
        sns.histplot(df_del["delivery_time_minutes"], kde=True, ax=axes[0], color="#2b5c8f", bins=20)
        axes[0].axvline(30.0, color="red", linestyle="--", lw=2, label="Null SLA (30.0 min)")
        axes[0].axvline(df_del["delivery_time_minutes"].mean(), color="green", linestyle="-", lw=2, 
                       label=f"Sample Mean ({df_del['delivery_time_minutes'].mean():.2f} min)")
        axes[0].set_title("E-Commerce Delivery Times Distribution (n=250)", fontweight="bold")
        axes[0].set_xlabel("Delivery Time (minutes)")
        axes[0].set_ylabel("Frequency")
        axes[0].legend()

        # Manufacturing weights
        sns.histplot(df_mfg["weight_grams"], kde=True, ax=axes[1], color="#17becf", bins=20)
        axes[1].axvline(500.0, color="red", linestyle="--", lw=2, label="Target Weight (500.0g)")
        axes[1].axvline(df_mfg["weight_grams"].mean(), color="green", linestyle="-", lw=2, 
                       label=f"Sample Mean ({df_mfg['weight_grams'].mean():.2f}g)")
        axes[1].set_title("Manufacturing Item Weight Distribution (n=300)", fontweight="bold")
        axes[1].set_xlabel("Weight (grams)")
        axes[1].set_ylabel("Frequency")
        axes[1].legend()

        plt.tight_layout()
        path = os.path.join(self.output_dir, "sample_distribution.png")
        plt.savefig(path, dpi=300)
        plt.close(fig)
        return path

    def plot_test_distribution(self, df_val: int = 249) -> str:
        """Chart 2: Sampling Distribution of Test Statistic under H0"""
        x = np.linspace(-4.5, 4.5, 1000)
        y = stats.t.pdf(x, df=df_val)
        
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.plot(x, y, color="#1f77b4", lw=2.5, label=f"Null Distribution t(df={df_val})")
        
        # Standard normal comparison
        y_norm = stats.norm.pdf(x)
        ax.plot(x, y_norm, color="#7f7f7f", linestyle=":", lw=1.8, label="Standard Normal N(0, 1)")
        
        ax.set_title("Null Distribution of Test Statistic Under H0", fontweight="bold")
        ax.set_xlabel("Standardized Test Statistic (t / z)")
        ax.set_ylabel("Probability Density")
        ax.legend()
        
        plt.tight_layout()
        path = os.path.join(self.output_dir, "test_distribution.png")
        plt.savefig(path, dpi=300)
        plt.close(fig)
        return path

    def plot_rejection_regions(self) -> str:
        """Chart 3: Critical Cutoffs and Rejection Regions Comparison"""
        x = np.linspace(-4.0, 4.0, 1000)
        y = stats.norm.pdf(x)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Two-tailed test
        axes[0].plot(x, y, color="#333333", lw=2)
        crit_two = stats.norm.ppf(0.975)
        axes[0].fill_between(x, 0, y, where=(x >= crit_two) | (x <= -crit_two), color="red", alpha=0.3, label=f"Rejection Region (alpha=0.05)")
        axes[0].fill_between(x, 0, y, where=(x > -crit_two) & (x < crit_two), color="green", alpha=0.15, label="Fail-to-Reject Region (95%)")
        axes[0].axvline(crit_two, color="red", linestyle="--", lw=1.8)
        axes[0].axvline(-crit_two, color="red", linestyle="--", lw=1.8)
        axes[0].set_title("Two-Tailed Test: Critical Cutoffs (+/- 1.96)", fontweight="bold")
        axes[0].set_xlabel("z-statistic")
        axes[0].legend(loc="upper right")

        # Right-tailed test
        axes[1].plot(x, y, color="#333333", lw=2)
        crit_one = stats.norm.ppf(0.95)
        axes[1].fill_between(x, 0, y, where=(x >= crit_one), color="red", alpha=0.3, label=f"Rejection Region (alpha=0.05)")
        axes[1].fill_between(x, 0, y, where=(x < crit_one), color="green", alpha=0.15, label="Fail-to-Reject Region (95%)")
        axes[1].axvline(crit_one, color="red", linestyle="--", lw=1.8)
        axes[1].set_title("Right-Tailed Test: Critical Cutoff (+1.645)", fontweight="bold")
        axes[1].set_xlabel("z-statistic")
        axes[1].legend(loc="upper right")

        plt.tight_layout()
        path = os.path.join(self.output_dir, "rejection_region.png")
        plt.savefig(path, dpi=300)
        plt.close(fig)
        return path

    def plot_p_value_concept(self) -> str:
        """Chart 4: P-Value Visual Representation"""
        x = np.linspace(-4.0, 4.0, 1000)
        y = stats.norm.pdf(x)
        z_obs = 2.32
        
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.plot(x, y, color="#2b5c8f", lw=2.5, label="H0 Sampling Distribution")
        ax.fill_between(x, 0, y, where=(x >= z_obs), color="#d62728", alpha=0.5, label=f"P-value Area (P = {1 - stats.norm.cdf(z_obs):.4f})")
        ax.axvline(z_obs, color="#d62728", lw=2.5, linestyle="-", label=f"Observed z = {z_obs:.2f}")
        ax.axvline(1.645, color="orange", lw=2, linestyle="--", label="Alpha = 0.05 Cutoff (z=1.645)")
        
        ax.set_title("P-Value Visualization: Tail Area Beyond Observed Statistic", fontweight="bold")
        ax.set_xlabel("Standardized Statistic")
        ax.set_ylabel("Probability Density")
        ax.legend()
        
        plt.tight_layout()
        path = os.path.join(self.output_dir, "p_value.png")
        plt.savefig(path, dpi=300)
        plt.close(fig)
        return path

    def plot_confidence_intervals(self, results: list[TestResult]) -> str:
        """Chart 5: Confidence Interval Duality with Null Hypotheses"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        mean_results = [r for r in results if r.spec.test_kind in (r.spec.test_kind.ONE_SAMPLE_T, r.spec.test_kind.ONE_SAMPLE_Z)]
        y_positions = np.arange(len(mean_results))
        
        for idx, res in enumerate(mean_results):
            mean = res.point_estimate
            low = res.ci_lower
            high = res.ci_upper
            null_val = res.spec.null_value
            
            # Draw CI bar
            color = "#d62728" if res.reject_null else "#2ca02c"
            ax.errorbar(mean, idx, xerr=[[mean - low], [high - mean]], fmt="o", color=color, 
                        ecolor=color, elinewidth=3, capsize=6, markersize=8)
            ax.scatter([null_val], [idx], marker="D", color="black", s=80, zorder=5)
            
        ax.set_yticks(y_positions)
        ax.set_yticklabels([r.spec.scenario_name for r in mean_results], fontsize=10)
        ax.set_xlabel("Parameter Value & 95% Confidence Interval")
        ax.set_title("Confidence Intervals vs Hypothesized Values (Duality)", fontweight="bold")
        
        # Legend proxies
        ax.scatter([], [], color="#d62728", marker="o", label="Rejected H0 (Significant)")
        ax.scatter([], [], color="#2ca02c", marker="o", label="Fail to Reject H0")
        ax.scatter([], [], color="black", marker="D", label="Hypothesized Null Value (H0)")
        ax.legend(loc="best")
        
        plt.tight_layout()
        path = os.path.join(self.output_dir, "confidence_interval.png")
        plt.savefig(path, dpi=300)
        plt.close(fig)
        return path

    def plot_effect_sizes(self, results: list[TestResult]) -> str:
        """Chart 6: Practical Significance & Cohen's d Benchmarks"""
        mean_results = [r for r in results if r.cohens_d is not None]
        names = [r.spec.scenario_name.split()[0] for r in mean_results]
        d_vals = [r.cohens_d for r in mean_results]
        
        fig, ax = plt.subplots(figsize=(10, 5.5))
        
        # Benchmark bands
        ax.axhspan(0.0, 0.2, color="#e5f5e0", alpha=0.6, label="Negligible (|d| < 0.2)")
        ax.axhspan(0.2, 0.5, color="#a1d99b", alpha=0.6, label="Small (0.2 <= |d| < 0.5)")
        ax.axhspan(0.5, 0.8, color="#41ab5d", alpha=0.6, label="Medium (0.5 <= |d| < 0.8)")
        ax.axhspan(0.8, 1.5, color="#005a32", alpha=0.4, label="Large (|d| >= 0.8)")
        
        bars = ax.bar(names, [abs(d) for d in d_vals], width=0.45, color="#1f77b4", edgecolor="black", zorder=3)
        for bar, d_raw in zip(bars, d_vals):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.02, f"d={d_raw:+.2f}", 
                    ha="center", va="bottom", fontweight="bold", fontsize=10)
            
        ax.set_ylim(0, 1.2)
        ax.set_ylabel("Absolute Cohen's d")
        ax.set_title("Standardized Effect Size (Cohen's d) Across Scenarios", fontweight="bold")
        ax.legend(loc="upper right")
        
        plt.tight_layout()
        path = os.path.join(self.output_dir, "effect_size.png")
        plt.savefig(path, dpi=300)
        plt.close(fig)
        return path

    def plot_executive_dashboard(self, results: list[TestResult]) -> str:
        """Chart 7: 4-Panel Executive Decision Dashboard"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Panel 1: Test Statistics vs Critical Values
        scenarios = [r.spec.scenario_name.split()[0] for r in results]
        t_stats = [r.test_statistic for r in results]
        crit_vals = [r.critical_value for r in results]
        
        x = np.arange(len(scenarios))
        width = 0.35
        axes[0, 0].bar(x - width/2, t_stats, width, label="Test Statistic", color="#1f77b4")
        axes[0, 0].bar(x + width/2, crit_vals, width, label="Critical Value", color="#ff7f0e")
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(scenarios)
        axes[0, 0].set_title("Observed Test Statistic vs Critical Threshold", fontweight="bold")
        axes[0, 0].legend()

        # Panel 2: P-Values vs Alpha (0.05) on Log Scale
        p_vals = [max(r.p_value, 1e-10) for r in results]
        colors = ["#d62728" if p <= 0.05 else "#2ca02c" for p in p_vals]
        axes[0, 1].bar(scenarios, -np.log10(p_vals), color=colors)
        axes[0, 1].axhline(-np.log10(0.05), color="red", linestyle="--", lw=2, label="alpha = 0.05 threshold (-log10(0.05)=1.30)")
        axes[0, 1].set_ylabel("-log10(P-value) [Higher = More Significant]")
        axes[0, 1].set_title("Statistical Significance Profile", fontweight="bold")
        axes[0, 1].legend()

        # Panel 3: Confidence Intervals Duality
        y_pos = np.arange(len(results))
        for idx, r in enumerate(results):
            col = "#d62728" if r.reject_null else "#2ca02c"
            axes[1, 0].errorbar(r.point_estimate, idx, xerr=[[r.point_estimate - r.ci_lower], [r.ci_upper - r.point_estimate]], 
                                fmt="o", color=col, elinewidth=2.5, capsize=5)
            axes[1, 0].scatter([r.spec.null_value], [idx], marker="X", color="black", s=70)
        axes[1, 0].set_yticks(y_pos)
        axes[1, 0].set_yticklabels(scenarios)
        axes[1, 0].set_title("Confidence Intervals & Hypothesized Nulls", fontweight="bold")

        # Panel 4: Decision Summary Table
        axes[1, 1].axis("off")
        table_data = []
        for r in results:
            table_data.append([
                r.spec.scenario_name.split()[0],
                f"{r.point_estimate:.3f}",
                f"{r.p_value:.4e}",
                "REJECT" if r.reject_null else "FAIL-REJ",
                r.effect_magnitude
            ])
        tbl = axes[1, 1].table(
            cellText=table_data,
            colLabels=["Scenario", "Estimate", "P-Value", "Decision", "Effect"],
            loc="center",
            cellLoc="center"
        )
        tbl.scale(1.0, 1.7)
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        axes[1, 1].set_title("Summary Decision Matrix", fontweight="bold")

        plt.suptitle("Statistical Hypothesis Testing Engine — Executive Dashboard", fontsize=15, fontweight="bold")
        plt.tight_layout()
        path = os.path.join(self.output_dir, "test_dashboard.png")
        plt.savefig(path, dpi=300)
        plt.close(fig)
        return path

    def generate_all_charts(self, results: list[TestResult]) -> list[str]:
        p1 = self.plot_sample_distributions()
        p2 = self.plot_test_distribution()
        p3 = self.plot_rejection_regions()
        p4 = self.plot_p_value_concept()
        p5 = self.plot_confidence_intervals(results)
        p6 = self.plot_effect_sizes(results)
        p7 = self.plot_executive_dashboard(results)
        return [p1, p2, p3, p4, p5, p6, p7]
