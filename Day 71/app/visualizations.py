"""
Publication-grade visualizations for Day 71 A/B Testing Engine.
Generates 8 diagnostic charts saved into output/charts/.
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
except (ImportError, ModuleNotFoundError):
    from config import config

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

    def plot_conversion_comparison(self, rate_c: float, rate_t: float) -> str:
        """Chart 1: Control vs Treatment Conversion Bar Chart"""
        fig, ax = plt.subplots(figsize=(8, 5))
        groups = ["Control (A)", "Treatment (B)"]
        rates = [rate_c * 100, rate_t * 100]
        colors = ["#4a7bb0", "#2ca02c" if rate_t >= rate_c else "#d62728"]
        
        bars = ax.bar(groups, rates, color=colors, width=0.45, edgecolor="black", zorder=3)
        for b, r in zip(bars, rates):
            ax.text(b.get_x() + b.get_width() / 2., r + 0.1, f"{r:.2f}%", ha="center", va="bottom", fontweight="bold")
            
        ax.set_ylim(0, max(rates) * 1.35)
        ax.set_ylabel("Conversion Rate (%)")
        ax.set_title("A/B Conversion Rate Comparison", fontweight="bold")
        
        path = os.path.join(self.output_dir, "conversion_comparison.png")
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close(fig)
        return path

    def plot_conversion_ci(self, diff: float, ci_low: float, ci_high: float) -> str:
        """Chart 2: Conversion Rate Difference with Confidence Interval"""
        fig, ax = plt.subplots(figsize=(9, 4.5))
        color = "#2ca02c" if ci_low > 0 else ("#d62728" if ci_high < 0 else "#7f7f7f")
        
        ax.errorbar(diff * 100, 0, xerr=[[(diff - ci_low) * 100], [(ci_high - diff) * 100]],
                    fmt="o", color=color, ecolor=color, elinewidth=3, capsize=8, markersize=9, zorder=4)
        ax.axvline(0, color="red", linestyle="--", lw=1.8, label="Zero Effect Line (H0)")
        
        ax.set_yticks([])
        ax.set_xlabel("Difference in Conversion Rate: p_B - p_A (percentage points)")
        ax.set_title(f"95% Confidence Interval for Absolute Lift: [{ci_low*100:+.2f}%, {ci_high*100:+.2f}%]", fontweight="bold")
        ax.legend()
        
        path = os.path.join(self.output_dir, "conversion_ci.png")
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close(fig)
        return path

    def plot_lift(self, abs_lift: float, rel_lift: float) -> str:
        """Chart 3: Absolute vs Relative Lift Visualization"""
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        
        # Absolute Lift
        axes[0].bar(["Absolute Lift"], [abs_lift * 100], color="#1f77b4", width=0.4, edgecolor="black")
        axes[0].text(0, (abs_lift * 100) + 0.02, f"{abs_lift*100:+.2f} pp", ha="center", va="bottom", fontweight="bold")
        axes[0].set_ylabel("Percentage Points")
        axes[0].set_title("Absolute Lift (p_B - p_A)", fontweight="bold")
        
        # Relative Lift
        axes[1].bar(["Relative Lift"], [rel_lift], color="#2ca02c" if rel_lift >= 0 else "#d62728", width=0.4, edgecolor="black")
        axes[1].text(0, rel_lift + (0.5 if rel_lift >= 0 else -1.5), f"{rel_lift:+.2f}%", ha="center", va="bottom", fontweight="bold")
        axes[1].set_ylabel("Percentage (%)")
        axes[1].set_title("Relative Lift over Baseline", fontweight="bold")
        
        path = os.path.join(self.output_dir, "lift.png")
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close(fig)
        return path

    def plot_revenue_comparison(self, rev_c: np.ndarray, rev_t: np.ndarray) -> str:
        """Chart 4: Revenue and ARPU Comparison"""
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        
        arpu_c = float(np.mean(rev_c))
        arpu_t = float(np.mean(rev_t))
        
        # ARPU Bar Chart
        axes[0].bar(["Control ARPU", "Treatment ARPU"], [arpu_c, arpu_t], color=["#4a7bb0", "#17becf"], width=0.45, edgecolor="black")
        axes[0].text(0, arpu_c + 0.05, f"${arpu_c:.2f}", ha="center", fontweight="bold")
        axes[0].text(1, arpu_t + 0.05, f"${arpu_t:.2f}", ha="center", fontweight="bold")
        axes[0].set_ylabel("USD ($)")
        axes[0].set_title("Average Revenue Per User (ARPU)", fontweight="bold")
        
        # Converted Revenue Distribution Boxplot
        c_conv = rev_c[rev_c > 0]
        t_conv = rev_t[rev_t > 0]
        data = [c_conv, t_conv]
        axes[1].boxplot(data, tick_labels=["Control (Converted)", "Treatment (Converted)"], patch_artist=True,
                        boxprops=dict(facecolor="#aec7e8", color="black"),
                        medianprops=dict(color="red", lw=2))
        axes[1].set_ylabel("Order Value ($)")
        axes[1].set_title("Revenue Distribution (Converted Users)", fontweight="bold")
        
        path = os.path.join(self.output_dir, "revenue_comparison.png")
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close(fig)
        return path

    def plot_user_distribution(self, n_c: int, n_t: int) -> str:
        """Chart 5: User Allocation Distribution (SRM check)"""
        fig, ax = plt.subplots(figsize=(7, 4.5))
        labels = [f"Control ({n_c:,})", f"Treatment ({n_t:,})"]
        sizes = [n_c, n_t]
        colors = ["#4a7bb0", "#e377c2"]
        
        ax.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90, 
               wedgeprops=dict(edgecolor="black", lw=1.2))
        ax.set_title("Traffic Split Verification (Intended 50/50)", fontweight="bold")
        
        path = os.path.join(self.output_dir, "user_distribution.png")
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close(fig)
        return path

    def plot_conversion_distribution(self, df_data) -> str:
        """Chart 6: Conversion Outcome Counts by Group"""
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(data=df_data, x="group", hue="converted", palette=["#bdbdbd", "#2ca02c"], ax=ax)
        ax.set_title("Conversion Outcomes by Variant", fontweight="bold")
        ax.set_xlabel("Experiment Group")
        ax.set_ylabel("Number of Users")
        ax.legend(["Non-Converted", "Converted"], title="Outcome")
        
        path = os.path.join(self.output_dir, "conversion_distribution.png")
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close(fig)
        return path

    def plot_timeline(self, df_data) -> str:
        """Chart 7: Cumulative Conversion Rate Evolution Over User Sequence"""
        fig, ax = plt.subplots(figsize=(10, 5))
        
        df_c = df_data[df_data["group"] == "Control"].copy().reset_index(drop=True)
        df_t = df_data[df_data["group"] == "Treatment"].copy().reset_index(drop=True)
        
        cum_rate_c = (df_c["converted"].cumsum() / (df_c.index + 1)) * 100
        cum_rate_t = (df_t["converted"].cumsum() / (df_t.index + 1)) * 100
        
        ax.plot(cum_rate_c, label="Control Cumulative Conversion", color="#4a7bb0", lw=2)
        ax.plot(cum_rate_t, label="Treatment Cumulative Conversion", color="#2ca02c", lw=2)
        ax.set_xlabel("User Arrivals (Sequential Index)")
        ax.set_ylabel("Conversion Rate (%)")
        ax.set_title("Cumulative Conversion Trajectory Over Time", fontweight="bold")
        ax.legend()
        
        path = os.path.join(self.output_dir, "timeline.png")
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close(fig)
        return path

    def plot_dashboard(self, conv_metrics: dict, stats_res: dict, decision_res, impact_res: dict) -> str:
        """Chart 8: Executive Summary Dashboard"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Panel 1: Conversion Rates
        rc = conv_metrics["rate_control"] * 100
        rt = conv_metrics["rate_treatment"] * 100
        axes[0, 0].bar(["Control", "Treatment"], [rc, rt], color=["#4a7bb0", "#2ca02c"], width=0.45, edgecolor="black")
        axes[0, 0].text(0, rc + 0.1, f"{rc:.2f}%", ha="center", fontweight="bold")
        axes[0, 0].text(1, rt + 0.1, f"{rt:.2f}%", ha="center", fontweight="bold")
        axes[0, 0].set_ylabel("Conversion Rate (%)")
        axes[0, 0].set_title(f"Primary Metric: Relative Lift = {conv_metrics['relative_lift_pct']:+.2f}%", fontweight="bold")
        
        # Panel 2: P-Value Profile
        p_val = stats_res["p_value"]
        axes[0, 1].bar(["p-value"], [-np.log10(max(p_val, 1e-10))], color="#2ca02c" if p_val < 0.05 else "#d62728", width=0.35)
        axes[0, 1].axhline(-np.log10(0.05), color="red", linestyle="--", lw=2, label="alpha = 0.05 (-log10=1.30)")
        axes[0, 1].set_ylabel("-log10(p-value)")
        axes[0, 1].set_title(f"Statistical Significance (p = {p_val:.4e})", fontweight="bold")
        axes[0, 1].legend()

        # Panel 3: Financial Impact
        axes[1, 0].bar(["Annual Net Revenue Impact"], [impact_res["projected_net_annual_profit"]], color="#17becf", width=0.4, edgecolor="black")
        axes[1, 0].text(0, impact_res["projected_net_annual_profit"] + 500, f"${impact_res['projected_net_annual_profit']:,.0f}", ha="center", fontweight="bold")
        axes[1, 0].set_ylabel("USD ($)")
        axes[1, 0].set_title(f"Projected Annual Net Gain ({impact_res['annual_incremental_conversions']:,} Conversions)", fontweight="bold")

        # Panel 4: Executive Decision Matrix Table
        axes[1, 1].axis("off")
        table_rows = [
            ["Metric", "Control", "Treatment", "Status"],
            ["Conversion Rate", f"{rc:.2f}%", f"{rt:.2f}%", f"Lift {conv_metrics['relative_lift_pct']:+.2f}%"],
            ["P-Value", "—", f"{p_val:.4e}", "SIGNIFICANT" if p_val < 0.05 else "NOT SIG"],
            ["Guardrails", "Baseline", "Healthy", "PASS" if decision_res.guardrails_passed else "FAIL"],
            ["RECOMMENDATION", "—", "—", decision_res.launch_recommendation.split()[0]]
        ]
        tbl = axes[1, 1].table(cellText=table_rows, loc="center", cellLoc="center")
        tbl.scale(1.0, 1.8)
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        axes[1, 1].set_title("Executive Decision Card", fontweight="bold")
        
        plt.suptitle("A/B Testing Analysis Engine — Executive Summary Dashboard", fontsize=15, fontweight="bold")
        path = os.path.join(self.output_dir, "dashboard.png")
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close(fig)
        return path

    def generate_all(self, df_data, conv_metrics, stats_res, decision_res, impact_res) -> list[str]:
        ctrl = df_data[df_data["group"] == "Control"]
        trt = df_data[df_data["group"] == "Treatment"]
        
        p1 = self.plot_conversion_comparison(conv_metrics["rate_control"], conv_metrics["rate_treatment"])
        p2 = self.plot_conversion_ci(stats_res["absolute_lift"], stats_res.get("ci_lower", 0.0), stats_res.get("ci_upper", 0.0))
        p3 = self.plot_lift(conv_metrics["absolute_lift"], conv_metrics["relative_lift_pct"])
        p4 = self.plot_revenue_comparison(ctrl["revenue"].values, trt["revenue"].values)
        p5 = self.plot_user_distribution(len(ctrl), len(trt))
        p6 = self.plot_conversion_distribution(df_data)
        p7 = self.plot_timeline(df_data)
        p8 = self.plot_dashboard(conv_metrics, stats_res, decision_res, impact_res)
        return [p1, p2, p3, p4, p5, p6, p7, p8]
