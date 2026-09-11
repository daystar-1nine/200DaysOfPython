"""
Challenge 4: Visualizing Hypothesis Testing Curve
Plots the null distribution density curve, critical cutoff lines, rejection regions, and the observed statistic.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

def plot_hypothesis_test(
    t_stat: float,
    df: int,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    output_path: str = None
) -> str:
    x = np.linspace(-4.5, 4.5, 1000)
    y = stats.t.pdf(x, df=df)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, color="#1f77b4", lw=2.5, label=f"Null Distribution t(df={df})")
    
    if alternative == "two-sided":
        t_crit = stats.t.ppf(1 - alpha / 2, df=df)
        ax.axvline(t_crit, color="red", linestyle="--", lw=2, label=f"Critical Cutoff (+/- {t_crit:.2f})")
        ax.axvline(-t_crit, color="red", linestyle="--", lw=2)
        # Shade rejection regions
        ax.fill_between(x, 0, y, where=(x >= t_crit) | (x <= -t_crit), color="red", alpha=0.25, label="Rejection Region (alpha=0.05)")
    elif alternative == "greater":
        t_crit = stats.t.ppf(1 - alpha, df=df)
        ax.axvline(t_crit, color="red", linestyle="--", lw=2, label=f"Critical Cutoff (+{t_crit:.2f})")
        ax.fill_between(x, 0, y, where=(x >= t_crit), color="red", alpha=0.25, label="Rejection Region (alpha=0.05)")
    else:
        t_crit = stats.t.ppf(alpha, df=df)
        ax.axvline(t_crit, color="red", linestyle="--", lw=2, label=f"Critical Cutoff ({t_crit:.2f})")
        ax.fill_between(x, 0, y, where=(x <= t_crit), color="red", alpha=0.25, label="Rejection Region (alpha=0.05)")
        
    # Observed t-statistic
    ax.axvline(t_stat, color="#2ca02c", lw=2.5, linestyle="-", label=f"Observed t = {t_stat:.2f}")
    ax.scatter([t_stat], [stats.t.pdf(t_stat, df=df)], color="#2ca02c", s=80, zorder=5)
    
    ax.set_title(f"Hypothesis Test Null Distribution & Rejection Region ({alternative})", fontsize=13, fontweight="bold")
    ax.set_xlabel("t-statistic", fontsize=11)
    ax.set_ylabel("Probability Density", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left")
    
    if output_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_path = os.path.join(base_dir, "output", "charts", "challenge4_hypothesis_curve.png")
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path

if __name__ == "__main__":
    path = plot_hypothesis_test(t_stat=2.35, df=49, alpha=0.05, alternative="two-sided")
    print(f"=== Challenge 4: Saved visualization to {path} ===")
