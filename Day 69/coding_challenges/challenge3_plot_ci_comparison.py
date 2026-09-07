"""
Day 69 Challenge 3: Visual Confidence Interval Comparison
Generates a publication-grade horizontal error bar plot contrasting 90%, 95%, and 99% CIs.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def plot_ci_comparison(data: np.ndarray | list[float], output_path: str | Path) -> Path:
    arr = np.asarray(data, dtype=float)
    if len(arr) < 2:
        raise ValueError("Data must have at least 2 points.")
        
    n = len(arr)
    df = n - 1
    mean_val = float(np.mean(arr))
    s = float(np.std(arr, ddof=1))
    se = float(s / np.sqrt(n))
    
    levels = [0.90, 0.95, 0.99]
    labels = ["90% CI", "95% CI", "99% CI"]
    colors = ["#2b5c8f", "#d95f02", "#7570b3"]
    
    margins = []
    for conf in levels:
        alpha = 1.0 - conf
        t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=df))
        margins.append(t_crit * se)
        
    fig, ax = plt.subplots(figsize=(9, 4.5))
    y_pos = np.arange(len(levels))
    
    for y, me, color, lbl in zip(y_pos, margins, colors, labels):
        ax.errorbar(mean_val, y, xerr=me, fmt="o", color=color, ecolor=color,
                    elinewidth=3, capsize=6, capthick=2, markersize=8, label=f"{lbl} (+/- {me:.2f})")
        ax.text(mean_val - me, y + 0.15, f"{mean_val - me:.2f}", ha="center", fontsize=9, color=color, fontweight="bold")
        ax.text(mean_val + me, y + 0.15, f"{mean_val + me:.2f}", ha="center", fontsize=9, color=color, fontweight="bold")
        
    ax.axvline(mean_val, color="black", linestyle="--", linewidth=1.5, alpha=0.7, label=f"Sample Mean: {mean_val:.2f}")
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_xlabel("Estimate Value", fontsize=11, fontweight="bold")
    ax.set_title(f"Confidence Level Precision Comparison (n = {n})", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right")
    
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_p, dpi=300, bbox_inches="tight")
    plt.close()
    
    return out_p

def main():
    print("=" * 70)
    print("  CHALLENGE 3: VISUAL CONFIDENCE INTERVAL COMPARISON")
    print("=" * 70)
    data = [2450.0, 2380.0, 2520.0, 2410.0, 2600.0, 2350.0, 2490.0, 2470.0, 2550.0, 2390.0]
    out_file = Path(__file__).resolve().parent.parent / "output" / "charts" / "challenge3_ci_comparison.png"
    p = plot_ci_comparison(data, out_file)
    print(f"Figure saved successfully to: {p.name}")

if __name__ == "__main__":
    main()
