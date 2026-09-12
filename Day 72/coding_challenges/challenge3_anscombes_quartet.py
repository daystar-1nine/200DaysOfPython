"""
Day 72 — Challenge 3: Anscombe's Quartet Analysis & Plotting
Constructs the 4 classic datasets, verifies identical statistical profiles,
and renders a publication-grade 4-panel comparison plot.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

def get_anscombe_data():
    x1 = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
    y1 = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]
    
    x2 = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
    y2 = [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]
    
    x3 = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
    y3 = [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]
    
    x4 = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 19]
    y4 = [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 5.56, 7.91, 6.89, 12.50]
    
    return [
        (np.array(x1, dtype=float), np.array(y1, dtype=float), "Dataset I: Linear + Normal Noise"),
        (np.array(x2, dtype=float), np.array(y2, dtype=float), "Dataset II: Non-Linear Parabola"),
        (np.array(x3, dtype=float), np.array(y3, dtype=float), "Dataset III: Strict Linear with 1 Outlier"),
        (np.array(x4, dtype=float), np.array(y4, dtype=float), "Dataset IV: Vertical Cluster with Leverage Point")
    ]

def main():
    datasets = get_anscombe_data()
    out_dir = r"s:\Programming\Python200days\Day 72\output\charts"
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, "anscombes_quartet.png")
    
    print("=" * 65)
    print("DAY 72 — CHALLENGE 3: ANSCOMBE'S QUARTET")
    print("=" * 65)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    
    summary_rows = []
    
    for idx, (x, y, title) in enumerate(datasets):
        mean_x, var_x = np.mean(x), np.var(x, ddof=1)
        mean_y, var_y = np.mean(y), np.var(y, ddof=1)
        r, p_val = stats.pearsonr(x, y)
        rho, rho_p = stats.spearmanr(x, y)
        slope, intercept, _, _, _ = stats.linregress(x, y)
        
        summary_rows.append({
            "Dataset": f"Dataset {idx+1}",
            "Mean_X": round(mean_x, 2),
            "Var_X": round(var_x, 2),
            "Mean_Y": round(mean_y, 2),
            "Var_Y": round(var_y, 2),
            "Pearson_r": round(r, 3),
            "Spearman_rho": round(rho, 3),
            "Regression": f"Y = {intercept:.2f} + {slope:.2f}X"
        })
        
        # Plotting
        ax = axes[idx]
        ax.scatter(x, y, color="#1f77b4", s=80, edgecolors="black", linewidth=1.2, zorder=3)
        x_line = np.linspace(2, 20, 100)
        y_line = intercept + slope * x_line
        ax.plot(x_line, y_line, color="#d62728", linestyle="--", linewidth=2, label=f"Fit: Y = {intercept:.1f} + {slope:.2f}X")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("X Variable")
        ax.set_ylabel("Y Variable")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper left")
        ax.set_xlim(2, 20)
        ax.set_ylim(2, 14)
        
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    
    df_summary = pd.DataFrame(summary_rows)
    print(df_summary.to_string(index=False))
    print(f"\nRendered publication chart: {out_png}")
    
    # Assert statistical equivalence
    for r in summary_rows:
        assert np.isclose(r["Mean_X"], 9.0, atol=0.05)
        assert np.isclose(r["Mean_Y"], 7.5, atol=0.05)
        assert np.isclose(r["Pearson_r"], 0.816, atol=0.01)
    print("Verification: All 4 datasets have identical summary stats with divergent geometries.")

if __name__ == "__main__":
    main()
