"""
Day 63 - Coding Challenge 1: Examination Marks Distribution & Normality Diagnostics
=================================================================================
Analyzes continuous examination scores across subjects using histplot, KDE overlay,
empirical skewness, and statistical normality threshold evaluations.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def run_challenge1():
    # What is used: Synthetic reproducible examination dataset generation
    # Why it is used: Simulates realistic multimodal and skewed academic distributions
    # How it works: Uses NumPy random generator with fixed seed for determinism
    np.random.seed(42)
    n = 600
    math_scores = np.clip(np.random.normal(loc=68, scale=14, size=n), 15, 100)
    physics_scores = np.clip(np.concatenate([
        np.random.normal(loc=52, scale=10, size=int(n * 0.6)),
        np.random.normal(loc=82, scale=8, size=int(n * 0.4))
    ]), 10, 100)
    cs_scores = np.clip(np.random.beta(a=5, b=2, size=n) * 100, 20, 100)

    df_marks = pd.DataFrame({
        "Mathematics": math_scores,
        "Physics": physics_scores,
        "Computer_Science": cs_scores
    })

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "..", "output", "charts")
    os.makedirs(output_dir, exist_ok=True)

    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    subjects = [
        ("Mathematics", "#1f77b4", axes[0]),
        ("Physics", "#2ca02c", axes[1]),
        ("Computer_Science", "#ff7f0e", axes[2])
    ]

    for subj, col, ax in subjects:
        # What is used: sns.histplot with kde=True
        # Why it is used: Displays discrete bin counts and continuous density profile
        # How it works: Calculates optimal bins and Gaussian kernel density curve
        sns.histplot(df_marks[subj], kde=True, color=col, bins=25, stat="density", ax=ax)

        mean_val = df_marks[subj].mean()
        med_val = df_marks[subj].median()
        skew_val = df_marks[subj].skew()

        ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.8, label=f"Mean: {mean_val:.1f}")
        ax.axvline(med_val, color="black", linestyle=":", linewidth=1.8, label=f"Median: {med_val:.1f}")

        # Normality assessment heuristic
        status = "Near Normal" if abs(skew_val) < 0.5 else ("Left-Skewed" if skew_val < -0.5 else "Right-Skewed")
        ax.annotate(
            f"Skew: {skew_val:+.2f}\n({status})",
            xy=(0.05, 0.85),
            xycoords="axes fraction",
            fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffff", edgecolor="#cccccc")
        )

        ax.set_title(f"{subj} Marks Distribution", fontsize=13, fontweight="bold")
        ax.set_xlabel("Examination Score (0–100)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Density", fontsize=10, fontweight="bold")
        ax.set_xlim(0, 105)
        ax.legend(loc="upper right")
        sns.despine(ax=ax, top=True, right=True)

    fig.suptitle("Academic Cohort Examination Marks & Distribution Modeling", fontsize=15, fontweight="bold", y=1.03)
    fig.tight_layout()

    out_file = os.path.join(output_dir, "challenge1_marks_distribution.png")
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] Challenge 1 saved to: {out_file}")

if __name__ == "__main__":
    run_challenge1()
