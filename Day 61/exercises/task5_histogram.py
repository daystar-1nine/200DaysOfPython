"""
Task 5: Histogram — Distribution of Exam Marks Across Multiple Bin Sizes
Demonstrates creating histograms to analyze the continuous frequency distribution of 100 exam marks.
Compares bins=5, bins=10, and bins=20 side-by-side to illustrate binning trade-offs.
"""

# What is used: Non-interactive backend selection
# Why it is used: Allows headless rendering in server/testing environments
# How it works: Switches to "Agg" backend before importing pyplot
import matplotlib
matplotlib.use("Agg")

# What is used: pyplot and numpy modules
# Why it is used: pyplot constructs multi-panel figures; numpy generates reproducible random marks
# How it works: numpy.random.normal generates marks with mean 72 and std 12; pyplot creates subplots
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def create_histogram_comparison(output_dir: Path) -> Path:
    """Creates a 3-panel comparative histogram for bins=5, 10, and 20."""
    # Seed for deterministic reproducibility
    np.random.seed(42)
    # Generate 100 exam marks clamped between 40 and 100
    raw_marks = np.random.normal(loc=72, scale=12, size=100)
    marks = np.clip(raw_marks, 40, 100)

    # What is used: plt.subplots(1, 3, figsize=(16, 5))
    # Why it is used: Creates a 1-row by 3-column grid of Axes sharing the same y-axis scale
    # How it works: Returns a figure and an array of 3 Axes objects (axes[0], axes[1], axes[2])
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    bin_configs = [
        (5, "5 Bins (Under-Smoothed / Coarse)", "#3b6e8c"),
        (10, "10 Bins (Balanced Resolution)", "#2b8c5a"),
        (20, "20 Bins (Fine / Detailed Grain)", "#8c4a2b")
    ]

    mean_val = np.mean(marks)
    median_val = np.median(marks)

    for ax, (b, title, color) in zip(axes, bin_configs):
        # What is used: ax.hist()
        # Why it is used: Partitions continuous numerical values into adjacent intervals and counts occurrences
        # How it works: Computes bin edges, counts frequencies, and draws contiguous rectangular bars
        counts, edges, patches = ax.hist(
            marks,
            bins=b,
            color=color,
            edgecolor="#1a1a1a",
            alpha=0.75,
            linewidth=1.1
        )

        # What is used: ax.axvline()
        # Why it is used: Draws vertical reference lines indicating parametric and non-parametric central tendencies
        # How it works: Draws vertical line at specified x coordinate spanning 0% to 100% of y-axis
        ax.axvline(mean_val, color="#d95f02", linestyle="--", linewidth=1.8, label=f"Mean: {mean_val:.1f}")
        ax.axvline(median_val, color="#7570b3", linestyle=":", linewidth=1.8, label=f"Median: {median_val:.1f}")

        # Formatting each panel
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Exam Marks (%)", fontsize=10, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(loc="upper left", fontsize=8.5, frameon=True)

    # Global y-axis label on leftmost panel
    axes[0].set_ylabel("Student Frequency (Count)", fontsize=10, fontweight="bold")

    # Global super title
    fig.suptitle("Impact of Bin Sizing on Frequency Distribution Perception (N = 100)", fontsize=14, fontweight="bold", y=1.02)

    # Save chart
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / "task5_histogram.png"

    fig.tight_layout()
    fig.savefig(target_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return target_path


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "output" / "charts"
    saved = create_histogram_comparison(out_dir)
    print(f"[SUCCESS] Task 5 Histogram saved to: {saved}")
