"""
Task 4: Scatter Plot — Study Hours vs Exam Marks
Demonstrates investigating bivariate relationships between two numerical variables.
Includes transparency to manage overlapping points, an OLS trendline, and correlation coefficient display.
"""

# What is used: Non-interactive backend selection
# Why it is used: Ensures execution across headless environments
# How it works: Activates "Agg" backend before importing pyplot
import matplotlib
matplotlib.use("Agg")

# What is used: pyplot and numpy modules
# Why it is used: For plotting and computing trendline polynomial fit
# How it works: pyplot manages axes; numpy computes linear regression coefficients
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def create_scatter_plot(output_dir: Path) -> Path:
    """Creates and saves a study hours vs marks scatter plot with trendline."""
    # Data definition (16 students)
    study_hours = [1.5, 2.0, 2.5, 3.0, 3.2, 3.8, 4.0, 4.5, 5.0, 5.2, 5.8, 6.0, 6.5, 7.0, 7.5, 8.0]
    exam_marks  = [42,  48,  51,  58,  55,  64,  68,  66,  74,  78,  82,  85,  88,  91,  94,  97]

    # Convert to numpy arrays
    x = np.array(study_hours)
    y = np.array(exam_marks)

    # What is used: Pearson correlation coefficient via np.corrcoef
    # Why it is used: Quantifies the strength and direction of linear association
    # How it works: Computes normalized covariance matrix; extracts r element
    corr_matrix = np.corrcoef(x, y)
    r = corr_matrix[0, 1]

    # What is used: plt.subplots()
    # Why it is used: Sets up OO figure canvas
    # How it works: Allocates 9x6 inch figure
    fig, ax = plt.subplots(figsize=(9, 6))

    # What is used: ax.scatter()
    # Why it is used: Plots individual observation points in a 2D Cartesian plane
    # How it works: Draws circles at (x, y) with specified size, color, alpha transparency, and edges
    ax.scatter(
        x,
        y,
        s=80,
        color="#2b7bba",
        edgecolors="#103d63",
        alpha=0.85,
        label="Student Scores"
    )

    # What is used: np.polyfit() and np.poly1d()
    # Why it is used: Computes an Ordinary Least Squares (OLS) linear trendline to visualize trajectory
    # How it works: Fits a degree-1 polynomial slope and intercept; evaluates across x range
    slope, intercept = np.polyfit(x, y, 1)
    trend_line = slope * x + intercept
    ax.plot(
        x,
        trend_line,
        color="#d95f02",
        linestyle="--",
        linewidth=2.0,
        label=f"Trendline (r = {r:.3f})"
    )

    # What is used: ax.set_title(), ax.set_xlabel(), ax.set_ylabel()
    # Why it is used: Fully labels chart with descriptive context
    # How it works: Sets titles and axis text labels
    ax.set_title("Study Hours vs Exam Marks: Bivariate Relationship", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Weekly Study Hours (hrs)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Exam Marks (%)", fontsize=11, fontweight="bold")

    # Grid and legend
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", frameon=True)

    # Annotation answering the question: "Does the chart visually suggest a positive relationship?"
    ax.text(
        0.05, 0.20,
        f"Strong Positive Correlation (r = {r:.3f})\nVisually confirms more study hours\nare strongly associated with higher marks.",
        transform=ax.transAxes,
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f7fb", edgecolor="#b0d4ec")
    )

    # Save chart
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / "task4_scatter_plot.png"

    fig.tight_layout()
    fig.savefig(target_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return target_path


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "output" / "charts"
    saved = create_scatter_plot(out_dir)
    print(f"[SUCCESS] Task 4 Scatter Plot saved to: {saved}")
