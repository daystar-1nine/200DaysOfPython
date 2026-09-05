"""
Task 3: Dynamic Annotation — Automated Maximum Peak Detection
Demonstrates programmatic detection of data extremes (idxmax) and dynamic rendering
of an arrowed annotation without hard-coded coordinates.
"""

# What is used: Non-interactive Agg backend
# Why it is used: Ensures headless rendering across CLI and test processes
# How it works: Switches Matplotlib backend to "Agg"
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def create_auto_annotation_chart(output_dir: Path) -> Path:
    """Generates a line chart with automated peak detection and arrow callout."""
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    revenue = [125000, 138000, 132000, 147000, 162000, 158000, 171000, 184000, 179000, 198000, 245000, 285000]

    df = pd.DataFrame({"Month": months, "Revenue": revenue})

    # What is used: df["Revenue"].idxmax()
    # Why it is used: Locates the index of the highest metric dynamically
    # How it works: Evaluates Series maximum and returns row index
    max_idx = df["Revenue"].idxmax()
    max_month = df.loc[max_idx, "Month"]
    max_revenue = df.loc[max_idx, "Revenue"]

    # Also dynamically locate minimum for comprehensive contrast
    min_idx = df["Revenue"].idxmin()
    min_month = df.loc[min_idx, "Month"]
    min_revenue = df.loc[min_idx, "Revenue"]

    # Canvas setup
    fig, ax = plt.subplots(figsize=(11, 6))

    # Plot revenue trajectory
    ax.plot(df["Month"], df["Revenue"], marker="o", color="#1f77b4", linewidth=2.5, markersize=6, label="Monthly Revenue")

    # What is used: ax.annotate()
    # Why it is used: Creates a dynamic arrowed callout highlighting critical business peaks
    # How it works: xy specifies data point; xytext specifies offset; arrowprops creates the pointer
    ax.annotate(
        f"All-Time High (Peak)\n{max_month}: ₹{max_revenue:,}",
        xy=(max_month, max_revenue),
        xytext=(-90, -40),
        textcoords="offset points",
        arrowprops=dict(facecolor="#d62728", edgecolor="#990000", arrowstyle="->", lw=1.8),
        fontweight="bold",
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff0f0", edgecolor="#ff9999")
    )

    # Dynamic annotation for minimum
    ax.annotate(
        f"Annual Trough\n{min_month}: ₹{min_revenue:,}",
        xy=(min_month, min_revenue),
        xytext=(30, 25),
        textcoords="offset points",
        arrowprops=dict(facecolor="#31a354", edgecolor="#006d2c", arrowstyle="->", lw=1.5),
        fontweight="bold",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0fff0", edgecolor="#99e699")
    )

    # Formatting
    ax.set_title("Automated High/Low Extrema Detection in Revenue Trajectory", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Month", fontsize=11, fontweight="bold")
    ax.set_ylabel("Revenue (₹)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")

    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / "task3_auto_annotation.png"

    fig.tight_layout()
    fig.savefig(target_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return target_path


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "output" / "charts"
    saved = create_auto_annotation_chart(out)
    print(f"[SUCCESS] Task 3 saved to: {saved}")