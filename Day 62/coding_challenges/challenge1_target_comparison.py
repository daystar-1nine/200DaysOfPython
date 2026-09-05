"""
Challenge 1: Actual Revenue vs Target Revenue with Target Reference Line
Visualizes monthly actual sales performance against an executive target milestone.
Highlights performance gaps with shaded regions and color-coded markers.
"""

# What is used: Non-interactive Agg backend
# Why it is used: Headless execution
# How it works: Sets "Agg" before importing pyplot
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def create_target_comparison_chart(output_dir: Path) -> Path:
    """Generates an Actual vs Target revenue comparison chart with an axhline reference line."""
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    actual = [48000, 52000, 49000, 58000, 64000, 61000, 69000, 74000, 71000, 82000, 91000, 98000]
    monthly_target = 65000

    df = pd.DataFrame({"Month": months, "Actual": actual})

    fig, ax = plt.subplots(figsize=(11, 6))

    # Line chart of actuals
    ax.plot(df["Month"], df["Actual"], marker="o", color="#1f77b4", linewidth=2.4, markersize=6, label="Actual Revenue (₹)")

    # What is used: ax.axhline()
    # Why it is used: Establishes the corporate target baseline across all 12 periods
    # How it works: Draws a dashed red benchmark line at y = 65,000
    ax.axhline(
        monthly_target,
        color="#d62728",
        linestyle="--",
        linewidth=2.0,
        label=f"Monthly Target Baseline (₹{monthly_target:,})"
    )

    # Shading: green when above target, light red when below target
    ax.fill_between(
        df["Month"],
        df["Actual"],
        monthly_target,
        where=(df["Actual"] >= monthly_target),
        interpolate=True,
        color="#2ca02c",
        alpha=0.25,
        label="Target Surpassed (+)"
    )
    ax.fill_between(
        df["Month"],
        df["Actual"],
        monthly_target,
        where=(df["Actual"] < monthly_target),
        interpolate=True,
        color="#d62728",
        alpha=0.20,
        label="Target Deficit (-)"
    )

    # Title & labels
    ax.set_title("Actual Monthly Revenue vs Annual Corporate Target (2026)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Calendar Month", fontsize=11, fontweight="bold")
    ax.set_ylabel("Revenue (₹)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", frameon=True, fontsize=9.5)

    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / "challenge1_target_comparison.png"

    fig.tight_layout()
    fig.savefig(target_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return target_path


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "output" / "charts"
    saved = create_target_comparison_chart(out)
    print(f"[SUCCESS] Challenge 1 saved to: {saved}")