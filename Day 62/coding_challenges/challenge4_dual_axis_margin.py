"""
Challenge 4: Dual-Axis Performance Chart (twinx)
Overlays Gross Revenue (left y-axis, bar) and Net Profit Margin % (right y-axis, line)
sharing a unified chronological x-axis, using twinx() with distinct color coding.
"""

# What is used: Non-interactive Agg backend
# Why it is used: Headless execution
# How it works: Sets "Agg" before importing pyplot
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter
from pathlib import Path


def create_dual_axis_margin_chart(output_dir: Path) -> Path:
    """Generates a dual-axis chart synchronizing Revenue and Profit Margin."""
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    revenue = [180000, 210000, 190000, 240000, 280000, 270000, 310000, 340000, 320000, 380000, 450000, 520000]
    profit = [41400, 50400, 43700, 57600, 72800, 67500, 80600, 91800, 83200, 102600, 126000, 150800]

    df = pd.DataFrame({"Month": months, "Revenue": revenue, "Profit": profit})
    # Profit Margin %
    df["Margin_Pct"] = (df["Profit"] / df["Revenue"]) * 100

    fig, ax1 = plt.subplots(figsize=(11, 6))

    # Left Y-Axis: Revenue Bars
    color_rev = "#1f77b4"
    bars = ax1.bar(df["Month"], df["Revenue"], color=color_rev, alpha=0.7, width=0.5, label="Gross Revenue (₹)")
    ax1.set_xlabel("Calendar Month", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Gross Revenue (₹)", fontsize=11, fontweight="bold", color=color_rev)
    ax1.tick_params(axis="y", labelcolor=color_rev)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"₹{x*1e-3:.0f}K"))
    ax1.set_ylim(0, max(revenue) * 1.25)
    ax1.grid(axis="y", linestyle=":", alpha=0.4)

    # What is used: ax1.twinx()
    # Why it is used: Creates secondary independent y-axis sharing the x-axis
    # How it works: Generates new Axes ax2 overlaid directly on ax1 with right-aligned ticks
    ax2 = ax1.twinx()
    color_margin = "#d95f02"
    line_margin = ax2.plot(
        df["Month"],
        df["Margin_Pct"],
        color=color_margin,
        marker="s",
        linewidth=2.5,
        markersize=6,
        label="Net Profit Margin (%)"
    )
    ax2.set_ylabel("Net Profit Margin (%)", fontsize=11, fontweight="bold", color=color_margin)
    ax2.tick_params(axis="y", labelcolor=color_margin)
    ax2.set_ylim(15, 35)

    # Combined Legend across both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True, fontsize=10)

    ax1.set_title("Revenue Volume vs Net Margin Efficiency (Dual-Axis Analysis)", fontsize=14, fontweight="bold", pad=12)

    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / "challenge4_dual_axis_margin.png"

    fig.tight_layout()
    fig.savefig(target_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return target_path


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "output" / "charts"
    saved = create_dual_axis_margin_chart(out)
    print(f"[SUCCESS] Challenge 4 saved to: {saved}")