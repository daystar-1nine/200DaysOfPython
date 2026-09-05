"""
Challenge 5: Monthly Revenue & 3-Month Rolling Average Overlay
Combines actual monthly sales volatility with a smooth 3-month moving average.
Connects Day 59 rolling() aggregations with visual time-series trend identification.
"""

# What is used: Non-interactive backend selection
# Why it is used: Allows headless rendering across automated test and CLI environments
# How it works: Activates "Agg" backend before importing pyplot
import matplotlib
matplotlib.use("Agg")

# What is used: pyplot and pandas modules
# Why it is used: pandas computes rolling mean; pyplot renders the comparative time series
# How it works: rolling(window=3).mean() computes smoothed trend; ax.plot overlays both series
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def create_revenue_rolling_chart(output_dir: Path) -> Path:
    """Generates monthly revenue line chart with 3-month rolling average."""
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    revenue = [450000, 480000, 430000, 520000, 590000, 560000, 610000, 640000, 620000, 710000, 790000, 850000]

    df = pd.DataFrame({"Month": months, "Revenue": revenue})

    # What is used: df["Revenue"].rolling(window=3).mean()
    # Why it is used: Smooths out high-frequency month-to-month noise to uncover underlying momentum
    # How it works: Computes average over a sliding 3-month window
    df["Rolling_3M"] = df["Revenue"].rolling(window=3).mean()

    # What is used: plt.subplots()
    # Why it is used: Sets up OO figure canvas
    # How it works: Allocates 11x6 inch canvas
    fig, ax = plt.subplots(figsize=(11, 6))

    # Actual revenue series (semi-transparent line with circles)
    ax.plot(
        df["Month"],
        df["Revenue"],
        color="#74add1",
        linewidth=1.8,
        marker="o",
        markersize=6,
        alpha=0.8,
        label="Actual Monthly Revenue"
    )

    # 3-Month Rolling Average series (solid bold line)
    ax.plot(
        df["Month"],
        df["Rolling_3M"],
        color="#d73027",
        linewidth=2.8,
        label="3-Month Rolling Average (Smoothed Trend)"
    )

    # Shaded band highlighting variance
    ax.fill_between(
        df["Month"],
        df["Revenue"],
        df["Rolling_3M"],
        where=df["Rolling_3M"].notna(),
        color="#fee090",
        alpha=0.4,
        label="Monthly Volatility Spread"
    )

    # Formatting
    ax.set_title("Annual Corporate Revenue Trajectory & 3-Month Moving Average", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Month", fontsize=11, fontweight="bold")
    ax.set_ylabel("Revenue (₹)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", frameon=True, fontsize=9.5)

    # Key Business Insight Annotation
    ax.text(
        0.05, 0.50,
        "Strategic Business Insight:\nWhile April and September experienced short-term dips,\n"
        "the 3-month rolling average maintains an uninterrupted\n"
        "positive slope throughout the year, demonstrating healthy\n"
        "macro-level corporate expansion (+88.9% annualized).",
        transform=ax.transAxes,
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f7fb", edgecolor="#b0d4ec")
    )

    # Save chart
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / "challenge5_revenue_rolling_avg.png"

    fig.tight_layout()
    fig.savefig(target_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return target_path


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "output" / "charts"
    saved = create_revenue_rolling_chart(out_dir)
    print(f"[SUCCESS] Challenge 5 Chart saved to: {saved}")
