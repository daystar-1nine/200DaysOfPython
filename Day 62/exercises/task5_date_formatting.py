"""
Task 5: Date Formatting and Locators with matplotlib.dates
Demonstrates professional handling of datetime objects on the x-axis.
Uses MonthLocator() and DateFormatter("%b '%y") to eliminate label overlap.
"""

# What is used: Non-interactive Agg backend
# Why it is used: Headless compatibility
# How it works: Activates "Agg" before importing pyplot
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from pathlib import Path


def create_date_formatted_chart(output_dir: Path) -> Path:
    """Generates a 12-month continuous time series using matplotlib.dates."""
    # Create pandas DatetimeIndex for 12 months
    date_range = pd.date_range(start="2026-01-01", periods=12, freq="MS")
    revenue = [110000, 125000, 120000, 138000, 150000, 145000, 160000, 175000, 168000, 190000, 230000, 270000]

    df = pd.DataFrame({"Date": date_range, "Revenue": revenue})

    fig, ax = plt.subplots(figsize=(11, 6))

    # Plot using datetime objects directly
    ax.plot(df["Date"], df["Revenue"], marker="o", color="#2b5c8f", linewidth=2.4, markersize=6, label="Monthly Revenue")

    # What is used: mdates.MonthLocator() and mdates.DateFormatter()
    # Why it is used: Ensures ticks correspond exactly to month boundaries and format cleanly
    # How it works: Locator places ticks at the 1st of each month; Formatter prints abbreviated month and 2-digit year
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

    # Rotate dates slightly for crisp readability
    ax.tick_params(axis="x", rotation=30, labelsize=9.5)

    # Formatting
    ax.set_title("Annual Corporate Revenue Trajectory (Datetime Axis Formatting)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Calendar Timeline", fontsize=11, fontweight="bold")
    ax.set_ylabel("Gross Revenue (₹)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")

    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / "task5_date_formatting.png"

    fig.tight_layout()
    fig.savefig(target_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return target_path


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "output" / "charts"
    saved = create_date_formatted_chart(out)
    print(f"[SUCCESS] Task 5 saved to: {saved}")