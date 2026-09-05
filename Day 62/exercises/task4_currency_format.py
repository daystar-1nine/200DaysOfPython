"""
Task 4: Precision Currency Formatting with FuncFormatter
Demonstrates custom y-axis tick formatting to render formatted Indian Rupee (₹) amounts.
Replaces raw integer strings with comma-separated, currency-prefixed labels.
"""

# What is used: Non-interactive Agg backend
# Why it is used: Headless compatibility
# How it works: Activates "Agg" before importing pyplot
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path


def create_currency_formatted_chart(output_dir: Path) -> Path:
    """Generates a revenue chart where the Y-axis explicitly displays formatted Indian Rupees (₹)."""
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug"]
    revenue = [15000, 22000, 28000, 35000, 42000, 39000, 48000, 56000]

    # What is used: FuncFormatter custom function
    # Why it is used: Transforms numerical tick values into human-readable currency strings
    # How it works: Accepts value and tick position; returns formatted string with ₹ and commas
    def format_currency_inr(val, pos):
        return f"₹{val:,.0f}"

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(months, revenue, marker="s", color="#006699", linewidth=2.4, markersize=7, label="Gross Revenue")

    # What is used: ax.yaxis.set_major_formatter()
    # Why it is used: Binds the custom formatting function to all major y-axis ticks
    # How it works: Overrides default scalar formatter with FuncFormatter(format_currency_inr)
    ax.yaxis.set_major_formatter(FuncFormatter(format_currency_inr))

    # Formatting
    ax.set_title("Enterprise Revenue Progression with Explicit Currency Ticks", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Operational Period", fontsize=11, fontweight="bold")
    ax.set_ylabel("Revenue Amount (₹)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left")

    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / "task4_currency_format.png"

    fig.tight_layout()
    fig.savefig(target_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return target_path


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "output" / "charts"
    saved = create_currency_formatted_chart(out)
    print(f"[SUCCESS] Task 4 saved to: {saved}")