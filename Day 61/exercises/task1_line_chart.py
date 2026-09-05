"""
Task 1: Line Chart — Monthly Revenue Visualization
Demonstrates the creation of a clean, labeled time-series line chart using Matplotlib.
Includes markers, explicit axis labels, a title, gridlines, and figure export.
"""

# What is used: Non-interactive backend selection
# Why it is used: Ensures rendering works across headless servers and automated test environments
# How it works: Switches Matplotlib backend to "Agg" before importing pyplot
import matplotlib
matplotlib.use("Agg")

# What is used: pyplot module from Matplotlib
# Why it is used: Factory interface to create Figure and Axes instances
# How it works: Exposes subplots() to create object-oriented chart canvases
import matplotlib.pyplot as plt
from pathlib import Path


def create_line_chart(output_dir: Path) -> Path:
    """Creates and saves a monthly revenue line chart."""
    # Data definition
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
    revenue = [25000, 28000, 32000, 30000, 38000, 42000]

    # What is used: plt.subplots()
    # Why it is used: Instantiates Figure and Axes objects in an Object-Oriented paradigm
    # How it works: Allocates a canvas of 9x5 inches and returns canvas (fig) and subplot (ax)
    fig, ax = plt.subplots(figsize=(9, 5))

    # What is used: ax.plot()
    # Why it is used: Renders a continuous polyline connecting discrete chronological data points
    # How it works: Draws line with specified color, line width, circular markers, and label
    ax.plot(
        months,
        revenue,
        color="#1f77b4",
        linewidth=2.5,
        marker="o",
        markersize=7,
        markerfacecolor="#ff7f0e",
        label="Monthly Revenue"
    )

    # What is used: ax.set_title(), ax.set_xlabel(), ax.set_ylabel()
    # Why it is used: Contextualizes the chart so viewers understand metrics and units without guessing
    # How it works: Attaches formatted text strings to the respective axis and figure title positions
    ax.set_title("H1 Monthly Revenue Trajectory (2026)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Month", fontsize=11, fontweight="bold")
    ax.set_ylabel("Revenue (₹)", fontsize=11, fontweight="bold")

    # What is used: ax.grid()
    # Why it is used: Provides subtle horizontal and vertical reference guides for value estimation
    # How it works: Overlays a dashed grid on the plotting area with 50% opacity
    ax.grid(True, linestyle="--", alpha=0.5)

    # What is used: ax.legend()
    # Why it is used: Displays a key identifying the plotted series
    # How it works: Reads labels assigned in ax.plot calls and creates a formatted legend box
    ax.legend(loc="upper left", frameon=True)

    # Annotate peak value
    max_rev = max(revenue)
    max_idx = revenue.index(max_rev)
    ax.annotate(
        f"Peak: ₹{max_rev:,}",
        xy=(months[max_idx], max_rev),
        xytext=(-30, 15),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="#333333"),
        fontweight="bold"
    )

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / "task1_line_chart.png"

    # What is used: fig.tight_layout() and fig.savefig()
    # Why it is used: Prevents clipping of margins/labels and serializes chart to high-res PNG
    # How it works: Adjusts subplot parameters then writes bitmap data at 300 DPI
    fig.tight_layout()
    fig.savefig(target_path, dpi=300, bbox_inches="tight")

    # What is used: plt.close(fig)
    # Why it is used: Prevents memory leaks by releasing the figure canvas from Matplotlib's internal registry
    # How it works: Cleans up figure memory immediately after saving
    plt.close(fig)

    return target_path


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "output" / "charts"
    saved = create_line_chart(out_dir)
    print(f"[SUCCESS] Task 1 Line Chart saved to: {saved}")
