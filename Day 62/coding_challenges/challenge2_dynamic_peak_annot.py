"""
Challenge 2: Automated Peak Month Detection and Annotation
Dynamically discovers the highest revenue month using pandas idxmax()
and creates a high-visibility arrowed annotation without hard-coded coordinates.
"""

# What is used: Non-interactive Agg backend
# Why it is used: Headless execution
# How it works: Sets "Agg" before importing pyplot
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def create_dynamic_peak_annotation_chart(output_dir: Path) -> Path:
    """Discovers peak month automatically and annotates it on the revenue chart."""
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    revenue = [185000, 210000, 195000, 230000, 260000, 255000, 280000, 310000, 295000, 340000, 420000, 390000]

    df = pd.DataFrame({"Month": months, "Revenue": revenue})

    # What is used: df["Revenue"].idxmax()
    # Why it is used: Locates highest value index programmatically
    # How it works: Evaluates Series maximum and returns corresponding row position
    peak_idx = df["Revenue"].idxmax()
    peak_month = df.loc[peak_idx, "Month"]
    peak_val = df.loc[peak_idx, "Revenue"]

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.plot(df["Month"], df["Revenue"], marker="o", color="#3182bd", linewidth=2.5, markersize=6, label="Monthly Revenue")

    # What is used: ax.annotate() with dynamic textcoords and arrowprops
    # Why it is used: Programmatically directs viewer attention to all-time revenue peak
    # How it works: Places pointer arrow directly at (peak_month, peak_val)
    ax.annotate(
        f"HISTORIC PEAK\nMonth: {peak_month}\nRevenue: ₹{peak_val:,}",
        xy=(peak_month, peak_val),
        xytext=(-70, 30),
        textcoords="offset points",
        arrowprops=dict(facecolor="#de2d26", edgecolor="#a50f15", arrowstyle="->", lw=2.0),
        fontsize=9.5,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#fee5d9", edgecolor="#fcae91")
    )

    # Formatting
    ax.set_title("Automated High-Performance Peak Month Identification", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Operational Month", fontsize=11, fontweight="bold")
    ax.set_ylabel("Revenue (₹)", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")

    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / "challenge2_dynamic_peak_annot.png"

    fig.tight_layout()
    fig.savefig(target_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return target_path


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "output" / "charts"
    saved = create_dynamic_peak_annotation_chart(out)
    print(f"[SUCCESS] Challenge 2 saved to: {saved}")