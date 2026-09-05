"""
Challenge 3: Executive KPI Metric Card Dashboard
Creates a dedicated dashboard featuring 4 prominent KPI metric cards
(Total Revenue, Net Profit, Total Orders, Active Customers) with trend context.
"""

# What is used: Non-interactive Agg backend
# Why it is used: Headless execution
# How it works: Sets "Agg" before importing pyplot
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from pathlib import Path


def create_kpi_dashboard(output_dir: Path) -> Path:
    """Generates an executive KPI card dashboard displaying top-line metric cards."""
    # Synthetic KPI parameters
    kpi_data = [
        {"title": "TOTAL REVENUE", "value": "₹52.48 L", "sub": "+22.4% vs Last Year", "color": "#1f77b4"},
        {"title": "NET PROFIT", "value": "₹14.82 L", "sub": "28.2% Net Margin", "color": "#2ca02c"},
        {"title": "TOTAL ORDERS", "value": "750", "sub": "Avg Order: ₹6,997", "color": "#ff7f0e"},
        {"title": "ACTIVE CUSTOMERS", "value": "184", "sub": "Repeat Rate: 64.2%", "color": "#9467bd"}
    ]

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    revenue = [320000, 350000, 340000, 390000, 420000, 410000, 450000, 480000, 470000, 530000, 610000, 680000]

    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1.2, 2.2], hspace=0.35, wspace=0.25)

    # Render 4 KPI cards in row 0
    for col_idx, kpi in enumerate(kpi_data):
        ax_card = fig.add_subplot(gs[0, col_idx])
        # Turn off traditional coordinate axes
        ax_card.axis("off")
        
        # What is used: FancyBboxPatch simulation via rounded box
        # Why it is used: Creates modern tile-like visual card containers
        # How it works: Draws text within a rounded bounding box
        card_text = f"{kpi['title']}\n\n{kpi['value']}\n\n{kpi['sub']}"
        ax_card.text(
            0.5, 0.5,
            card_text,
            ha="center", va="center",
            fontsize=11,
            fontweight="bold",
            color=kpi["color"],
            bbox=dict(boxstyle="round,pad=0.8", facecolor="#f8f9fa", edgecolor=kpi["color"], lw=1.8)
        )

    # Bottom main chart: Revenue progression spanning all 4 columns
    ax_bottom = fig.add_subplot(gs[1, :])
    ax_bottom.plot(months, revenue, marker="o", color="#1f77b4", linewidth=2.5, label="Monthly Revenue (₹)")
    ax_bottom.set_title("Supporting Metric Trend: 12-Month Revenue Trajectory", fontsize=13, fontweight="bold")
    ax_bottom.set_ylabel("Revenue (₹)", fontsize=10, fontweight="bold")
    ax_bottom.grid(True, linestyle="--", alpha=0.5)
    ax_bottom.legend(loc="upper left")

    fig.suptitle("Executive Commercial KPI Dashboard (2026)", fontsize=16, fontweight="bold", y=0.98)

    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / "challenge3_kpi_dashboard.png"

    fig.savefig(target_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return target_path


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "output" / "charts"
    saved = create_kpi_dashboard(out)
    print(f"[SUCCESS] Challenge 3 saved to: {saved}")