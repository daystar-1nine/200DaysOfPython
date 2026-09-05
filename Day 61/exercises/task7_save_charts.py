"""
Task 7: Reusable Chart Saving Architecture
Demonstrates a production-grade utility function `save_chart()` that encapsulates
layout fitting, high-DPI rasterization, margin clipping, and mandatory figure memory closure.
"""

# What is used: Non-interactive backend selection
# Why it is used: Allows headless rendering across automated test and CLI environments
# How it works: Activates "Agg" backend before importing pyplot
import matplotlib
matplotlib.use("Agg")

# What is used: pyplot module from Matplotlib
# Why it is used: Core plotting interface
# How it works: Instantiates figures and axes
import matplotlib.pyplot as plt
from pathlib import Path


def save_chart(fig: plt.Figure, path: Path, dpi: int = 300) -> Path:
    """
    Saves a Matplotlib figure safely and frees memory.

    # What is used: fig.tight_layout(), fig.savefig(), and plt.close(fig)
    # Why it is used: Prevents clipped labels, exports publication-grade resolution, and prevents memory leaks
    # How it works: Auto-adjusts padding, renders bitmap to disk, and deregisters figure from pyplot
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def run_batch_demo(output_dir: Path) -> list[Path]:
    """Runs a batch generation demonstrating the reusable save_chart helper."""
    saved_paths = []

    # Chart A: Simple metric progression
    fig_a, ax_a = plt.subplots(figsize=(8, 4.5))
    ax_a.plot([1, 2, 3, 4, 5], [10, 18, 24, 35, 50], marker="o", color="#2a75a9", linewidth=2.0)
    ax_a.set_title("System Scalability Benchmark (Transactions/sec)", fontsize=13, fontweight="bold")
    ax_a.set_xlabel("Worker Thread Count", fontsize=10)
    ax_a.set_ylabel("Throughput (TPS)", fontsize=10)
    ax_a.grid(True, linestyle=":", alpha=0.5)
    saved_paths.append(save_chart(fig_a, output_dir / "task7_batch_throughput.png"))

    # Chart B: Resource utilization bars
    fig_b, ax_b = plt.subplots(figsize=(8, 4.5))
    servers = ["Node-01", "Node-02", "Node-03", "Node-04"]
    cpu_pct = [45, 78, 62, 31]
    bars = ax_b.bar(servers, cpu_pct, color="#e66101", edgecolor="#5c2600", width=0.5)
    ax_b.bar_label(bars, fmt="%d%%", padding=3, fontweight="bold")
    ax_b.set_ylim(0, 100)
    ax_b.set_title("Cluster CPU Utilization Audit", fontsize=13, fontweight="bold")
    ax_b.set_xlabel("Cluster Node", fontsize=10)
    ax_b.set_ylabel("CPU Usage (%)", fontsize=10)
    ax_b.grid(axis="y", linestyle=":", alpha=0.5)
    saved_paths.append(save_chart(fig_b, output_dir / "task7_batch_cpu_usage.png"))

    return saved_paths


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "output" / "charts"
    saved = run_batch_demo(out_dir)
    for p in saved:
        print(f"[SUCCESS] Saved via reusable helper: {p}")
