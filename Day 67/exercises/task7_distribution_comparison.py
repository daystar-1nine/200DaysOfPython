"""
Day 67 Task 7: Normal Distribution Variance Scaling
Compares N(50, 5), N(50, 10), and N(50, 20) demonstrating the effect
of increasing variance on peak height and spread.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

def main():
    mu = 50.0
    sigmas = [5.0, 10.0, 20.0]
    colors = ["#2563EB", "#059669", "#DC2626"]
    
    x = np.linspace(0, 100, 500)
    
    fig, ax = plt.subplots(figsize=(9, 5))
    
    for s, c in zip(sigmas, colors):
        pdf = norm.pdf(x, loc=mu, scale=s)
        ax.plot(x, pdf, label=f"N(mu={mu:.0f}, sigma={s:.0f})", color=c, linewidth=2)
        ax.fill_between(x, pdf, alpha=0.15, color=c)
        
    ax.set_title("Effect of Standard Deviation on Normal Distribution", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("x", fontsize=11)
    ax.set_ylabel("Probability Density f(x)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True)
    
    plt.tight_layout()
    chart_path = Path(__file__).resolve().parent.parent / "output" / "charts" / "normal_comparison.png"
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(chart_path, dpi=200)
    plt.close(fig)
    
    print("=" * 60)
    print("  TASK 7: NORMAL VARIANCE COMPARISON")
    print("=" * 60)
    print(f"Chart saved to: {chart_path}")
    print("Observation:")
    print("- As sigma increases from 5 to 20, the distribution spreads wider.")
    print("- Peak density drops proportionally to keep the total area under curve equal to 1.0.")

if __name__ == "__main__":
    main()
