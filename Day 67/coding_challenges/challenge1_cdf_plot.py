"""
Day 67 Challenge 1: Multi-Distribution Cumulative Distribution Function (CDF) Plots
Visualizes and compares CDF shapes across Normal, Continuous Uniform, and Poisson distributions.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform, poisson

def plot_cdf_comparison(output_path: str | Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    # 1. Normal CDF (Smooth S-curve)
    x_norm = np.linspace(-3.5, 3.5, 300)
    y_norm = norm.cdf(x_norm, loc=0, scale=1)
    axes[0].plot(x_norm, y_norm, color="#2563EB", linewidth=2.5)
    axes[0].set_title("Normal CDF (mu=0, sigma=1)\nSmooth Sigmoidal S-Curve", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("F(x) = P(X <= x)")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    
    # 2. Uniform CDF (Piecewise Linear Ramp)
    x_unif = np.linspace(-1, 11, 300)
    y_unif = uniform.cdf(x_unif, loc=0, scale=10)
    axes[1].plot(x_unif, y_unif, color="#059669", linewidth=2.5)
    axes[1].set_title("Uniform CDF (a=0, b=10)\nLinear Constant Slope Ramp", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("x")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    
    # 3. Poisson CDF (Discrete Step Function)
    x_pois = np.arange(0, 15)
    y_pois = poisson.cdf(x_pois, mu=4.0)
    axes[2].step(x_pois, y_pois, where="post", color="#DC2626", linewidth=2.5)
    axes[2].scatter(x_pois, y_pois, color="#DC2626", s=25)
    axes[2].set_title("Poisson CDF (lambda=4.0)\nDiscrete Staircase Step Function", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("k (Count)")
    axes[2].grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"CDF comparison figure saved to: {output_path}")

def main():
    print("=" * 65)
    print("  CHALLENGE 1: MULTI-DISTRIBUTION CDF COMPARISON")
    print("=" * 65)
    chart_path = Path(__file__).resolve().parent.parent / "output" / "charts" / "cdf_comparison.png"
    plot_cdf_comparison(chart_path)
    print("CDF Characteristics:")
    print("1. Normal: Smooth sigmoidal curve reflecting dense probability mass near mean.")
    print("2. Uniform: Perfectly linear diagonal ramp reflecting constant probability density.")
    print("3. Poisson: Monotonic step function with jump discontinuities at integer outcomes.")

if __name__ == "__main__":
    main()
