"""
Day 66 Coding Challenge 5: Monte Carlo Pi Estimation & Geometric Probability
Approximates Pi by uniform sampling on [-1, 1] x [-1, 1], tracking error
and producing a visual scatter plot saved to output/charts/monte_carlo_pi.png.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def estimate_pi_monte_carlo(n_samples: int = 200_000, seed: int = 42) -> dict:
    """Estimate Pi using uniform bivariate sampling."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=n_samples)
    y = rng.uniform(-1.0, 1.0, size=n_samples)
    
    dists_sq = x**2 + y**2
    inside_mask = dists_sq <= 1.0
    n_inside = int(np.sum(inside_mask))
    
    pi_estimate = 4.0 * n_inside / n_samples
    abs_error = abs(pi_estimate - np.pi)
    rel_error_pct = (abs_error / np.pi) * 100
    
    return {
        "n_samples": n_samples,
        "n_inside": n_inside,
        "pi_estimate": round(pi_estimate, 6),
        "true_pi": round(float(np.pi), 6),
        "abs_error": round(abs_error, 6),
        "rel_error_pct": round(rel_error_pct, 4),
        "x": x,
        "y": y,
        "inside_mask": inside_mask
    }

def plot_monte_carlo_pi(res: dict, output_path: str | Path):
    """Plot Monte Carlo circle quadrant and scatter points."""
    sample_sub = 5_000  # subsample points for fast, clean rendering
    x_sub = res["x"][:sample_sub]
    y_sub = res["y"][:sample_sub]
    mask_sub = res["inside_mask"][:sample_sub]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(x_sub[mask_sub], y_sub[mask_sub], color="#2563EB", s=4, alpha=0.6, label="Inside Circle (r <= 1)")
    ax.scatter(x_sub[~mask_sub], y_sub[~mask_sub], color="#DC2626", s=4, alpha=0.6, label="Outside Circle (r > 1)")
    
    # Draw boundary circle
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color="#1E293B", linewidth=2.5, linestyle="--", label="Unit Circle (x^2 + y^2 = 1)")
    
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.6)
    
    ax.set_title(
        f"Monte Carlo Pi Estimation (N={res['n_samples']:,})\n"
        f"Estimate: {res['pi_estimate']:.5f} | True Pi: {res['true_pi']:.5f} | Error: {res['rel_error_pct']:.3f}%",
        fontsize=12, fontweight="bold", pad=12
    )
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.legend(loc="upper right", frameon=True)
    
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Chart saved successfully: {output_path}")

def main():
    print("=" * 70)
    print("  CHALLENGE 5: MONTE CARLO PI ESTIMATION")
    print("=" * 70)
    
    res = estimate_pi_monte_carlo(n_samples=250_000, seed=2026)
    print(f"Total Points:     {res['n_samples']:,}")
    print(f"Points Inside:    {res['n_inside']:,}")
    print(f"Estimated Pi:     {res['pi_estimate']}")
    print(f"True Pi:          {res['true_pi']}")
    print(f"Absolute Error:   {res['abs_error']}")
    print(f"Rel Error (%):    {res['rel_error_pct']:.4f}%")
    
    chart_path = Path(__file__).resolve().parent.parent / "output" / "charts" / "monte_carlo_pi.png"
    plot_monte_carlo_pi(res, chart_path)

if __name__ == "__main__":
    main()
