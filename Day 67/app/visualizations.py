"""
Publication-grade probability distribution visualizations.
All charts rendered headlessly using matplotlib 'Agg' backend.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from app.config import CHARTS_DIR
from distributions.base import BaseDistribution

def plot_bernoulli(p: float = 0.7, output_path: Path = CHARTS_DIR / "bernoulli.png"):
    """Chart 1: Bernoulli PMF."""
    fig, ax = plt.subplots(figsize=(7, 5))
    x = [0, 1]
    probs = [1 - p, p]
    labels = ["0 (Failure)", "1 (Success)"]
    
    bars = ax.bar(x, probs, width=0.4, color=["#DC2626", "#2563EB"], alpha=0.85, edgecolor="#1E293B")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, h + 0.02, f"{h:.2f} ({h*100:.0f}%)", ha="center", va="bottom", fontsize=11, fontweight="bold")
        
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_title(f"Bernoulli Distribution PMF (p = {p})", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Probability P(X = x)", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {output_path}")

def plot_binomial(n: int = 20, p: float = 0.4, output_path: Path = CHARTS_DIR / "binomial.png"):
    """Chart 2: Binomial PMF."""
    fig, ax = plt.subplots(figsize=(9, 5))
    k = np.arange(0, n + 1)
    probs = stats.binom.pmf(k, n, p)
    
    markerline, stemlines, baseline = ax.stem(k, probs, linefmt="#2563EB", markerfmt="o", basefmt="gray")
    plt.setp(markerline, markersize=6, color="#1D4ED8")
    plt.setp(stemlines, linewidth=1.8, color="#3B82F6")
    
    # Highlight expected value
    ev = n * p
    ax.axvline(ev, color="#DC2626", linestyle="--", linewidth=2, label=f"E[X] = np = {ev:.1f}")
    
    ax.set_title(f"Binomial Distribution PMF (n = {n}, p = {p})", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Successes (k)", fontsize=11)
    ax.set_ylabel("Probability P(X = k)", fontsize=11)
    ax.set_xticks(k)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {output_path}")

def plot_uniform(a: float = 0.0, b: float = 10.0, output_path: Path = CHARTS_DIR / "uniform.png"):
    """Chart 3: Uniform PDF."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.linspace(a - 2, b + 2, 400)
    y = stats.uniform.pdf(x, loc=a, scale=b - a)
    
    ax.plot(x, y, color="#059669", linewidth=2.5, label=f"Uniform PDF: f(x) = {1/(b-a):.2f}")
    ax.fill_between(x, y, where=(x >= a) & (x <= b), color="#10B981", alpha=0.25)
    
    ax.set_ylim(-0.02, 1.2 / (b - a))
    ax.set_title(f"Continuous Uniform Distribution PDF [a={a}, b={b}]", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("x", fontsize=11)
    ax.set_ylabel("Probability Density f(x)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {output_path}")

def plot_normal(mu: float = 70.0, sigma: float = 10.0, output_path: Path = CHARTS_DIR / "normal.png"):
    """Chart 4: Normal PDF with Empirical Rule Bands."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.linspace(mu - 3.5*sigma, mu + 3.5*sigma, 500)
    y = stats.norm.pdf(x, loc=mu, scale=sigma)
    
    ax.plot(x, y, color="#1E293B", linewidth=2.5, label="Normal PDF")
    
    # 68%, 95%, 99.7% regions
    ax.fill_between(x, y, where=(x >= mu - sigma) & (x <= mu + sigma), color="#3B82F6", alpha=0.4, label="Within 1s (~68.3%)")
    ax.fill_between(x, y, where=((x >= mu - 2*sigma) & (x < mu - sigma)) | ((x > mu + sigma) & (x <= mu + 2*sigma)), color="#60A5FA", alpha=0.3, label="Within 2s (~95.5%)")
    ax.fill_between(x, y, where=((x >= mu - 3*sigma) & (x < mu - 2*sigma)) | ((x > mu + 2*sigma) & (x <= mu + 3*sigma)), color="#93C5FD", alpha=0.2, label="Within 3s (~99.7%)")
    
    ax.axvline(mu, color="#DC2626", linestyle="--", linewidth=1.8, label=f"Mean mu = {mu}")
    ax.set_title(f"Normal Distribution & Empirical Rule (mu = {mu}, sigma = {sigma})", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("x", fontsize=11)
    ax.set_ylabel("Density f(x)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {output_path}")

def plot_poisson(lam: float = 5.0, output_path: Path = CHARTS_DIR / "poisson.png"):
    """Chart 5: Poisson PMF."""
    fig, ax = plt.subplots(figsize=(9, 5))
    k = np.arange(0, 16)
    probs = stats.poisson.pmf(k, mu=lam)
    
    bars = ax.bar(k, probs, width=0.55, color="#D97706", alpha=0.85, edgecolor="#78350F")
    ax.axvline(lam, color="#DC2626", linestyle="--", linewidth=2, label=f"Mean = Variance = lambda = {lam}")
    
    for bar in bars:
        h = bar.get_height()
        if h >= 0.05:
            ax.text(bar.get_x() + bar.get_width()/2.0, h + 0.005, f"{h:.3f}", ha="center", va="bottom", fontsize=8)
            
    ax.set_title(f"Poisson Distribution PMF (lambda = {lam})", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Events (k)", fontsize=11)
    ax.set_ylabel("Probability P(X = k)", fontsize=11)
    ax.set_xticks(k)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {output_path}")

def plot_normal_comparison(mu: float = 50.0, sigmas: list[float] = [5.0, 10.0, 20.0], output_path: Path = CHARTS_DIR / "normal_comparison.png"):
    """Chart 6: Normal distribution comparison."""
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.linspace(0, 100, 500)
    colors = ["#2563EB", "#059669", "#DC2626"]
    
    for s, c in zip(sigmas, colors):
        pdf = stats.norm.pdf(x, loc=mu, scale=s)
        ax.plot(x, pdf, label=f"N(mu={mu:.0f}, sigma={s:.0f})", color=c, linewidth=2)
        ax.fill_between(x, pdf, alpha=0.15, color=c)
        
    ax.set_title("Comparison of Normal Distributions with Varying Spread (sigma)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("x", fontsize=11)
    ax.set_ylabel("Density f(x)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {output_path}")

def plot_binomial_simulation(n: int = 20, p: float = 0.5, n_sims: int = 10_000, output_path: Path = CHARTS_DIR / "binomial_simulation.png"):
    """Chart 7: Binomial theoretical vs simulated."""
    fig, ax = plt.subplots(figsize=(10, 5))
    k = np.arange(0, n + 1)
    theo_probs = stats.binom.pmf(k, n, p)
    
    sim_counts = stats.binom.rvs(n=n, p=p, size=n_sims, random_state=42)
    emp_freqs = np.bincount(sim_counts, minlength=n + 1) / n_sims
    
    width = 0.35
    ax.bar(k - width/2, theo_probs, width=width, label="Theoretical PMF", color="#2563EB", alpha=0.85)
    ax.bar(k + width/2, emp_freqs, width=width, label=f"Simulated Frequency (N={n_sims:,})", color="#10B981", alpha=0.85)
    
    ax.set_title(f"Binomial Theory vs Simulation (n={n}, p={p}, N={n_sims:,})", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Successes (k)", fontsize=11)
    ax.set_ylabel("Probability", fontsize=11)
    ax.set_xticks(k)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {output_path}")

def plot_normal_simulation(mu: float = 70.0, sigma: float = 10.0, n_sims: int = 10_000, output_path: Path = CHARTS_DIR / "normal_simulation.png"):
    """Chart 8: Normal theoretical PDF vs simulated histogram."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    samples = stats.norm.rvs(loc=mu, scale=sigma, size=n_sims, random_state=42)
    
    # Histogram of simulated observations
    ax.hist(samples, bins=50, density=True, color="#93C5FD", edgecolor="#1D4ED8", alpha=0.6, label=f"Simulated Histogram (N={n_sims:,})")
    
    # Theoretical PDF curve
    x = np.linspace(mu - 3.5*sigma, mu + 3.5*sigma, 400)
    y = stats.norm.pdf(x, loc=mu, scale=sigma)
    ax.plot(x, y, color="#DC2626", linewidth=2.5, label="Theoretical PDF N(70, 10)")
    
    ax.set_title(f"Normal Theory vs Simulation Histogram (N={n_sims:,})", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Value", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {output_path}")
