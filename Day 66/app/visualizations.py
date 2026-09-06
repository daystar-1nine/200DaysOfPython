"""
Publication-grade probability charts and visual analytics.
All charts rendered headlessly using matplotlib 'Agg' backend.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.config import CHARTS_DIR

def plot_coin_convergence(coin_res: dict, output_path: Path = CHARTS_DIR / "coin_convergence.png"):
    """Plot cumulative heads proportion converging to 0.5 with theoretical error bounds."""
    trials = coin_res["trials"]
    cum_prop = coin_res["cum_prop"]
    
    fig, ax = plt.subplots(figsize=(10, 5.5))
    
    ax.plot(trials, cum_prop, color="#2563EB", linewidth=1.5, label="Empirical Proportion P_hat(Heads)")
    ax.axhline(0.5, color="#DC2626", linestyle="--", linewidth=2.0, label="Theoretical P(Heads) = 0.5")
    
    # 95% Confidence error bands (+/- 1.96 * sqrt(0.25 / N))
    upper_band = 0.5 + 1.96 * np.sqrt(0.25 / trials)
    lower_band = 0.5 - 1.96 * np.sqrt(0.25 / trials)
    
    ax.plot(trials, upper_band, color="#64748B", linestyle=":", alpha=0.8, label="Theoretical 95% CI Bands")
    ax.plot(trials, lower_band, color="#64748B", linestyle=":", alpha=0.8)
    ax.fill_between(trials, lower_band, upper_band, color="#E2E8F0", alpha=0.5)
    
    ax.set_xscale("log")
    ax.set_ylim(0.40, 0.60)
    ax.set_title("Law of Large Numbers: Coin Flip Convergence (N=50,000 Flips)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Flips (Log Scale)", fontsize=11)
    ax.set_ylabel("Proportion of Heads", fontsize=11)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {output_path}")

def plot_dice_distribution(dice_res: dict, output_path: Path = CHARTS_DIR / "dice_distribution.png"):
    """Plot single die roll counts vs uniform theoretical probability."""
    df_dist = dice_res["df_distribution"]
    
    fig, ax = plt.subplots(figsize=(9, 5))
    
    faces = df_dist["Face"]
    emp_probs = df_dist["Empirical_Probability"]
    theo_prob = df_dist["Theoretical_Probability"].iloc[0]
    
    bars = ax.bar(faces, emp_probs, color="#3B82F6", alpha=0.85, width=0.55, edgecolor="#1D4ED8", label="Empirical P(Face)")
    ax.axhline(theo_prob, color="#DC2626", linestyle="--", linewidth=2.0, label=f"Theoretical Uniform P = 1/6 (~{theo_prob:.4f})")
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.002, f"{yval:.4f}", ha="center", va="bottom", fontsize=10, fontweight="semibold")
        
    ax.set_ylim(0.12, 0.20)
    ax.set_xticks(faces)
    ax.set_title(f"Fair Die Uniformity Check (N={dice_res['n_rolls']:,} Rolls)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Die Face Value", fontsize=11)
    ax.set_ylabel("Empirical Probability", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {output_path}")

def plot_dice_sum_distribution(dice_two_res: dict, output_path: Path = CHARTS_DIR / "dice_sum_distribution.png"):
    """Plot triangular distribution of two dice sum."""
    df_sums = dice_two_res["df_sums"]
    
    fig, ax = plt.subplots(figsize=(10, 5.5))
    
    x = df_sums["Sum"]
    emp_p = df_sums["Empirical_Prob"]
    theo_p = df_sums["Theoretical_Prob"]
    
    ax.bar(x - 0.15, emp_p, width=0.3, color="#059669", alpha=0.85, label="Empirical Probability")
    ax.bar(x + 0.15, theo_p, width=0.3, color="#D97706", alpha=0.85, label="Theoretical Probability")
    
    ax.set_xticks(x)
    ax.set_title(f"Sum of Two Dice: Triangular Probability Distribution (N={dice_two_res['n_rolls']:,})", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Sum of Two Dice (X + Y)", fontsize=11)
    ax.set_ylabel("Probability P(Sum = k)", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {output_path}")

def plot_expected_vs_actual(risk_res: dict, output_path: Path = CHARTS_DIR / "expected_vs_actual.png"):
    """Compare theoretical expected value vs empirical simulated outcomes."""
    df_bets = risk_res["df_bets"]
    
    fig, ax = plt.subplots(figsize=(10, 5.5))
    
    props = [p.split("(")[0].strip() for p in df_bets["Proposition"]]
    # Normalize values for visual display or plot side-by-side
    x = np.arange(len(props))
    width = 0.35
    
    theo_vals = df_bets["Theoretical_EV"]
    emp_vals = df_bets["Empirical_EV"]
    
    rects1 = ax.bar(x - width/2, theo_vals, width, label="Theoretical E[X]", color="#2563EB", alpha=0.85)
    rects2 = ax.bar(x + width/2, emp_vals, width, label="Simulated Mean", color="#10B981", alpha=0.85)
    
    ax.set_title("Expected Value: Theoretical vs Empirical Mean (N=100,000)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(props, fontsize=10)
    ax.set_ylabel("Expected Value ($)", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.axhline(0, color="#64748B", linewidth=1.0)
    ax.legend(loc="upper left", frameon=True)
    
    # Add values on top of bars
    for rect in rects1:
        h = rect.get_height()
        va = "bottom" if h >= 0 else "top"
        ax.annotate(f"{h:,.1f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3 if h>=0 else -8),
                    textcoords="offset points", ha="center", va=va, fontsize=9)
    for rect in rects2:
        h = rect.get_height()
        va = "bottom" if h >= 0 else "top"
        ax.annotate(f"{h:,.1f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3 if h>=0 else -8),
                    textcoords="offset points", ha="center", va=va, fontsize=9)
                    
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {output_path}")

def plot_fraud_probability(bayes_df: pd.DataFrame, output_path: Path = CHARTS_DIR / "fraud_probability.png"):
    """Plot Bayesian posterior fraud probability as a function of prior prevalence."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    
    priors_pct = bayes_df["Prior_Fraud_Rate"] * 100
    posteriors_pct = bayes_df["Posterior_P_Fraud_Given_Flag"] * 100
    
    ax.plot(priors_pct, posteriors_pct, marker="o", color="#7C3AED", linewidth=2.5, markersize=7, label="Posterior P(Fraud | Flagged)")
    
    for _, row in bayes_df.iterrows():
        px = row["Prior_Fraud_Rate"] * 100
        py = row["Posterior_P_Fraud_Given_Flag"] * 100
        ax.annotate(f"{py:.1f}%", xy=(px, py), xytext=(-5, 8), textcoords="offset points", fontsize=9, fontweight="bold")
        
    ax.set_title("Bayesian Base Rate Fallacy: Fraud Detection Engine\nSensitivity=99%, False Alarm Rate=1%", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Prior Fraud Rate in Population (%)", fontsize=11)
    ax.set_ylabel("Posterior Probability Given Alarm (%)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", frameon=True)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {output_path}")
