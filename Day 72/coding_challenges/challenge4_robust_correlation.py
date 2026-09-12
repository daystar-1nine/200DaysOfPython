"""
Day 72 — Challenge 4: Robust Correlation Benchmarking
Compares Pearson r, Spearman rho, Kendall's Tau, and Winsorized correlation across
clean Gaussian distributions versus heavy outlier contamination.
"""
import numpy as np
from scipy import stats

def winsorized_pearson(x: np.ndarray, y: np.ndarray, trim: float = 0.10) -> float:
    # 10% Winsorization on both arrays
    x_win = stats.mstats.winsorize(x, limits=[trim, trim])
    y_win = stats.mstats.winsorize(y, limits=[trim, trim])
    return float(stats.pearsonr(x_win, y_win).statistic)

def main():
    np.random.seed(42)
    n = 200
    
    # 1. Clean synthetic bivariate normal with true rho = 0.75
    mean = [0, 0]
    cov = [[1.0, 0.75], [0.75, 1.0]]
    clean_data = np.random.multivariate_normal(mean, cov, size=n)
    x_clean, y_clean = clean_data[:, 0], clean_data[:, 1]
    
    # 2. Contaminated with 5% catastrophic discordant outliers
    x_contam = x_clean.copy()
    y_contam = y_clean.copy()
    n_outliers = 10
    outlier_idx = np.random.choice(n, size=n_outliers, replace=False)
    x_contam[outlier_idx] = np.random.uniform(8.0, 12.0, size=n_outliers)
    y_contam[outlier_idx] = np.random.uniform(-12.0, -8.0, size=n_outliers)
    
    # Compute metrics on Clean
    r_c = stats.pearsonr(x_clean, y_clean).statistic
    rho_c = stats.spearmanr(x_clean, y_clean).statistic
    tau_c = stats.kendalltau(x_clean, y_clean).statistic
    win_c = winsorized_pearson(x_clean, y_clean, trim=0.05)
    
    # Compute metrics on Contaminated
    r_bad = stats.pearsonr(x_contam, y_contam).statistic
    rho_bad = stats.spearmanr(x_contam, y_contam).statistic
    tau_bad = stats.kendalltau(x_contam, y_contam).statistic
    win_bad = winsorized_pearson(x_contam, y_contam, trim=0.08)
    
    print("=" * 70)
    print("DAY 72 — CHALLENGE 4: ROBUST CORRELATION BENCHMARK")
    print("=" * 70)
    print(f"{'Correlation Method':<25} | {'Clean Data (N=200)':<18} | {'Contaminated (5% Outliers)':<26}")
    print("-" * 75)
    print(f"{'Pearson r':<25} | {r_c:<18.4f} | {r_bad:<26.4f} (Degraded by {abs(r_c-r_bad):.4f})")
    print(f"{'Spearman rho':<25} | {rho_c:<18.4f} | {rho_bad:<26.4f} (Degraded by {abs(rho_c-rho_bad):.4f})")
    print(f"{'Kendall Tau':<25} | {tau_c:<18.4f} | {tau_bad:<26.4f} (Degraded by {abs(tau_c-tau_bad):.4f})")
    print(f"{'Winsorized Pearson':<25} | {win_c:<18.4f} | {win_bad:<26.4f} (Degraded by {abs(win_c-win_bad):.4f})")
    print("-" * 75)
    
    assert r_bad < 0.0, "Pearson r should be inverted or decimated by outliers!"
    assert rho_bad > 0.40, "Spearman should remain robust above 0.40!"
    assert win_bad > 0.35, "Winsorized Pearson should remain positive!"
    print("\nTakeaway: In noisy data environments, rank and Winsorized correlations")
    print("prevent erroneous strategic conclusions driven by leverage points.")

if __name__ == "__main__":
    main()
