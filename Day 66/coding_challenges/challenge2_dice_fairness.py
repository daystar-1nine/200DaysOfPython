"""
Day 66 Coding Challenge 2: Chi-Square Dice Fairness Test
Simulates 1,000,000 rolls of fair and slightly biased dice, applying
Goodness-of-Fit Chi-Square testing to mathematically detect loaded dice.
"""
import numpy as np
from scipy import stats

def test_dice_fairness(rolls: np.ndarray, alpha: float = 0.05) -> dict:
    """
    Perform a Chi-Square Goodness-of-Fit test against a uniform 6-sided die.
    H0: The die is fair (all outcomes equally probable at 1/6).
    H1: The die is loaded/biased.
    """
    n_rolls = len(rolls)
    observed_counts = np.bincount(rolls, minlength=7)[1:7]  # faces 1 to 6
    expected_counts = np.full(6, n_rolls / 6.0)
    
    chi2_stat, p_value = stats.chisquare(f_obs=observed_counts, f_exp=expected_counts)
    is_fair = bool(p_value >= alpha)
    
    face_proportions = {f"Face {i+1}": round(count / n_rolls, 5) for i, count in enumerate(observed_counts)}
    
    return {
        "total_rolls": n_rolls,
        "observed_counts": observed_counts.tolist(),
        "expected_counts": expected_counts.tolist(),
        "face_proportions": face_proportions,
        "chi2_statistic": round(float(chi2_stat), 4),
        "p_value": float(p_value),
        "alpha": alpha,
        "is_fair": is_fair,
        "verdict": "Fail to reject H0 (Die appears fair)" if is_fair else "Reject H0 (Die is statistically biased)"
    }

def main():
    print("=" * 70)
    print("  CHALLENGE 2: CHI-SQUARE DICE FAIRNESS VERIFICATION")
    print("=" * 70)
    
    rng = np.random.default_rng(42)
    n = 1_000_000
    
    # 1. Fair Die
    fair_rolls = rng.integers(low=1, high=7, size=n)
    fair_res = test_dice_fairness(fair_rolls)
    
    print("\n[Test 1: Standard Fair Die Simulation (N = 1,000,000)]")
    print(f"Chi2 Stat: {fair_res['chi2_statistic']} | p-value: {fair_res['p_value']:.4f}")
    print(f"Verdict:   {fair_res['verdict']}")
    print(f"Empirical Proportions: {fair_res['face_proportions']}")
    
    # 2. Loaded Die (Face 6 has 20% probability instead of 16.67%)
    biased_probs = [0.16, 0.16, 0.16, 0.16, 0.16, 0.20]
    biased_rolls = rng.choice([1, 2, 3, 4, 5, 6], size=n, p=biased_probs)
    biased_res = test_dice_fairness(biased_rolls)
    
    print("\n[Test 2: Biased Die Simulation (N = 1,000,000, Face 6 loaded at 20%)]")
    print(f"Chi2 Stat: {biased_res['chi2_statistic']} | p-value: {biased_res['p_value']}")
    print(f"Verdict:   {biased_res['verdict']}")
    print(f"Empirical Proportions: {biased_res['face_proportions']}")

if __name__ == "__main__":
    main()
