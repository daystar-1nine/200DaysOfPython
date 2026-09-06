"""
Single and double dice roll simulations and sum distributions.
"""
import numpy as np
import pandas as pd

def simulate_dice_rolls(n_rolls: int = 100_000, seed: int = 42) -> dict:
    """Simulate single fair 6-sided die rolls and calculate empirical distribution."""
    rng = np.random.default_rng(seed)
    rolls = rng.integers(1, 7, size=n_rolls)
    
    counts = np.bincount(rolls, minlength=7)[1:7]
    empirical_probs = counts / n_rolls
    theoretical_prob = 1.0 / 6.0
    
    df_distribution = pd.DataFrame({
        "Face": list(range(1, 7)),
        "Count": counts,
        "Empirical_Probability": np.round(empirical_probs, 6),
        "Theoretical_Probability": round(theoretical_prob, 6),
        "Absolute_Error": np.round(np.abs(empirical_probs - theoretical_prob), 6)
    })
    
    return {
        "n_rolls": n_rolls,
        "rolls": rolls,
        "counts": counts,
        "df_distribution": df_distribution
    }

def simulate_two_dice_sum(n_rolls: int = 200_000, seed: int = 42) -> dict:
    """Simulate sum of two independent 6-sided dice."""
    rng = np.random.default_rng(seed)
    die1 = rng.integers(1, 7, size=n_rolls)
    die2 = rng.integers(1, 7, size=n_rolls)
    sums = die1 + die2
    
    # Theoretical probabilities for sums 2 to 12
    # Combinations: 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:5, 9:4, 10:3, 11:2, 12:1 out of 36
    combos = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
    theo_probs = {s: c / 36.0 for s, c in combos.items()}
    
    counts = np.bincount(sums, minlength=13)[2:13]
    empirical_probs = counts / n_rolls
    
    df_sums = pd.DataFrame({
        "Sum": list(range(2, 13)),
        "Count": counts,
        "Empirical_Prob": np.round(empirical_probs, 6),
        "Theoretical_Prob": [round(theo_probs[s], 6) for s in range(2, 13)],
        "Abs_Diff": np.round([abs(empirical_probs[i] - theo_probs[i+2]) for i in range(11)], 6)
    })
    
    return {
        "n_rolls": n_rolls,
        "sums": sums,
        "df_sums": df_sums
    }
