"""
Coin flip simulations and Law of Large Numbers convergence analysis.
"""
import numpy as np
import pandas as pd

def simulate_coin_flips(n_flips: int = 50_000, p_heads: float = 0.5, seed: int = 42) -> dict:
    """Simulate a sequence of Bernoulli coin tosses and compute running statistics."""
    rng = np.random.default_rng(seed)
    # 1 for heads, 0 for tails
    flips = rng.binomial(n=1, p=p_heads, size=n_flips)
    cum_heads = np.cumsum(flips)
    trials = np.arange(1, n_flips + 1)
    cum_prop = cum_heads / trials
    
    final_heads = int(cum_heads[-1])
    empirical_p = final_heads / n_flips
    abs_diff = abs(empirical_p - p_heads)
    
    df_results = pd.DataFrame({
        "Trial": trials,
        "Outcome": flips,
        "Cumulative_Heads": cum_heads,
        "Cumulative_Proportion": np.round(cum_prop, 6)
    })
    
    return {
        "n_flips": n_flips,
        "final_heads": final_heads,
        "final_tails": n_flips - final_heads,
        "empirical_p": round(empirical_p, 6),
        "theoretical_p": p_heads,
        "abs_error": round(abs_diff, 6),
        "trials": trials,
        "cum_prop": cum_prop,
        "df_results": df_results
    }

def run_coin_convergence_study(sample_sizes: list[int], p_heads: float = 0.5, seed: int = 42) -> pd.DataFrame:
    """Analyze convergence across multi-scale sample batches."""
    rng = np.random.default_rng(seed)
    records = []
    for n in sample_sizes:
        flips = rng.binomial(n=1, p=p_heads, size=n)
        heads = int(np.sum(flips))
        p_hat = heads / n
        se = np.sqrt(p_heads * (1 - p_heads) / n)
        err = abs(p_hat - p_heads)
        records.append({
            "N": n,
            "Heads": heads,
            "P_hat": round(p_hat, 6),
            "Abs_Error": round(err, 6),
            "Std_Error": round(se, 6),
            "Within_95CI": bool(err <= 1.96 * se)
        })
    return pd.DataFrame(records)
