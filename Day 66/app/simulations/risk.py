"""
Financial decision and risk games simulation comparing theoretical vs empirical expected values.
"""
import numpy as np
import pandas as pd

def simulate_financial_risk_bets(n_simulations: int = 100_000, seed: int = 42) -> dict:
    """
    Simulate three distinct financial/gambling propositions:
    1. Roulette Red Bet (P=18/38, payout 1:1, loss -1)
    2. Corporate Product Launch (P(Success)=0.4, Gain 50k, Loss -20k, Cost 5k)
    3. High-Risk Crypto Venture (P=0.05, Gain 1M, P=0.95, Loss -25k)
    """
    rng = np.random.default_rng(seed)
    
    # 1. Roulette Red
    p_roulette_win = 18 / 38
    ev_roulette_theo = (p_roulette_win * 1.0) + ((1 - p_roulette_win) * -1.0)
    roulette_outcomes = rng.choice([1.0, -1.0], size=n_simulations, p=[p_roulette_win, 1 - p_roulette_win])
    ev_roulette_emp = np.mean(roulette_outcomes)
    
    # 2. Product Launch
    p_launch_win = 0.40
    ev_launch_theo = (0.40 * 50_000) + (0.60 * -20_000)
    launch_outcomes = rng.choice([50_000.0, -20_000.0], size=n_simulations, p=[p_launch_win, 1 - p_launch_win])
    ev_launch_emp = np.mean(launch_outcomes)
    
    # 3. High Risk Venture
    p_venture_win = 0.05
    ev_venture_theo = (0.05 * 1_000_000) + (0.95 * -25_000)
    venture_outcomes = rng.choice([1_000_000.0, -25_000.0], size=n_simulations, p=[p_venture_win, 1 - p_venture_win])
    ev_venture_emp = np.mean(venture_outcomes)
    
    df_bets = pd.DataFrame([
        {
            "Proposition": "Roulette Even-Money Bet ($1)",
            "Theoretical_EV": round(ev_roulette_theo, 4),
            "Empirical_EV": round(float(ev_roulette_emp), 4),
            "Abs_Difference": round(abs(ev_roulette_emp - ev_roulette_theo), 4),
            "Verdict": "Negative EV (House Advantage)"
        },
        {
            "Proposition": "Corporate Product Launch ($)",
            "Theoretical_EV": round(ev_launch_theo, 2),
            "Empirical_EV": round(float(ev_launch_emp), 2),
            "Abs_Difference": round(abs(ev_launch_emp - ev_launch_theo), 2),
            "Verdict": "Positive EV (Profitable Long-term)"
        },
        {
            "Proposition": "High-Risk Tech Venture ($)",
            "Theoretical_EV": round(ev_venture_theo, 2),
            "Empirical_EV": round(float(ev_venture_emp), 2),
            "Abs_Difference": round(abs(ev_venture_emp - ev_venture_theo), 2),
            "Verdict": "Positive EV (High Variance)"
        }
    ])
    
    return {
        "n_simulations": n_simulations,
        "df_bets": df_bets,
        "roulette_cum": np.cumsum(roulette_outcomes) / np.arange(1, n_simulations + 1),
        "launch_cum": np.cumsum(launch_outcomes) / np.arange(1, n_simulations + 1),
        "venture_cum": np.cumsum(venture_outcomes) / np.arange(1, n_simulations + 1)
    }
