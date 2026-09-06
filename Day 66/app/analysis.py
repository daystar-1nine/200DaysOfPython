"""
Comprehensive analysis engine synthesizing simulations, Bayesian models, and tests.
"""
from pathlib import Path
import pandas as pd
from app.config import (
    DEFAULT_COIN_FLIPS,
    DEFAULT_DICE_ROLLS,
    DEFAULT_TWO_DICE_ROLLS,
    DEFAULT_CARD_DRAWS,
    DEFAULT_CONVERSION_VISITORS,
    RANDOM_SEED
)
from app.simulations.coin import simulate_coin_flips, run_coin_convergence_study
from app.simulations.dice import simulate_dice_rolls, simulate_two_dice_sum
from app.simulations.cards import simulate_card_draws
from app.simulations.conversion import simulate_conversion_experiment
from app.simulations.risk import simulate_financial_risk_bets
from app.probability.bayes import bayes_update

def run_full_pipeline() -> dict:
    """Execute all probability simulations and synthesize structured outputs."""
    print("--> Simulating Coin Toss Convergence...")
    coin_res = simulate_coin_flips(DEFAULT_COIN_FLIPS, p_heads=0.5, seed=RANDOM_SEED)
    coin_conv_df = run_coin_convergence_study(
        sample_sizes=[10, 50, 100, 500, 1_000, 5_000, 10_000, 50_000],
        seed=RANDOM_SEED
    )
    
    print("--> Simulating Single & Two-Dice Distributions...")
    dice_res = simulate_dice_rolls(DEFAULT_DICE_ROLLS, seed=RANDOM_SEED)
    dice_two_res = simulate_two_dice_sum(DEFAULT_TWO_DICE_ROLLS, seed=RANDOM_SEED)
    
    print("--> Simulating Card Deck Events...")
    card_res = simulate_card_draws(DEFAULT_CARD_DRAWS, seed=RANDOM_SEED)
    
    print("--> Simulating Marketing A/B Conversions...")
    conv_res = simulate_conversion_experiment(
        n_visitors_control=DEFAULT_CONVERSION_VISITORS,
        n_visitors_variant=DEFAULT_CONVERSION_VISITORS,
        seed=RANDOM_SEED
    )
    
    print("--> Simulating Financial Risk Expected Values...")
    risk_res = simulate_financial_risk_bets(n_simulations=100_000, seed=RANDOM_SEED)
    
    print("--> Calculating Bayesian Fraud Detection Curves...")
    priors = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
    # Sensitivity = 99%, FPR = 1%
    bayes_fraud_records = []
    for pr in priors:
        b_res = bayes_update(prior=pr, likelihood_true=0.99, likelihood_false=0.01)
        bayes_fraud_records.append({
            "Prior_Fraud_Rate": pr,
            "Posterior_P_Fraud_Given_Flag": b_res["posterior"],
            "Marginal_Flag_Rate": b_res["marginal_evidence"]
        })
    df_fraud_bayes = pd.DataFrame(bayes_fraud_records)
    
    return {
        "coin": coin_res,
        "coin_convergence": coin_conv_df,
        "dice": dice_res,
        "dice_two": dice_two_res,
        "cards": card_res,
        "conversion": conv_res,
        "risk": risk_res,
        "fraud_bayes": df_fraud_bayes
    }
