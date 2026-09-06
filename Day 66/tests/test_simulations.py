"""
Unit and integration tests for stochastic simulations.
"""
import pytest
from app.simulations.coin import simulate_coin_flips, run_coin_convergence_study
from app.simulations.dice import simulate_dice_rolls, simulate_two_dice_sum
from app.simulations.cards import simulate_card_draws, CardDeck
from app.simulations.conversion import simulate_conversion_experiment
from app.simulations.risk import simulate_financial_risk_bets

def test_simulate_coin_flips():
    res = simulate_coin_flips(n_flips=1_000, p_heads=0.5, seed=42)
    assert res["n_flips"] == 1_000
    assert 400 <= res["final_heads"] <= 600
    assert len(res["df_results"]) == 1_000

def test_run_coin_convergence_study():
    df = run_coin_convergence_study([100, 1000, 10000], seed=42)
    assert len(df) == 3
    # Absolute error should generally diminish
    assert df.loc[2, "Abs_Error"] <= 0.05

def test_simulate_dice_rolls():
    res = simulate_dice_rolls(n_rolls=6_000, seed=42)
    assert len(res["df_distribution"]) == 6
    # Each face should be close to 1000
    for count in res["counts"]:
        assert 800 <= count <= 1200

def test_simulate_two_dice_sum():
    res = simulate_two_dice_sum(n_rolls=10_000, seed=42)
    df = res["df_sums"]
    assert len(df) == 11  # sums 2 to 12
    # 7 should have the maximum count
    max_sum = df.loc[df["Count"].idxmax(), "Sum"]
    assert max_sum == 7

def test_card_deck_and_draws():
    deck = CardDeck()
    assert len(deck) == 52
    
    res = simulate_card_draws(n_draws=5_000, seed=42)
    df = res["df_cards"]
    assert len(df) == 5
    red_row = df[df["Event"] == "Red Card"].iloc[0]
    assert 0.45 <= red_row["Emp_Prob"] <= 0.55

def test_simulate_conversion_experiment():
    res = simulate_conversion_experiment(n_visitors_control=5_000, n_visitors_variant=5_000, p_control=0.05, p_variant=0.07, seed=42)
    df = res["df_summary"]
    assert len(df) == 2
    assert res["z_score"] > 0
    assert res["relative_uplift_pct"] > 0

def test_simulate_financial_risk_bets():
    res = simulate_financial_risk_bets(n_simulations=5_000, seed=42)
    df = res["df_bets"]
    assert len(df) == 3
    roulette = df[df["Proposition"].str.contains("Roulette")].iloc[0]
    assert roulette["Theoretical_EV"] < 0
