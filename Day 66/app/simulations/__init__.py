"""Empirical stochastic simulation subpackage."""
try:
    from .coin import simulate_coin_flips, run_coin_convergence_study
    from .dice import simulate_dice_rolls, simulate_two_dice_sum
    from .cards import simulate_card_draws, CardDeck
    from .conversion import simulate_conversion_experiment
    from .risk import simulate_financial_risk_bets
except ImportError:
    from app.simulations.coin import simulate_coin_flips, run_coin_convergence_study
    from app.simulations.dice import simulate_dice_rolls, simulate_two_dice_sum
    from app.simulations.cards import simulate_card_draws, CardDeck
    from app.simulations.conversion import simulate_conversion_experiment
    from app.simulations.risk import simulate_financial_risk_bets

__all__ = [
    "simulate_coin_flips",
    "run_coin_convergence_study",
    "simulate_dice_rolls",
    "simulate_two_dice_sum",
    "simulate_card_draws",
    "CardDeck",
    "simulate_conversion_experiment",
    "simulate_financial_risk_bets",
]
