"""
Card deck simulation and poker/card event probability models.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd

SUITS = ["Hearts", "Diamonds", "Clubs", "Spades"]
RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King", "Ace"]

@dataclass(frozen=True)
class Card:
    rank: str
    suit: str
    
    @property
    def is_red(self) -> bool:
        return self.suit in ("Hearts", "Diamonds")
        
    @property
    def is_face(self) -> bool:
        return self.rank in ("Jack", "Queen", "King")

class CardDeck:
    def __init__(self):
        self.cards = [Card(r, s) for s in SUITS for r in RANKS]
        
    def __len__(self):
        return len(self.cards)

def simulate_card_draws(n_draws: int = 100_000, seed: int = 42) -> dict:
    """Simulate random single-card draws with replacement to measure event probabilities."""
    deck = CardDeck().cards
    n_deck = len(deck)
    rng = np.random.default_rng(seed)
    drawn_indices = rng.integers(0, n_deck, size=n_draws)
    
    drawn_cards = [deck[i] for i in drawn_indices]
    
    # Event tracking
    n_aces = sum(1 for c in drawn_cards if c.rank == "Ace")
    n_red = sum(1 for c in drawn_cards if c.is_red)
    n_face = sum(1 for c in drawn_cards if c.is_face)
    n_hearts = sum(1 for c in drawn_cards if c.suit == "Hearts")
    n_red_face = sum(1 for c in drawn_cards if c.is_red and c.is_face)
    
    events = [
        {"Event": "Ace", "Count": n_aces, "Theo_Prob": 4/52, "Emp_Prob": n_aces/n_draws},
        {"Event": "Red Card", "Count": n_red, "Theo_Prob": 26/52, "Emp_Prob": n_red/n_draws},
        {"Event": "Face Card", "Count": n_face, "Theo_Prob": 12/52, "Emp_Prob": n_face/n_draws},
        {"Event": "Heart", "Count": n_hearts, "Theo_Prob": 13/52, "Emp_Prob": n_hearts/n_draws},
        {"Event": "Red Face Card", "Count": n_red_face, "Theo_Prob": 6/52, "Emp_Prob": n_red_face/n_draws},
    ]
    
    df_cards = pd.DataFrame(events)
    df_cards["Abs_Error"] = (df_cards["Emp_Prob"] - df_cards["Theo_Prob"]).abs().round(6)
    df_cards["Emp_Prob"] = df_cards["Emp_Prob"].round(6)
    df_cards["Theo_Prob"] = df_cards["Theo_Prob"].round(6)
    
    return {
        "n_draws": n_draws,
        "df_cards": df_cards
    }
