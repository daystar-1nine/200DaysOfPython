"""
Task 6 — Independent Events: Two Sequential Coin Flips
Simulate two independent coins and verify P(HH) = P(HT) = P(TH) = P(TT) = 0.25.
"""

import numpy as np

rng = np.random.default_rng(42)
trials = 100000

coin1 = rng.choice(["H", "T"], size=trials)
coin2 = rng.choice(["H", "T"], size=trials)

combos = np.char.add(coin1, coin2)
unique, counts = np.unique(combos, return_counts=True)

print("=== TASK 6: INDEPENDENT EVENTS (TWO COIN FLIPS) ===")
print("Theoretical P for each combination (HH, HT, TH, TT) = 0.5 * 0.5 = 0.25")
print("-" * 50)
print(f"{'Combination':<15}{'Count':<15}{'Experimental P':<15}")
print("-" * 50)

for combo, count in zip(unique, counts):
    exp_p = count / trials
    print(f"{combo:<15}{count:<15}{exp_p:<15.5f}")

print("-" * 50)
