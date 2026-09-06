"""
Task 2 — Dice Probability Simulation
Simulate 100,000 rolls of a fair six-sided die and compare frequencies against 1/6.
"""

import numpy as np

rng = np.random.default_rng(42)
trials = 100000
rolls = rng.integers(1, 7, size=trials)

faces, counts = np.unique(rolls, return_counts=True)
theoretical_p = 1.0 / 6.0

print("=== TASK 2: DICE PROBABILITY SIMULATION (100,000 ROLLS) ===")
print(f"Theoretical Probability per Face: 1/6 = {theoretical_p:.5f}")
print("-" * 60)
print(f"{'Face':<8}{'Observed Count':<18}{'Experimental P':<18}{'Abs Deviation':<15}")
print("-" * 60)

for face, count in zip(faces, counts):
    exp_p = count / trials
    dev = abs(exp_p - theoretical_p)
    print(f"{face:<8}{count:<18}{exp_p:<18.5f}{dev:<15.5f}")

print("-" * 60)
