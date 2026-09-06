"""
Task 9 — Expected Value of a Random Variable
Calculate theoretical expected value and compare with empirical simulation over 100,000 plays.
"""

import numpy as np

# Game specification
payouts = np.array([0, 20, 100, 500])
probabilities = np.array([0.50, 0.30, 0.15, 0.05])

# 1. Theoretical Expected Value: E(X) = sum(x * P(x))
theoretical_ev = np.sum(payouts * probabilities)

# 2. Experimental Simulation
rng = np.random.default_rng(42)
trials = 100000
simulated_payouts = rng.choice(payouts, size=trials, p=probabilities)
experimental_ev = np.mean(simulated_payouts)

print("=== TASK 9: EXPECTED VALUE SIMULATION ===")
print("Payout Table:")
for p, pr in zip(payouts, probabilities):
    print(f"  Rs. {p:<5} with probability {pr:.2f}")
print("-" * 50)
print(f"Theoretical Expected Value:  Rs. {theoretical_ev:.2f}")
print(f"Experimental Average Return: Rs. {experimental_ev:.2f} (over {trials:,} plays)")
print(f"Absolute Difference:         Rs. {abs(experimental_ev - theoretical_ev):.4f}")
print("-" * 50)
print("Insight: The player will average Rs. 46 per game in the long run.")
