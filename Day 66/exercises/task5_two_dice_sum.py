"""
Task 5 — Two Dice Sum Simulation: P(Sum = 7)
Simulate rolling two independent dice and track the probability of the sum equalling 7.
"""

import numpy as np

rng = np.random.default_rng(42)
trials = 100000

die1 = rng.integers(1, 7, size=trials)
die2 = rng.integers(1, 7, size=trials)
sums = die1 + die2

sum7_count = np.sum(sums == 7)
p_sum7_exp = sum7_count / trials
p_sum7_theo = 6.0 / 36.0  # 6 combinations out of 36

print("=== TASK 5: TWO DICE SUM EQUALS 7 ===")
print("Combinations yielding 7: (1,6), (2,5), (3,4), (4,3), (5,2), (6,1) -> 6 / 36 = 1/6")
print(f"Total Trials:              {trials:,}")
print(f"Observed Sum=7:            {sum7_count:,}")
print(f"Experimental P(Sum=7):     {p_sum7_exp:.5f}")
print(f"Theoretical P(Sum=7):      {p_sum7_theo:.5f} (1/6)")
print(f"Difference:                {abs(p_sum7_exp - p_sum7_theo):.5f}")
