"""
Task 4 — Probability of Rolling a Number Greater Than 4
Simulate dice rolls and evaluate P(X > 4) = P({5, 6}) = 2/6 = 1/3.
"""

import numpy as np

rng = np.random.default_rng(42)
trials = 100000
rolls = rng.integers(1, 7, size=trials)

gt4_count = np.sum(rolls > 4)
p_gt4_exp = gt4_count / trials
p_gt4_theo = 2.0 / 6.0

print("=== TASK 4: PROBABILITY OF DIE ROLL > 4 ===")
print(f"Favorable Outcomes: {{5, 6}} -> 2 outcomes")
print(f"Total Trials:              {trials:,}")
print(f"Observed Rolls > 4:        {gt4_count:,}")
print(f"Experimental P(X > 4):     {p_gt4_exp:.5f}")
print(f"Theoretical P(X > 4):      {p_gt4_theo:.5f} (1/3)")
print(f"Difference:                {abs(p_gt4_exp - p_gt4_theo):.5f}")
