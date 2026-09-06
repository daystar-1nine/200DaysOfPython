"""
Task 3 — Probability of Rolling an Even Number
Simulate dice rolls and evaluate P(Even) = P({2, 4, 6}) vs 0.5.
"""

import numpy as np

rng = np.random.default_rng(42)
trials = 100000
rolls = rng.integers(1, 7, size=trials)

even_mask = (rolls % 2 == 0)
even_count = np.sum(even_mask)
p_even_exp = even_count / trials
p_even_theo = 3.0 / 6.0

print("=== TASK 3: PROBABILITY OF AN EVEN DIE ROLL ===")
print(f"Sample Space: {{1, 2, 3, 4, 5, 6}}")
print(f"Favorable Event (Even): {{2, 4, 6}} -> 3 outcomes")
print(f"Total Trials:              {trials:,}")
print(f"Observed Even Rolls:       {even_count:,}")
print(f"Experimental P(Even):      {p_even_exp:.5f}")
print(f"Theoretical P(Even):       {p_even_theo:.5f} (3/6)")
print(f"Difference:                {abs(p_even_exp - p_even_theo):.5f}")
