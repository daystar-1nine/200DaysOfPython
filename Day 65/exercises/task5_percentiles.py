"""
Task 5 — Percentiles: P25, P50, P75, P90
Calculate percentiles using NumPy and Pandas.
"""

import numpy as np
import pandas as pd

values = [12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78]

percentiles_to_calc = [25, 50, 75, 90]
np_results = {p: float(np.percentile(values, p)) for p in percentiles_to_calc}

series = pd.Series(values)
pd_results = {p: float(series.quantile(p / 100.0)) for p in percentiles_to_calc}

print("=== TASK 5: PERCENTILES ===")
print(f"Dataset: {values}")
for p in percentiles_to_calc:
    print(f"P{p:02d}: NumPy = {np_results[p]:.2f} | Pandas = {pd_results[p]:.2f}")
