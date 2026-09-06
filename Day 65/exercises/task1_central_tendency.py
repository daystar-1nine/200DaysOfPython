"""
Task 1 — Central Tendency: Mean, Median, Mode
Calculate central tendency metrics manually and using NumPy & Pandas.
"""

from collections import Counter
import numpy as np
import pandas as pd

values = [10, 20, 20, 30, 40, 50, 50, 50, 60, 70]

# 1. Manual Calculations
# Mean
manual_mean = sum(values) / len(values)

# Median
sorted_vals = sorted(values)
n = len(sorted_vals)
if n % 2 == 1:
    manual_median = float(sorted_vals[n // 2])
else:
    manual_median = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0

# Mode
counts = Counter(values)
max_freq = max(counts.values())
manual_mode = [val for val, freq in counts.items() if freq == max_freq]

# 2. NumPy / Pandas Calculations
np_mean = float(np.mean(values))
np_median = float(np.median(values))
series = pd.Series(values)
pd_mode = series.mode().tolist()

print("=== TASK 1: CENTRAL TENDENCY ===")
print(f"Dataset: {values}")
print(f"Number of observations: {n}")
print(f"Manual Mean:   {manual_mean:.4f}  | NumPy Mean:   {np_mean:.4f}")
print(f"Manual Median: {manual_median:.4f}  | NumPy Median: {np_median:.4f}")
print(f"Manual Mode:   {manual_mode}        | Pandas Mode:  {pd_mode}")
