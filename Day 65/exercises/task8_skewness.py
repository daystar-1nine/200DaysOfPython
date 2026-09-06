"""
Task 8 — Skewness: Symmetric vs Right-Skewed Distributions
Compare distribution asymmetry using Pandas .skew().
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# Dataset A: Approximately symmetric (normal distribution)
data_a = np.random.normal(loc=50, scale=10, size=500)

# Dataset B: Strongly right-skewed (lognormal distribution)
data_b = np.random.lognormal(mean=2.5, sigma=0.8, size=500)

series_a = pd.Series(data_a, name="Symmetric")
series_b = pd.Series(data_b, name="Right-Skewed")

skew_a = series_a.skew()
skew_b = series_b.skew()

print("=== TASK 8: SKEWNESS COMPARISON ===")
print(f"Dataset A (Normal)       -> Mean: {series_a.mean():.2f}, Median: {series_a.median():.2f}, Skewness: {skew_a:.4f}")
print(f"Dataset B (Right-Skewed) -> Mean: {series_b.mean():.2f}, Median: {series_b.median():.2f}, Skewness: {skew_b:.4f}")
print("\nInterpretation:")
print(f"-> Dataset A skewness is close to 0 ({skew_a:.2f}), confirming symmetry.")
print(f"-> Dataset B skewness is substantially positive ({skew_b:.2f}), confirming a heavy right tail (Mean > Median).")
