"""
Task 9 — Mean vs Median Sensitivity to Outliers
Demonstrate resistance of median versus volatility of mean under extreme values.
"""

import numpy as np
import pandas as pd

dataset_a = [10, 20, 30, 40, 50]
dataset_b = [10, 20, 30, 40, 500]

mean_a, median_a, std_a = np.mean(dataset_a), np.median(dataset_a), np.std(dataset_a, ddof=1)
mean_b, median_b, std_b = np.mean(dataset_b), np.median(dataset_b), np.std(dataset_b, ddof=1)

print("=== TASK 9: MEAN VS MEDIAN SENSITIVITY ===")
print(f"Dataset A: {dataset_a}")
print(f"  Mean:   {mean_a:.2f}")
print(f"  Median: {median_a:.2f}")
print(f"  Std:    {std_a:.2f}")

print(f"\nDataset B: {dataset_b}")
print(f"  Mean:   {mean_b:.2f}  (+{((mean_b - mean_a) / mean_a) * 100:.1f}%)")
print(f"  Median: {median_b:.2f}  (0.0% change)")
print(f"  Std:    {std_b:.2f}  (+{((std_b - std_a) / std_a) * 100:.1f}%)")

print("\nKey Insight:")
print("-> Adding a single extreme value (500) tripled the mean and exploded the standard deviation,")
print("   yet the median remained completely unaltered at 30.0.")
