"""
Task 7 — Z-Score Standardization & Outlier Screening
Compute Z-scores and filter observations exceeding threshold bounds.
"""

import numpy as np

values = np.array([45, 50, 55, 60, 65, 70, 75, 80, 85, 150])

mean_val = np.mean(values)
std_val = np.std(values, ddof=1)  # sample std

z_scores = (values - mean_val) / std_val

outliers_z2 = values[np.abs(z_scores) > 2.0]
outliers_z3 = values[np.abs(z_scores) > 3.0]

print("=== TASK 7: Z-SCORES & EXTREME VALUES ===")
print(f"Mean: {mean_val:.2f}, Sample Std: {std_val:.2f}")
print("-" * 50)
print(f"{'Value':<10}{'Z-Score':<15}{'Flag (|z| > 2)':<15}{'Flag (|z| > 3)':<15}")
print("-" * 50)
for v, z in zip(values, z_scores):
    flag2 = "YES" if abs(z) > 2.0 else "NO"
    flag3 = "YES" if abs(z) > 3.0 else "NO"
    print(f"{v:<10}{z:<15.4f}{flag2:<15}{flag3:<15}")
print("-" * 50)
print(f"Observations with |z| > 2: {outliers_z2.tolist()}")
print(f"Observations with |z| > 3: {outliers_z3.tolist()}")
