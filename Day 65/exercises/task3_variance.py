"""
Task 3 — Population vs Sample Variance
Calculate population variance (ddof=0) and sample variance (ddof=1) using NumPy.
"""

import numpy as np

values = [2, 4, 6, 8, 10]
n = len(values)
mean_val = np.mean(values)

# Step-by-step manual deviations
deviations = [x - mean_val for x in values]
squared_diffs = [d ** 2 for d in deviations]
sum_sq_diffs = sum(squared_diffs)

pop_var_manual = sum_sq_diffs / n
sample_var_manual = sum_sq_diffs / (n - 1)

# NumPy calculations
pop_var_np = np.var(values, ddof=0)
sample_var_np = np.var(values, ddof=1)

print("=== TASK 3: VARIANCE & DEGREES OF FREEDOM ===")
print(f"Values: {values}, Mean: {mean_val}")
print(f"Sum of Squared Deviations: {sum_sq_diffs}")
print(f"Population Variance (ddof=0, N={n}):     {pop_var_np:.4f} (Manual: {pop_var_manual:.4f})")
print(f"Sample Variance     (ddof=1, N-1={n-1}):   {sample_var_np:.4f} (Manual: {sample_var_manual:.4f})")
print("\nWhy do they differ?")
print("-> When calculating sample variance, estimating the population mean consumes 1 degree of freedom.")
print("   Dividing by N underestimates population variance. Bessel's correction (N - 1) yields an unbiased estimator.")
