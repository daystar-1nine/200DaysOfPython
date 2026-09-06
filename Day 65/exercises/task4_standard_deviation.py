"""
Task 4 — Standard Deviation
Calculate population vs sample standard deviation and compare with Pandas .std().
"""

import numpy as np
import pandas as pd

values = [15, 20, 25, 30, 35, 40]

pop_std = np.std(values, ddof=0)
sample_std = np.std(values, ddof=1)

series = pd.Series(values)
pd_std = series.std()

print("=== TASK 4: STANDARD DEVIATION ===")
print(f"Dataset: {values}")
print(f"Population Std (ddof=0): {pop_std:.4f}")
print(f"Sample Std     (ddof=1): {sample_std:.4f}")
print(f"Pandas .std()  (default): {pd_std:.4f}")
print(f"Does Pandas match sample standard deviation? -> {np.isclose(sample_std, pd_std)}")
print("Explanation: Pandas assumes tabular data is almost always a sample of a wider population,")
print("hence Series.std() defaults to ddof=1.")
