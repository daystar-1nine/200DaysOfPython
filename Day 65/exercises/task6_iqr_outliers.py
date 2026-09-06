"""
Task 6 — Interquartile Range (IQR) & Outlier Fences
Calculate Q1, Q3, IQR, Tukey fences, and detect potential outliers.
"""

import numpy as np

values = np.array([10, 12, 14, 15, 16, 18, 19, 21, 22, 25, 55, -15])

q1 = float(np.percentile(values, 25))
q3 = float(np.percentile(values, 75))
iqr = q3 - q1

lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

outliers = values[(values < lower_fence) | (values > upper_fence)]
inliers = values[(values >= lower_fence) & (values <= upper_fence)]

print("=== TASK 6: IQR & OUTLIER BOUNDARIES ===")
print(f"Dataset: {values.tolist()}")
print(f"Q1 (25th percentile): {q1:.2f}")
print(f"Q3 (75th percentile): {q3:.2f}")
print(f"IQR (Q3 - Q1):        {iqr:.2f}")
print(f"Lower Fence (Q1 - 1.5*IQR): {lower_fence:.2f}")
print(f"Upper Fence (Q3 + 1.5*IQR): {upper_fence:.2f}")
print(f"Potential Outliers Flagged: {outliers.tolist()}")
print(f"Inliers Count: {len(inliers)} / {len(values)}")
