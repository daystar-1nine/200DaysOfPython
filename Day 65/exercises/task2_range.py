"""
Task 2 — Range, Minimum and Maximum
Compute the spread of observations without using DataFrame.describe().
"""

import numpy as np

values = [14, 28, 42, 56, 70, 84, 98, 112]

# Manual Loop
val_min = values[0]
val_max = values[0]
for v in values[1:]:
    if v < val_min:
        val_min = v
    if v > val_max:
        val_max = v

manual_range = val_max - val_min

# NumPy Implementation
np_min = int(np.min(values))
np_max = int(np.max(values))
np_range = np_max - np_min

print("=== TASK 2: RANGE, MINIMUM & MAXIMUM ===")
print(f"Dataset: {values}")
print(f"Minimum: {np_min}")
print(f"Maximum: {np_max}")
print(f"Range:   {np_range} (Max - Min)")
print(f"Verification: manual_range == np_range -> {manual_range == np_range}")
