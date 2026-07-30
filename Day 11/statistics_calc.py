# ==============================================================================
# Program    : Calculate Average and Summary Using Statistics Module
# Objective  : Compute mean, median, mode, and standard deviation for numeric dataset.
# Concept    : Mathematical Statistics (statistics module)
# Why Used   : statistics module provides exact statistical summaries without writing manual loop math.
# ==============================================================================

# What is used : import statistics
# Why it is used: Loads statistical functions for dataset analysis
import statistics

dataset = [10, 20, 20, 30, 40, 50, 60, 70, 80, 90]
print("Dataset:", dataset)

# What is used : statistics.mean(dataset)
# Why it is used: Computes arithmetic average (sum of elements / total count)
# How it works : Sums list values and divides by len(dataset)
mean_val = statistics.mean(dataset)

# What is used : statistics.median(dataset)
# Why it is used: Computes middle value of sorted dataset
# How it works : Sorts values and picks middle element (or average of two middle elements)
median_val = statistics.median(dataset)

# What is used : statistics.mode(dataset)
# Why it is used: Computes most frequently occurring value in dataset
# How it works : Counts element frequencies and returns key with maximum count
mode_val = statistics.mode(dataset)

# What is used : statistics.stdev(dataset)
# Why it is used: Computes sample standard deviation (variance square root)
stdev_val = statistics.stdev(dataset)

print("\n--- Statistical Summary ---")
print(f"Mean (Average)     : {mean_val:.2f}")
print(f"Median (Middle)    : {median_val:.2f}")
print(f"Mode (Most Frequent): {mode_val}")
print(f"Standard Deviation : {stdev_val:.2f}")
