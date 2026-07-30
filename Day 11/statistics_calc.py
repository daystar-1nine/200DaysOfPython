# ==============================================================================
# Program    : Calculate Average and Summary Using Statistics Module
# Objective  : Compute mean, median, mode, and standard deviation for numeric dataset.
# Concept    : Mathematical Statistics (statistics module)
# Why Used   : statistics module provides exact statistical calculations without manual loop math.
# ==============================================================================

# What is used : import statistics
# Why it is used: Loads statistical functions for mean, median, mode, stdev
import statistics

dataset = [10, 20, 20, 30, 40, 50, 60, 70, 80, 90]
print("Dataset:", dataset)

# What is used : statistics.mean(dataset)
# Why it is used: Computes arithmetic mean (sum / count)
mean_val = statistics.mean(dataset)

# What is used : statistics.median(dataset)
# Why it is used: Computes middle value of sorted dataset
median_val = statistics.median(dataset)

# What is used : statistics.mode(dataset)
# Why it is used: Computes most frequently occurring value in dataset
mode_val = statistics.mode(dataset)

# What is used : statistics.stdev(dataset)
# Why it is used: Computes sample standard deviation
stdev_val = statistics.stdev(dataset)

print(f"\n--- Statistical Summary ---")
print(f"Mean (Average)     : {mean_val:.2f}")
print(f"Median (Middle)    : {median_val:.2f}")
print(f"Mode (Most Frequent): {mode_val}")
print(f"Standard Deviation : {stdev_val:.2f}")
