# ==============================================================================
# Program    : Calculate Product Using reduce()
# Objective  : Aggregate numbers in a list into a single product value using functools.reduce().
# Concept    : Sequence Aggregation via reduce()
# Why Used   : reduce() applies a 2-argument accumulator lambda across all sequence elements.
# ==============================================================================

# What is used : Importing reduce from functools standard library
from functools import reduce

numbers = [1, 2, 3, 4, 5]
print("Numbers List:", numbers)

# What is used : reduce() with accumulator lambda 'lambda acc, x: acc * x'
# Why it is used: Multiplies sequence elements cumulatively ((((1 * 2) * 3) * 4) * 5)
# How it works : Maintains running accumulator acc and multiplies next element x
total_product = reduce(lambda acc, x: acc * x, numbers)

print("\n--- Calculated Total Product ---")
print(f"Product of {numbers} = {total_product}")
