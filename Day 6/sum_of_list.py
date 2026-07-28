# ==============================================================================
# Program    : Find the sum of all numbers in a list
# Objective  : Practice and master find the sum of all numbers in a list logic.
# Concept    : Accumulator variable in loops vs built-in sum()
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Stores ordered, mutable collections of items allowing dynamic modification.
# ==============================================================================

numbers = [10, 20, 30, 40, 50]
print("Original List:", numbers)

# Method 1: Using built-in sum() function
total_builtin = sum(numbers)
print("Sum of numbers (using sum()):", total_builtin)

# Method 2: Manual accumulation loop
total_manual = 0

# What is used : for loop iteration
# Why it is used: Iterates sequentially over elements in an iterable or range sequence
# How it works : Assigns each element to loop variable one by one until exhausted
for num in numbers:
    total_manual += num

print("Sum of numbers (using loop):", total_manual)
