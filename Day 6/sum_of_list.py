# Program: Find the sum of all numbers in a list
# Concept: Accumulator variable in loops vs built-in sum()

numbers = [10, 20, 30, 40, 50]
print("Original List:", numbers)

# Method 1: Using built-in sum() function
total_builtin = sum(numbers)
print("Sum of numbers (using sum()):", total_builtin)

# Method 2: Manual accumulation loop
total_manual = 0
for num in numbers:
    total_manual += num

print("Sum of numbers (using loop):", total_manual)
