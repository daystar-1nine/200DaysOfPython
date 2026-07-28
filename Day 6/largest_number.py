# ==============================================================================
# Program    : Find the largest number in a list
# Objective  : Practice and master find the largest number in a list logic.
# Concept    : Iterating over list items to track maximum value vs built-in max()
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Stores ordered, mutable collections of items allowing dynamic modification.
# ==============================================================================

numbers = [23, 89, 12, 56, 99, 45, 78]
print("Original List:", numbers)

# Method 1: Using built-in max() function
largest_builtin = max(numbers)
print("Largest number (using max()):", largest_builtin)

# Method 2: Manual iteration using a loop
largest_manual = numbers[0]
for num in numbers:
    if num > largest_manual:
        largest_manual = num

print("Largest number (using loop):", largest_manual)
