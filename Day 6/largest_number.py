# ==============================================================================
# Program    : Find Largest Number in a List
# Objective  : Discover maximum value in a list of numbers.
# Concept    : Linear Iteration vs Built-in max()
# Why Used   : Compares built-in max() function O(n) against manual loop comparison.
# ==============================================================================

# What is used : List collection data type [...]
# Why it is used: Stores multiple numeric values in single ordered variable
numbers = [23, 89, 12, 56, 99, 45, 78]
print("Original List:", numbers)

# What is used : Built-in max() function
# Why it is used: Automatically iterates list and returns highest element
# How it works : Performs single pass linear search comparing elements
largest_builtin = max(numbers)
print("Largest number (using max()):", largest_builtin)

# What is used : Manual loop initialization 'largest_manual = numbers[0]'
# Why it is used: Sets baseline assumed maximum to first element of list
largest_manual = numbers[0]

# What is used : for loop iterating through list items
# How it works : Inspects each element 'num' sequentially
for num in numbers:
    # What is used : Comparison operator (>)
    # How it works : Updates largest_manual if current number is greater
    if num > largest_manual:
        largest_manual = num

print("Largest number (using loop):", largest_manual)
