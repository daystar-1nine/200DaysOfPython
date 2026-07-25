# Program: Reverse a list
# Concept: List slicing [::-1] vs .reverse() method vs reversed()

numbers = [1, 2, 3, 4, 5, 6, 7]
print("Original List:", numbers)

# Method 1: Using list slicing [::-1] (Creates a new list)
reversed_slice = numbers[::-1]
print("Reversed (Slicing):", reversed_slice)

# Method 2: Using .reverse() method (Modifies list in-place)
numbers_copy = numbers.copy()
numbers_copy.reverse()
print("Reversed (.reverse()):", numbers_copy)
