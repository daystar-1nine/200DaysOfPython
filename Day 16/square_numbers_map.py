# ==============================================================================
# Program    : Square Numbers Using map() and Lambda
# Objective  : Compute squares of all numbers in a list using map() and lambda.
# Concept    : Higher-Order Functions (map) & Anonymous Functions (lambda)
# Why Used   : map() applies lambda x: x * x to every element in the list lazily.
# ==============================================================================

# What is used : Python list of integers
# Why it is used: Serves as input dataset sequence
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print("Original Numbers List:", numbers)

# What is used : map() with lambda function 'lambda x: x * x'
# Why it is used: Transforms every number into its square value
# How it works : Applies lambda function to each element in numbers list sequentially
squares_map = map(lambda x: x * x, numbers)

# What is used : list() constructor
# Why it is used: Converts map iterator object into a concrete Python list
squares_list = list(squares_map)

print("\n--- Squared Numbers Output (via map & lambda) ---")
print("Resulting Squares List:", squares_list)
