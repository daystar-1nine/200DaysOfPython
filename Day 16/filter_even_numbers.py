# ==============================================================================
# Program    : Filter Even Numbers Using filter() and Lambda
# Objective  : Filter even integers from a list using filter().
# Concept    : Predicate Filtering via filter()
# Why Used   : filter() keeps only items for which lambda predicate evaluates to True (x % 2 == 0).
# ==============================================================================

numbers = [12, 7, 19, 24, 30, 5, 14, 21, 8, 3]
print("Original Numbers List:", numbers)

# What is used : filter() with lambda predicate 'lambda x: x % 2 == 0'
# Why it is used: Filters out odd numbers and retains even numbers
# How it works : Tests each element; if x % 2 == 0 is True, element is retained
even_numbers = list(filter(lambda x: x % 2 == 0, numbers))

print("\n--- Filtered Even Numbers Output ---")
print("Even Numbers List:", even_numbers)
