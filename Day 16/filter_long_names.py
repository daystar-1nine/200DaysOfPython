# ==============================================================================
# Program    : Filter Names Longer Than 5 Letters Using filter()
# Objective  : Retain only string elements with length strictly greater than 5 characters.
# Concept    : String Predicate Filtering (len(name) > 5)
# Why Used   : Applies len() condition inside filter() lambda.
# ==============================================================================

names = ["Guido", "Alexander", "Suraj", "Christopher", "Ana", "Elizabeth"]
print("Original Names List:", names)

# What is used : filter() with lambda checking length 'lambda name: len(name) > 5'
# Why it is used: Filters out names with 5 or fewer characters
# How it works : Returns True only for names where len(name) > 5
long_names = list(filter(lambda name: len(name) > 5, names))

print("\n--- Filtered Long Names (> 5 letters) ---")
print("Long Names List:", long_names)
