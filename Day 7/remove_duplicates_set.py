# ==============================================================================
# Program    : Remove Duplicates Using Set
# Objective  : Deduplicate a list of numbers.
# Concept    : Type Casting to set() & Uniqueness
# Why Used   : set() data structure automatically eliminates duplicate elements upon insertion.
# ==============================================================================

# What is used : List with duplicate elements
numbers = [10, 20, 30, 20, 10, 40, 50, 30, 60]
print("Original List with Duplicates:", numbers)

# What is used : set() type casting constructor
# Why it is used: Converts list into set, which internally hashes items and discards duplicates
# How it works : Evaluates uniqueness via hash comparison
unique_set = set(numbers)
print("Unique Set:", unique_set)

# What is used : list() constructor
# Why it is used: Converts deduplicated set back into list format for standard list indexing
unique_list = list(unique_set)
print("Deduplicated List:", unique_list)
