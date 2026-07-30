# ==============================================================================
# Program    : Generate Squares Using Generator Expression
# Objective  : Demonstrate memory-efficient generator expressions vs list comprehensions.
# Concept    : Generator Expressions (expr for var in iterable)
# Why Used   : Uses parentheses () to create a generator expression with O(1) memory footprint.
# ==============================================================================

import sys

# What is used : List comprehension [...]
# Why it is used: Demonstrates eager memory allocation
list_squares = [x * x for x in range(10000)]

# What is used : Generator Expression (...)
# Why it is used: Demonstrates lazy evaluation (evaluates values only when requested)
gen_squares = (x * x for x in range(10000))

print("=== Generator Expression vs List Memory Comparison ===")
print(f"Memory size of List (10,000 items)     : {sys.getsizeof(list_squares):,} bytes")
print(f"Memory size of Generator (10,000 items): {sys.getsizeof(gen_squares):,} bytes")

print("\nRetrieving first 5 values from generator via next():")
for _ in range(5):
    print(next(gen_squares), end=" ")
print()
