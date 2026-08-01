# ==============================================================================
# Program    : Create Squares Using List Comprehension
# Objective  : Generate squares of numbers using List Comprehension with condition.
# Concept    : List Comprehensions ([expr for var in iterable if condition])
# Why Used   : Provides declarative, readable, and fast list creation syntax.
# ==============================================================================

# What is used : List comprehension generating squares for range(1, 11)
# Why it is used: Replaces traditional loop + append() with inline syntax
squares_all = [i * i for i in range(1, 11)]

# What is used : List comprehension with condition 'if i % 2 == 0'
# Why it is used: Squares only even numbers in range
squares_even = [i * i for i in range(1, 11) if i % 2 == 0]

print("=== List Comprehensions Output ===")
print("Squares of all numbers (1-10) :", squares_all)
print("Squares of even numbers (1-10):", squares_even)
