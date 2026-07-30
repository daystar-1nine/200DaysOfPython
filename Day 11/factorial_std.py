# ==============================================================================
# Program    : Calculate Factorial Using Math Module
# Objective  : Compute factorial of a non-negative integer using math.factorial().
# Concept    : Standard Library Import & Integer Factorial Math
# Why Used   : math.factorial() is faster and memory-optimized compared to manual loops.
# ==============================================================================

# What is used : import math
# Why it is used: Loads built-in mathematical function math.factorial()
import math

# What is used : try-except exception handling block
# Why it is used: Handles non-integer or negative inputs gracefully
try:
    # What is used : int() wrapped around input()
    # Why it is used: Factorials are defined strictly for non-negative whole numbers (integers)
    # How it works : Parses string into integer object
    n = int(input("Enter a non-negative integer: "))

    # What is used : Condition check (n < 0)
    # Why it is used: Prevents negative inputs which are mathematically invalid for factorials
    if n < 0:
        raise ValueError("Factorial is not defined for negative integers!")

    # What is used : math.factorial(n)
    # Why it is used: Computes n! = 1 * 2 * 3 * ... * n
    # How it works : Executes C-optimized integer multiplication loop internally
    fact = math.factorial(n)

    # What is used : f-string with digit grouping format specifier ':, '
    # Why it is used: Formats large numbers with thousand commas (e.g. 120 -> 120, 3628800 -> 3,628,800)
    print(f"Factorial of {n} ({n}!) = {fact:,}")

except ValueError as err:
    # What is used : ValueError exception block
    print(f"Input Error: {err}")
