# ==============================================================================
# Program    : Calculate Factorial Using Math Module
# Objective  : Compute factorial of a non-negative integer using math.factorial().
# Concept    : Standard Library Import & Integer Product Math
# Why Used   : math.factorial() is faster and memory-optimized compared to recursive loops.
# ==============================================================================

# What is used : import math
# Why it is used: Loads built-in mathematical function math.factorial()
import math

try:
    # What is used : int() wrapped around input()
    # Why it is used: Factorials are defined strictly for non-negative whole numbers (integers)
    n = int(input("Enter a non-negative integer: "))

    if n < 0:
        raise ValueError("Factorial is not defined for negative integers!")

    # What is used : math.factorial(n)
    # Why it is used: Computes n! = 1 * 2 * 3 * ... * n
    # How it works : Executes C-optimized multiplication loop
    fact = math.factorial(n)
    print(f"Factorial of {n} ({n}!) = {fact:,}")

except ValueError as err:
    # What is used : ValueError exception block
    print(f"Input Error: {err}")
