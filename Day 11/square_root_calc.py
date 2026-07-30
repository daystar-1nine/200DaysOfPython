# ==============================================================================
# Program    : Find Square Root Using Math Module
# Objective  : Calculate square root using Python standard library math.sqrt().
# Concept    : Standard Library Import (import math)
# Why Used   : math.sqrt() uses optimized C-level algorithms for high floating-point precision.
# ==============================================================================

# What is used : Standard library import 'import math'
# Why it is used: Grants access to mathematical functions like math.sqrt()
# How it works : Loads math module namespace into memory
import math

try:
    # What is used : float() wrapped around input()
    # Why it is used: Converts user input to decimal float number
    number = float(input("Enter a non-negative number: "))

    if number < 0:
        # What is used : raise ValueError
        # Why it is used: Square roots of negative numbers yield complex numbers, handled separately
        raise ValueError("Square root of a negative real number is undefined!")

    # What is used : Built-in math.sqrt() function
    # Why it is used: Calculates square root of given number
    # How it works : Computes floating-point square root value sqrt(x)
    result = math.sqrt(number)

    # What is used : f-string formatting with round(result, 4)
    print(f"Square root of {number} = {result:.4f}")

except ValueError as e:
    # What is used : Exception handler for ValueError
    print(f"Error: {e}")
