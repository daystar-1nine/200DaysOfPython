# ==============================================================================
# Program    : Find Square Root Using Math Module
# Objective  : Calculate square root using Python standard library math.sqrt().
# Concept    : Standard Library Import & Floating-Point Mathematics
# Why Used   : math.sqrt() uses C-level hardware instructions for maximum floating-point precision.
# ==============================================================================

# What is used : Standard library import 'import math'
# Why it is used: Grants access to high-performance math functions like math.sqrt()
# How it works : Loads math module namespace into process memory
import math

# What is used : try-except exception handling block
# Why it is used: Catches invalid user inputs or negative numbers safely
try:
    # What is used : float() wrapped around input()
    # Why it is used: Converts user input string into decimal float number
    # How it works : Reads string from stdin and parses into float object
    number = float(input("Enter a non-negative number: "))

    # What is used : Relational comparison operator (< 0) with raise keyword
    # Why it is used: Real square roots are mathematically undefined for negative numbers
    if number < 0:
        raise ValueError("Square root of a negative real number is undefined!")

    # What is used : Built-in math.sqrt(number) function
    # Why it is used: Calculates square root of given number
    # How it works : Computes floating-point square root value sqrt(x) using C-math library
    result = math.sqrt(number)

    # What is used : f-string formatting with specifier ':.4f'
    # Why it is used: Formats output float to 4 decimal places for clean presentation
    print(f"Square root of {number} = {result:.4f}")

except ValueError as e:
    # What is used : Exception handler for ValueError
    # How it works : Catches and displays user-friendly error message
    print(f"Input Error: {e}")
