# ==============================================================================
# Program    : Handle Division by Zero Exception
# Objective  : Safely compute division without crashing on 0 denominator.
# Concept    : Exception Handling (try-except ZeroDivisionError)
# Why Used   : ZeroDivisionError occurs when dividing by zero; try-except catches it gracefully.
# ==============================================================================

# What is used : try-except block
# Why it is used: Intercepts ZeroDivisionError during division operation
# How it works : Monitors code in try block; redirects execution to except block if division by zero occurs
try:
    # What is used : float() wrapped around input()
    # Why it is used: Converts user string input into float numbers
    numerator = float(input("Enter numerator: "))
    denominator = float(input("Enter denominator: "))

    # What is used : Division operator (/)
    # How it works : Computes quotient; raises ZeroDivisionError if denominator is 0.0
    result = numerator / denominator

    # What is used : print() with f-string
    print(f"Result: {numerator} / {denominator} = {result:.2f}")

except ZeroDivisionError:
    # What is used : Specific exception handler for ZeroDivisionError
    # Why it is used: Displays user-friendly error message instead of crashing program
    print("Error: Division by zero is not allowed!")
