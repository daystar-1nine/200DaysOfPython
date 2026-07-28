# ==============================================================================
# Program    : Find the square of a number using a function
# Objective  : Practice and master find the square of a number using a function logic.
# Concept    : Functions with return statements and exponentiation
# Why Used   : Encapsulates reusable modular logic to enforce DRY (Don't Repeat Yourself) principle. Pauses execution to capture interactive user input from standard input.
# ==============================================================================

def calculate_square(number):
    """Returns the square of a given number"""
    return number ** 2

# Test the function
num = float(input("Enter a number: "))
print(f"Square of {num} is: {calculate_square(num)}")
