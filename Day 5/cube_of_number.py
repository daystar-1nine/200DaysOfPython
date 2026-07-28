# ==============================================================================
# Program    : Find the cube of a number using a function
# Objective  : Practice and master find the cube of a number using a function logic.
# Concept    : Function return values and calculations
# Why Used   : Encapsulates reusable modular logic to enforce DRY (Don't Repeat Yourself) principle. Pauses execution to capture interactive user input from standard input.
# ==============================================================================

def calculate_cube(number):
    """Returns the cube of a given number"""
    return number ** 3

# Test the function
num = float(input("Enter a number: "))
print(f"Cube of {num} is: {calculate_cube(num)}")
