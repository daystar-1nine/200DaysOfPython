# ==============================================================================
# Program    : Find the cube of a number using a function
# Objective  : Practice and master find the cube of a number using a function logic.
# Concept    : Function return values and calculations
# Why Used   : Encapsulates reusable modular logic to enforce DRY (Don't Repeat Yourself) principle. Pauses execution to capture interactive user input from standard input.
# ==============================================================================


# What is used : Function definition 'def calculate_cube'
# Why it is used: Encapsulates reusable modular logic to enforce DRY principle
# How it works : Defines a named callable block of code that accepts parameters
def calculate_cube(number):
    """Returns the cube of a given number"""
    return number ** 3

# Test the function
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
num = float(input("Enter a number: "))
print(f"Cube of {num} is: {calculate_cube(num)}")
