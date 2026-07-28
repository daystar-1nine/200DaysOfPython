# ==============================================================================
# Program    : Check if a number is even or odd using a function
# Objective  : Practice and master check if a number is even or odd using a function logic.
# Concept    : Functions returning boolean values and modulo checks
# Why Used   : Encapsulates reusable modular logic to enforce DRY (Don't Repeat Yourself) principle. Pauses execution to capture interactive user input from standard input.
# ==============================================================================

def is_even(number):
    """Returns True if number is even, False otherwise"""
    return number % 2 == 0

# Test the function
num = int(input("Enter an integer: "))
if is_even(num):
    print(f"{num} is Even")
else:
    print(f"{num} is Odd")
