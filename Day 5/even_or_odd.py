# ==============================================================================
# Program    : Check if a number is even or odd using a function
# Objective  : Practice and master check if a number is even or odd using a function logic.
# Concept    : Functions returning boolean values and modulo checks
# Why Used   : Encapsulates reusable modular logic to enforce DRY (Don't Repeat Yourself) principle. Pauses execution to capture interactive user input from standard input.
# ==============================================================================


# What is used : Function definition 'def is_even'
# Why it is used: Encapsulates reusable modular logic to enforce DRY principle
# How it works : Defines a named callable block of code that accepts parameters
def is_even(number):
    """Returns True if number is even, False otherwise"""
    return number % 2 == 0

# Test the function
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
num = int(input("Enter an integer: "))
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
if is_even(num):
    print(f"{num} is Even")
else:
    print(f"{num} is Odd")
