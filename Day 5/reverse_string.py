# ==============================================================================
# Program    : Reverse a string using a function
# Objective  : Practice and master reverse a string using a function logic.
# Concept    : String slicing and string manipulation inside functions
# Why Used   : Encapsulates reusable modular logic to enforce DRY (Don't Repeat Yourself) principle. Pauses execution to capture interactive user input from standard input.
# ==============================================================================


# What is used : Function definition 'def reverse_string'
# Why it is used: Encapsulates reusable modular logic to enforce DRY principle
# How it works : Defines a named callable block of code that accepts parameters
def reverse_string(text):
    """Returns the reversed version of the input string"""
    return text[::-1]

# Test the function
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
user_str = input("Enter a string: ")
print(f"Reversed string: {reverse_string(user_str)}")
