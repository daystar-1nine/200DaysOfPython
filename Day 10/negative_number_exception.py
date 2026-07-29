# ==============================================================================
# Program    : Raise Exception for Negative Numbers
# Objective  : Enforce positive number constraints by explicitly raising exceptions.
# Concept    : Explicit Exception Raising (raise ValueError)
# Why Used   : 'raise' allows developers to enforce business validation logic explicitly.
# ==============================================================================

# What is used : Function definition 'def check_positive(num)'
# Why it is used: Encapsulates validation logic into reusable function
def check_positive(num):
    # What is used : Comparison operator (< 0) with raise keyword
    # Why it is used: Triggers ValueError if input violates non-negative rule
    # How it works : Instantiates and raises ValueError with custom error string message
    if num < 0:
        raise ValueError("Invalid Input: Negative numbers are strictly not allowed!")
    return f"Valid Positive Number: {num}"

try:
    number = float(input("Enter a positive number: "))
    # What is used : Function invocation
    result = check_positive(number)
    print(result)

except ValueError as e:
    # What is used : Catching raised exception object 'as e'
    # Why it is used: Captures and displays message passed into raise statement
    print(e)
