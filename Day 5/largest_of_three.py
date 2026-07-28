# ==============================================================================
# Program    : Find the largest of three numbers using a function
# Objective  : Practice and master find the largest of three numbers using a function logic.
# Concept    : Multi-parameter functions and comparison logic
# Why Used   : Encapsulates reusable modular logic to enforce DRY (Don't Repeat Yourself) principle. Pauses execution to capture interactive user input from standard input.
# ==============================================================================


# What is used : Function definition 'def find_largest'
# Why it is used: Encapsulates reusable modular logic to enforce DRY principle
# How it works : Defines a named callable block of code that accepts parameters
def find_largest(a, b, c):
    """Returns the maximum of three numbers"""
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
    if a >= b and a >= c:
        return a
# What is used : Conditional statement (elif)
# Why it is used: Evaluates boolean condition to control branching execution flow
    elif b >= a and b >= c:
        return b
    else:
        return c

# Test the function
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
x = float(input("Enter first number: "))
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
y = float(input("Enter second number: "))
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
z = float(input("Enter third number: "))
print(f"The largest number is: {find_largest(x, y, z)}")
