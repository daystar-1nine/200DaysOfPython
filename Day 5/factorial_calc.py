# ==============================================================================
# Program    : Calculate factorial of a number using a function
# Objective  : Practice and master calculate factorial of a number using a function logic.
# Concept    : Loop accumulation within a function
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Encapsulates reusable modular logic to enforce DRY (Don't Repeat Yourself) principle.
# ==============================================================================


# What is used : Function definition 'def calculate_factorial'
# Why it is used: Encapsulates reusable modular logic to enforce DRY principle
# How it works : Defines a named callable block of code that accepts parameters
def calculate_factorial(n):
    """Returns the factorial of a non-negative integer n"""
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
    if n < 0:
        return "Invalid input (Negative number)"
    fact = 1

# What is used : for loop iteration
# Why it is used: Iterates sequentially over elements in an iterable or range sequence
# How it works : Assigns each element to loop variable one by one until exhausted
    for i in range(1, n + 1):
        fact *= i
    return fact

# Test the function
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
num = int(input("Enter a positive integer: "))
print(f"Factorial of {num} is: {calculate_factorial(num)}")
