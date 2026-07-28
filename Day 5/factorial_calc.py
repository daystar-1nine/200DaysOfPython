# ==============================================================================
# Program    : Calculate factorial of a number using a function
# Objective  : Practice and master calculate factorial of a number using a function logic.
# Concept    : Loop accumulation within a function
# Why Used   : Repeats execution for a known sequence or range of items efficiently. Encapsulates reusable modular logic to enforce DRY (Don't Repeat Yourself) principle.
# ==============================================================================

def calculate_factorial(n):
    """Returns the factorial of a non-negative integer n"""
    if n < 0:
        return "Invalid input (Negative number)"
    fact = 1
    for i in range(1, n + 1):
        fact *= i
    return fact

# Test the function
num = int(input("Enter a positive integer: "))
print(f"Factorial of {num} is: {calculate_factorial(num)}")
