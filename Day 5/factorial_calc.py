# Program: Calculate factorial of a number using a function
# Concept: Loop accumulation within a function

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
