# ==============================================================================
# Program    : Find the largest of three numbers using a function
# Objective  : Practice and master find the largest of three numbers using a function logic.
# Concept    : Multi-parameter functions and comparison logic
# Why Used   : Encapsulates reusable modular logic to enforce DRY (Don't Repeat Yourself) principle. Pauses execution to capture interactive user input from standard input.
# ==============================================================================

def find_largest(a, b, c):
    """Returns the maximum of three numbers"""
    if a >= b and a >= c:
        return a
    elif b >= a and b >= c:
        return b
    else:
        return c

# Test the function
x = float(input("Enter first number: "))
y = float(input("Enter second number: "))
z = float(input("Enter third number: "))
print(f"The largest number is: {find_largest(x, y, z)}")
