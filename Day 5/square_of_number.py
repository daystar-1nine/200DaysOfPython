# Program: Find the square of a number using a function
# Concept: Functions with return statements and exponentiation

def calculate_square(number):
    """Returns the square of a given number"""
    return number ** 2

# Test the function
num = float(input("Enter a number: "))
print(f"Square of {num} is: {calculate_square(num)}")
