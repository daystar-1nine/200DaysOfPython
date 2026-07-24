# Program: Add two numbers using a function
# Concept: Function parameters, arguments, and return values

def add_numbers(num1, num2):
    """Returns the sum of two numbers"""
    return num1 + num2

# Test the function
x = float(input("Enter first number: "))
y = float(input("Enter second number: "))
result = add_numbers(x, y)
print(f"The sum of {x} and {y} is: {result}")
