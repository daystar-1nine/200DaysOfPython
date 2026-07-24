# Program: Find the cube of a number using a function
# Concept: Function return values and calculations

def calculate_cube(number):
    """Returns the cube of a given number"""
    return number ** 3

# Test the function
num = float(input("Enter a number: "))
print(f"Cube of {num} is: {calculate_cube(num)}")
