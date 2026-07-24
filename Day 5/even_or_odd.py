# Program: Check if a number is even or odd using a function
# Concept: Functions returning boolean values and modulo checks

def is_even(number):
    """Returns True if number is even, False otherwise"""
    return number % 2 == 0

# Test the function
num = int(input("Enter an integer: "))
if is_even(num):
    print(f"{num} is Even")
else:
    print(f"{num} is Odd")
