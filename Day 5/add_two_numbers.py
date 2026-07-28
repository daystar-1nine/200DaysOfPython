# ==============================================================================
# Program    : Add Two Numbers Using a Function
# Objective  : Perform addition using modular functions.
# Concept    : Function Parameters, Arguments & Return Values
# Why Used   : Encapsulates reusable mathematical logic into a modular function block.
# ==============================================================================

# What is used : Function definition keyword 'def' with parameters (num1, num2)
# Why it is used: Defining add_numbers(num1, num2) allows adding any 2 numbers anywhere in program
# How it works : Accepts 2 arguments when called and binds them to local variables num1 and num2
def add_numbers(num1, num2):
    """Returns the sum of two numbers"""
    # What is used : return statement
    # Why it is used: Sends calculated result back to function caller and exits function
    return num1 + num2

# What is used : float() wrapped around input()
# Why it is used: Converts user string input into float number to support decimals
x = float(input("Enter first number: "))
y = float(input("Enter second number: "))

# What is used : Function invocation 'add_numbers(x, y)'
# How it works : Passes values of x and y as arguments to function, receiving sum in 'result'
result = add_numbers(x, y)

# What is used : f-string formatting
print(f"The sum of {x} and {y} is: {result}")
