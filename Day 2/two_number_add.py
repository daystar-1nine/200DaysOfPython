# ==============================================================================
# Program    : Addition of Two Numbers
# Objective  : Calculate sum of two numbers provided by the user.
# Why Used   : Demonstrates user input conversion via float() and basic arithmetic (+).
# ==============================================================================

# Step 1: Accept two numbers from user
# float() is used to support both decimal numbers and integers
num1 = float(input("Enter first number: "))
num2 = float(input("Enter second number: "))

# Step 2: Compute addition
sum_result = num1 + num2

# Step 3: Display the result
print(f"\nThe sum of {num1} and {num2} is: {sum_result}")
