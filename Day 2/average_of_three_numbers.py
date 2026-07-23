# Program: Calculate Average of Three Numbers
# Concept: Float Input, Parentheses Precedence, Division Operator, and round() Function

# Step 1: Accept three floating-point numbers from user input
num1 = float(input("Enter first number: "))
num2 = float(input("Enter second number: "))
num3 = float(input("Enter third number: "))

# Step 2: Calculate sum inside parentheses first to maintain operator precedence, then divide by 3
average = (num1 + num2 + num3) / 3

# Step 3: Print the result rounded to 2 decimal places
print("Average =", round(average, 2))