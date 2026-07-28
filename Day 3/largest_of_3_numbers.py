# ==============================================================================
# Program    : Find Largest of Three Numbers
# Objective  : Determine the maximum value among three user inputs.
# Why Used   : Uses comparison operators (>=) combined with logical 'and' operators.
# ==============================================================================

# Step 1: Input 3 numbers
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
num1 = float(input("Enter first number: "))
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
num2 = float(input("Enter second number: "))
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
num3 = float(input("Enter third number: "))

# Step 2: Compare numbers using logical AND
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
if num1 >= num2 and num1 >= num3:
    largest = num1
# What is used : Conditional statement (elif)
# Why it is used: Evaluates boolean condition to control branching execution flow
elif num2 >= num1 and num2 >= num3:
    largest = num2
else:
    largest = num3

# Step 3: Print largest number
print(f"The largest number among {num1}, {num2}, and {num3} is: {largest}")
