# ==============================================================================
# Program    : Number Sign Checker (Positive, Negative, or Zero)
# Objective  : Classify a number based on its relationship to 0.
# Why Used   : Demonstrates relational comparison operators (>, <, ==).
# ==============================================================================

# Step 1: Accept number input
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
num = float(input("Enter a number: "))

# Step 2: Classify sign
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
if num > 0:
    print("The number is POSITIVE.")
# What is used : Conditional statement (elif)
# Why it is used: Evaluates boolean condition to control branching execution flow
elif num < 0:
    print("The number is NEGATIVE.")
else:
    print("The number is ZERO.")
