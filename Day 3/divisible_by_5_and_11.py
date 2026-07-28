# ==============================================================================
# Program    : Check Divisibility by both 5 and 11
# Objective  : Verify if a given number is divisible by 5 and 11 simultaneously.
# Why Used   : Demonstrates Modulus operator (%) to check remainders and Logical 'and' 
#              operator to ensure both conditions are satisfied.
# ==============================================================================

# Step 1: Accept an integer number from user
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
num = int(input("Enter number: "))

# Step 2: Use modulus (%) to check if remainder is 0 when divided by 5 AND 11.
# Both conditions must evaluate to True for the overall if statement to run.
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
if num % 5 == 0 and num % 11 == 0:
    print("Number is divisible by both 5 and 11!!")
else:
    print("Number is not divisible by both 5 and 11!!")
