# ==============================================================================
# Program    : Even or Odd Number Checker
# Objective  : Determine whether a number is even, odd, or zero.
# Why Used   : Uses Modulus operator (% 2) to test divisibility by 2.
# ==============================================================================

# Step 1: Accept integer input
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
num = int(input("Enter a number: "))

# Step 2: Check if number is 0, even, or odd using if-elif-else ladder
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
if num == 0:
    print("The number is Zero.")
# What is used : Conditional statement (elif)
# Why it is used: Evaluates boolean condition to control branching execution flow
elif num % 2 == 0:
    print(f"{num} is an Even number.")
else:
    print(f"{num} is an Odd number.")
