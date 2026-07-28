# ==============================================================================
# Program    : Leap Year Checker
# Objective  : Determine if a year is a leap year based on Gregorian calendar rules.
# Why Used   : Combines modulo operations: (Year % 4 == 0 and Year % 100 != 0) or (Year % 400 == 0).
# ==============================================================================

# Step 1: Input year
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
year = int(input("Enter year: "))

# Step 2: Evaluate leap year rules
# A year is leap if divisible by 4 AND not divisible by 100, OR if divisible by 400.
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
    print(f"{year} is a Leap Year!")
else:
    print(f"{year} is NOT a Leap Year.")
