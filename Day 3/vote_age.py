# ==============================================================================
# Program    : Voter Eligibility Checker
# Objective  : Verify if a person meets the legal voting age limit (18+).
# Why Used   : Uses relational operator (>=) to test eligibility threshold.
# ==============================================================================

# Step 1: Input age
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
age = int(input("Enter your age: "))

# Step 2: Evaluate age condition
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
if age >= 18:
    print("Eligible to Vote!")
else:
    print(f"Not Eligible to Vote. Please wait {18 - age} more year(s).")
