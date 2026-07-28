# ==============================================================================
# Program    : Student Grade Evaluator
# Objective  : Assign letter grades based on student percentage score.
# Why Used   : Demonstrates sequential checking using if-elif-else conditional ladder.
# ==============================================================================

# Step 1: Accept percentage score
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
marks = float(input("Enter student percentage (0-100): "))

# Step 2: Evaluate grade criteria from highest to lowest boundary
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
if marks >= 90:
    grade = "A+ (Outstanding)"
# What is used : Conditional statement (elif)
# Why it is used: Evaluates boolean condition to control branching execution flow
elif marks >= 80:
    grade = "A (Excellent)"
# What is used : Conditional statement (elif)
# Why it is used: Evaluates boolean condition to control branching execution flow
elif marks >= 70:
    grade = "B (Very Good)"
# What is used : Conditional statement (elif)
# Why it is used: Evaluates boolean condition to control branching execution flow
elif marks >= 60:
    grade = "C (Good)"
# What is used : Conditional statement (elif)
# Why it is used: Evaluates boolean condition to control branching execution flow
elif marks >= 40:
    grade = "D (Pass)"
else:
    grade = "F (Fail)"

# Step 3: Output resulting grade
print(f"Percentage: {marks}% | Grade: {grade}")
