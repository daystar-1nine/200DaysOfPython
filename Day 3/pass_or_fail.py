# ==============================================================================
# Program    : Student Pass or Fail Evaluator
# Objective  : Determine pass/fail status based on 40% threshold.
# Why Used   : Demonstrates basic binary conditional branching (if-else).
# ==============================================================================

# Step 1: Accept marks score
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
marks = float(input("Enter marks (out of 100): "))

# Step 2: Evaluate pass/fail boundary
# What is used : Conditional statement (if)
# Why it is used: Evaluates boolean condition to control branching execution flow
if marks >= 40:
    print("Result: PASS [PASS]")
else:
    print("Result: FAIL [FAIL]")
