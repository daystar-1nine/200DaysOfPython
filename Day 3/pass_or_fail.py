# ==============================================================================
# Program    : Student Pass or Fail Evaluator
# Objective  : Determine pass/fail status based on 40% threshold.
# Why Used   : Demonstrates basic binary conditional branching (if-else).
# ==============================================================================

# Step 1: Accept marks score
marks = float(input("Enter marks (out of 100): "))

# Step 2: Evaluate pass/fail boundary
if marks >= 40:
    print("Result: PASS [PASS]")
else:
    print("Result: FAIL [FAIL]")
