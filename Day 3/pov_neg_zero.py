# ==============================================================================
# Program    : Number Sign Checker (Positive, Negative, or Zero)
# Objective  : Classify a number based on its relationship to 0.
# Why Used   : Demonstrates relational comparison operators (>, <, ==).
# ==============================================================================

# Step 1: Accept number input
num = float(input("Enter a number: "))

# Step 2: Classify sign
if num > 0:
    print("The number is POSITIVE.")
elif num < 0:
    print("The number is NEGATIVE.")
else:
    print("The number is ZERO.")
