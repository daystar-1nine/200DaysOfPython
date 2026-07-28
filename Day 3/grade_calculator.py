# ==============================================================================
# Program    : Student Grade Evaluator
# Objective  : Assign letter grades based on student percentage score.
# Why Used   : Demonstrates sequential checking using if-elif-else conditional ladder.
# ==============================================================================

# Step 1: Accept percentage score
marks = float(input("Enter student percentage (0-100): "))

# Step 2: Evaluate grade criteria from highest to lowest boundary
if marks >= 90:
    grade = "A+ (Outstanding)"
elif marks >= 80:
    grade = "A (Excellent)"
elif marks >= 70:
    grade = "B (Very Good)"
elif marks >= 60:
    grade = "C (Good)"
elif marks >= 40:
    grade = "D (Pass)"
else:
    grade = "F (Fail)"

# Step 3: Output resulting grade
print(f"Percentage: {marks}% | Grade: {grade}")
