# ==============================================================================
# Program    : Voter Eligibility Checker
# Objective  : Verify if a person meets the legal voting age limit (18+).
# Why Used   : Uses relational operator (>=) to test eligibility threshold.
# ==============================================================================

# Step 1: Input age
age = int(input("Enter your age: "))

# Step 2: Evaluate age condition
if age >= 18:
    print("Eligible to Vote!")
else:
    print(f"Not Eligible to Vote. Please wait {18 - age} more year(s).")
