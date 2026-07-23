# Program: Vote Eligibility Checker
# Concept: Boundary Validation, Threshold comparison (age >= 18)

# Step 1: Input age from user
age = int(input("Enter age: "))

# Step 2: Validate if age is non-positive (invalid age)
if age <= 0:
    print("Invalid age!!")
# Step 3: Check eligibility based on the voting age limit of 18
elif age < 18:
    print("Not eligible to vote!!")
else:
    print("Eligible to vote!!")