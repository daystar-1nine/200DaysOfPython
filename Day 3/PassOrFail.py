# Program: Pass or Fail Checker
# Concept: Logical 'or' for boundaries, basic conditional statements

# Step 1: Input marks
marks = float(input("Enter your marks: "))

# Step 2: Validate the input range
if marks < 0 or marks > 100:
    print("Invalid marks!!")
# Step 3: Check if marks satisfy the passing criteria (40 or above)
elif marks >= 40:
    print("Pass!!")
else:
    print("Fail!!")