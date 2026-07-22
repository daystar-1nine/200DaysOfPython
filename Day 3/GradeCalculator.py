# Program: Student Grade Calculator
# Concept: Logical 'or' for input validation, If-elif-else ladder for grading thresholds

# Step 1: Input marks as a float
marks = float(input("Enter your marks: "))

# Step 2: Validate if marks are within the valid 0-100 range
if marks < 0 or marks > 100:
    print("Invalid marks!!")
# Step 3: Categorize grades sequentially from highest to lowest
elif marks >= 90:
    print("Grade A+")
elif marks >= 80:
    print("Grade A")
elif marks >= 70:
    print("Grade B")
elif marks >= 60:
    print("Grade C")
elif marks >= 40:
    print("Grade D")
# Step 4: If marks are below 40, student fails
else:
    print("Grade F - Fail!!")