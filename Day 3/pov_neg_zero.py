# Program: Check if Number is Positive, Negative, or Zero
# Concept: If-elif-else ladder, Comparison Operators (==, <, >)

# Step 1: Input integer number from user
num = int(input("Enter number: "))

# Step 2: Evaluate the sign of the number
if num == 0:
    print("Number is Zero!!")
elif num < 0:
    print("Number is Negative!!")
else:
    print("Number is Positive!!")
