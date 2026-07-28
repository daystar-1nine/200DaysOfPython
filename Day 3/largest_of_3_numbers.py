# ==============================================================================
# Program    : Find Largest of Three Numbers
# Objective  : Determine the maximum value among three user inputs.
# Why Used   : Uses comparison operators (>=) combined with logical 'and' operators.
# ==============================================================================

# Step 1: Input 3 numbers
num1 = float(input("Enter first number: "))
num2 = float(input("Enter second number: "))
num3 = float(input("Enter third number: "))

# Step 2: Compare numbers using logical AND
if num1 >= num2 and num1 >= num3:
    largest = num1
elif num2 >= num1 and num2 >= num3:
    largest = num2
else:
    largest = num3

# Step 3: Print largest number
print(f"The largest number among {num1}, {num2}, and {num3} is: {largest}")
