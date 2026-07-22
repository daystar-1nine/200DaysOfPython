# Program: Find the Largest of Three Numbers
# Concept: Logical 'and' Operator, Comparison Operators, and sequential check logic

# Step 1: Input three integer numbers
a = int(input("Enter first number: "))
b = int(input("Enter second number: "))
c = int(input("Enter third number: "))

# Step 2: Check if all numbers are equal
if a == b and a == c:
    print("All numbers are equal!!")
# Step 3: Compare a with b and c
elif a >= b and a >= c:
    print("First number is largest!!")
# Step 4: Compare b with a and c
elif b >= a and b >= c:
    print("Second number is largest!!")
# Step 5: If neither a nor b is largest, c must be the largest
else:
    print("Third number is largest!!")