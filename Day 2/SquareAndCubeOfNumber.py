# Program: Calculate Square and Cube of a Number
# Concept: Exponentiation Operator (**), pow() Math Function, and Formatting Output

# Step 1: Input a number from the user
num = float(input("Enter a number: "))

# Step 2: Calculate Square (num ** 2) and Cube (num ** 3 or pow(num, 3))
square = num ** 2
cube = pow(num, 3)

# Step 3: Print the results
print("Number =", num)
print("Square =", square)
print("Cube =", cube)
