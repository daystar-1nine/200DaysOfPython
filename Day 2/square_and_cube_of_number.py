# ==============================================================================
# Program    : Calculate Square and Cube of a Number
# Objective  : Compute exponentiation values for any numeric input.
# Why Used   : Demonstrates Python exponentiation operator (**) vs pow() function.
# ==============================================================================

# Step 1: Accept numeric input from user
# float() allows processing both integer and decimal inputs
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
num = float(input("Enter a number: "))

# Step 2: Compute square (num^2) and cube (num^3)
# Operator ** performs exponentiation (num ** 2 is num squared)
square = num ** 2
cube = num ** 3

# Step 3: Display results
print(f"\nNumber : {num}")
print(f"Square : {round(square, 2)}")
print(f"Cube   : {round(cube, 2)}")
