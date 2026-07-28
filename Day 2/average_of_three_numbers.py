# ==============================================================================
# Program    : Average of Three Numbers
# Objective  : Compute the arithmetic mean of three numeric inputs.
# Why Used   : Illustrates operator precedence — parentheses () are necessary to ensure 
#              addition occurs BEFORE division.
# ==============================================================================

# Step 1: Accept three float numbers
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
num1 = float(input("Enter first number: "))
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
num2 = float(input("Enter second number: "))
# What is used : Built-in input() function
# Why it is used: Pauses program execution to collect user input as string
num3 = float(input("Enter third number: "))

# Step 2: Compute average
# Parentheses force addition of all 3 numbers before dividing by 3
average = (num1 + num2 + num3) / 3

# Step 3: Display result
print(f"\nThe average of {num1}, {num2}, and {num3} is: {round(average, 2)}")
