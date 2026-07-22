# Program: Check Divisibility by both 5 and 11
# Concept: Modulus Operator (%), Logical 'and' Operator, and conditional statements

# Step 1: Accept an integer number from the user
num = int(input("Enter number: "))

# Step 2: Use modulus (%) to check if remainder is 0 when divided by 5 and 11.
# Both conditions must be True for the overall expression to be True.
if num % 5 == 0 and num % 11 == 0:
    print("Number is divisible by both 5 and 11!!")
else:
    print("Number is not divisible by both 5 and 11!!")